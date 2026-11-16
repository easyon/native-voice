from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import logging
import re
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cosyvoice_common.errors import service_unavailable

from ..core.audio import PCM_BYTES_PER_MS, pcm_duration_ms
from ..core.vad import (
    EnergyVadStream,
    PassthroughVadStream,
    SpeechEnd,
    SpeechStart,
    VadStream,
)
from ..schemas import Final, Partial, StreamParams, Transcript, TranscriptSegment, Word
from ..settings import EngineConfig, ModelConfig
from .base import ASREngine, ASRStream

logger = logging.getLogger("funasr_runtime.engine.funasr")

_TAG_RE = re.compile(r"<\|[^|]*\|>")
_NANO_LANGUAGE = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "auto": "",
}
_SENSEVOICE_LANGUAGE = {
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "yue": "yue",
    "auto": "auto",
}


@dataclass(slots=True)
class _LoadedModel:
    model: Any
    lock: threading.Lock


class FunasrEngine(ASREngine):
    """In-process FunASR ``AutoModel`` adapter.

    The realtime path uses fsmn-vad for endpointing and re-decodes a bounded
    window for interim text. This matches the official Fun-ASR realtime service
    for offline checkpoints (Nano / SenseVoice / Paraformer), which have no
    per-chunk cache API.
    """

    def __init__(
        self,
        config: EngineConfig,
        models: dict[str, ModelConfig],
    ) -> None:
        self._cfg = config.funasr
        self._vad_cfg = config.vad
        self._models = models
        self._loaded: dict[str, _LoadedModel] = {}
        self._vad_loaded: dict[str, Any] = {}
        # The VAD AutoModel carries mutable vad_opts; serialize all sessions
        # through one lock, matching FunASR's official realtime service.
        self._vad_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(self._cfg.executor_threads)),
            thread_name_prefix="funasr-model",
        )
        self._device = "cpu"
        self._closed = False

    async def startup(self) -> None:
        self._device = self._resolve_device()
        await self._run(self._load_all_sync)

    async def shutdown(self) -> None:
        self._closed = True
        self._loaded.clear()
        self._vad_loaded.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def ready(self) -> bool:
        return not self._closed and bool(self._loaded)

    async def warmup(self) -> None:
        if not self._cfg.warmup or not self._loaded:
            return
        silence = np.zeros(8_000, dtype=np.float32)
        name = next(iter(self._loaded))
        model = self._models[name]
        try:
            await self._decode(name, model, silence, _params_for_warmup())
        except Exception:  # noqa: BLE001 - startup must fail loudly
            logger.exception("FunASR warmup failed for model %s", name)
            raise
        vad = self._vad_for(model)
        if vad is not None and self._vad_cfg.type == "funasr":
            await self._run(self._warmup_vad_sync, vad)

    def create_stream(self, model: ModelConfig, params: StreamParams) -> ASRStream:
        name = _model_name(self._models, model)
        return FunasrASRStream(self, name, model, params)

    def create_vad_stream(
        self, model: ModelConfig, params: StreamParams
    ) -> VadStream:
        if self._vad_cfg.type == "passthrough":
            return PassthroughVadStream()
        if self._vad_cfg.type == "funasr":
            vad_model = self._vad_for(model)
            if vad_model is not None:
                try:
                    return FunasrVadStream(self, vad_model, params)
                except Exception:  # noqa: BLE001 - fall back to energy VAD
                    logger.exception("FunASR VAD could not be created; using energy VAD")
        return EnergyVadStream(
            sample_rate=params.sample_rate,
            threshold_db=self._vad_cfg.energy_threshold_db,
            frame_ms=self._vad_cfg.frame_ms,
            min_speech_ms=self._vad_cfg.min_speech_ms,
            silence_ms=params.max_sentence_silence_ms,
            max_segment_ms=params.max_segment_ms,
            pre_roll_ms=self._vad_cfg.pre_roll_ms,
        )

    async def transcribe(
        self, model: ModelConfig, pcm: bytes, params: StreamParams
    ) -> Transcript:
        name = _model_name(self._models, model)
        return await self._run(self._transcribe_sync, name, model, pcm, params)

    # -- internals --------------------------------------------------------
    def _resolve_device(self) -> str:
        device = (self._cfg.device or "auto").strip()
        if device != "auto":
            return device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 - torch is an optional extra
            return "cpu"

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        return await loop.run_in_executor(
            self._executor,
            context.run,
            lambda: fn(*args, **kwargs),
        )

    def _load_all_sync(self) -> None:
        try:
            from funasr import AutoModel
        except Exception as exc:  # noqa: BLE001
            raise service_unavailable(
                "funasr is not installed; install funasr-runtime[funasr]"
            ) from exc

        for name, model in self._models.items():
            artifact = self._resolve_artifact(model.artifact)
            kwargs: dict[str, Any] = {
                "model": artifact,
                "device": self._device,
                "disable_update": True,
                "hub": self._cfg.hub,
            }
            if self._cfg.trust_remote_code and (
                model.architecture == "fun-asr-nano"
                or Path(artifact, "model.py").is_file()
            ):
                kwargs["trust_remote_code"] = True
                if self._cfg.remote_code:
                    kwargs["remote_code"] = self._cfg.remote_code
            logger.info(
                "loading FunASR model %s: artifact=%s device=%s architecture=%s",
                name,
                artifact,
                self._device,
                model.architecture,
            )
            loaded = AutoModel(**kwargs)
            self._loaded[name] = _LoadedModel(model=loaded, lock=threading.Lock())
            logger.info("loaded FunASR model %s", name)

        for artifact in self._vad_artifacts():
            if artifact in self._vad_loaded:
                continue
            resolved = self._resolve_artifact(artifact, allow_builtin=True)
            logger.info("loading FunASR VAD: %s", resolved)
            self._vad_loaded[artifact] = AutoModel(
                model=resolved,
                device=self._cfg.vad_device,
                disable_update=True,
                hub=self._cfg.hub,
            )

    def _vad_artifacts(self) -> Iterable[str]:
        seen = set()
        if self._cfg.vad_model:
            seen.add(self._cfg.vad_model)
        for model in self._models.values():
            if model.vad_artifact:
                seen.add(model.vad_artifact)
        if self._vad_cfg.type == "funasr" and not seen and self._cfg.allow_download:
            seen.add("fsmn-vad")
        return seen

    def _vad_for(self, model: ModelConfig) -> Any | None:
        artifact = model.vad_artifact or self._cfg.vad_model
        if not artifact:
            if "fsmn-vad" in self._vad_loaded:
                return self._vad_loaded["fsmn-vad"]
            return None
        return self._vad_loaded.get(artifact)

    def _resolve_artifact(self, artifact: str, *, allow_builtin: bool = False) -> str:
        raw = (artifact or "").strip()
        if not raw:
            raise service_unavailable("empty FunASR artifact path")
        path = Path(raw).expanduser()
        if path.exists():
            return str(path)
        candidate = Path(self._cfg.model_root).expanduser() / path.name
        if candidate.exists():
            return str(candidate)
        if self._cfg.allow_download and not allow_builtin:
            return raw
        if allow_builtin and raw == "fsmn-vad" and self._cfg.allow_download:
            return raw
        raise service_unavailable(
            f"FunASR model artifact not found: {raw}; "
            f"mount it under {self._cfg.model_root} or set allow_download=true"
        )

    def _warmup_vad_sync(self, vad_model: Any) -> None:
        import torch
        from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

        vad = DynamicStreamingVAD(vad_model, sample_rate=16_000)
        with self._vad_lock:
            vad.feed(torch.zeros(3_200, dtype=torch.float32), is_final=True)

    async def _decode(
        self,
        name: str,
        model: ModelConfig,
        audio: np.ndarray,
        params: StreamParams,
    ) -> tuple[str, tuple[Word, ...]]:
        loaded = self._loaded.get(name)
        if loaded is None:
            raise service_unavailable(f"FunASR model is not loaded: {name}")
        kwargs = self._generate_kwargs(model, params, loaded.model)
        return await self._run(self._decode_sync, loaded, audio, kwargs)

    def _decode_sync(
        self,
        loaded: _LoadedModel,
        audio: np.ndarray,
        kwargs: dict[str, Any],
    ) -> tuple[str, tuple[Word, ...]]:
        with loaded.lock:
            with self._inference_mode():
                results = loaded.model.generate(input=[audio], **kwargs)
        return _extract_result(results)

    @staticmethod
    def _inference_mode():
        try:
            import torch

            return torch.inference_mode()
        except Exception:  # noqa: BLE001
            return contextlib.nullcontext()

    def _generate_kwargs(
        self, model: ModelConfig, params: StreamParams, target: Any = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"cache": {}, "batch_size": 1}
        kwargs.update(self._cfg.inference_args)
        kwargs.update(model.inference_args)

        architecture = model.architecture
        hotwords = _hotword_list(params.hotwords)
        if architecture == "fun-asr-nano":
            language = _NANO_LANGUAGE.get(params.language, params.language)
            if language and language != "auto":
                kwargs["language"] = language
            if hotwords:
                kwargs["hotwords"] = hotwords
            kwargs.setdefault("itn", params.itn)
        elif architecture == "sensevoice":
            kwargs["language"] = _SENSEVOICE_LANGUAGE.get(
                params.language, params.language
            )
            kwargs.setdefault("use_itn", params.itn)
        elif architecture == "paraformer":
            if hotwords:
                kwargs.setdefault("hotword", " ".join(hotwords))
            kwargs.setdefault("use_itn", params.itn)
            if params.timestamps:
                kwargs.setdefault("sentence_timestamp", True)
        else:
            if hotwords:
                kwargs.setdefault("hotwords", hotwords)
            kwargs.setdefault("use_itn", params.itn)

        return _filter_kwargs(target, kwargs)

    def _transcribe_sync(
        self,
        name: str,
        model: ModelConfig,
        pcm: bytes,
        params: StreamParams,
    ) -> Transcript:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        duration_ms = pcm_duration_ms(pcm)
        segments = self._segment_offline(model, audio, params)
        parts: list[TranscriptSegment] = []
        loaded = self._loaded.get(name)
        if loaded is None:
            raise service_unavailable(f"FunASR model is not loaded: {name}")
        for begin_ms, end_ms in segments:
            for chunk_begin, chunk_end in _split_segment(
                begin_ms, end_ms, params.max_segment_ms
            ):
                start_sample = max(0, chunk_begin * 16)
                end_sample = min(len(audio), chunk_end * 16)
                chunk = audio[start_sample:end_sample]
                if chunk.size < 1_600:
                    continue
                text, words = self._decode_sync(
                    loaded,
                    chunk,
                    self._generate_kwargs(model, params, loaded.model),
                )
                if not text:
                    continue
                parts.append(
                    TranscriptSegment(
                        text=text,
                        begin_time=chunk_begin,
                        end_time=chunk_end,
                        words=list(_shift_words(words, chunk_begin)),
                    )
                )
        text = "".join(segment.text for segment in parts).strip()
        return Transcript(
            text=text,
            language=params.language or None,
            duration_ms=duration_ms,
            segments=parts,
        )

    def _segment_offline(
        self, model: ModelConfig, audio: np.ndarray, params: StreamParams
    ) -> list[tuple[int, int]]:
        duration_ms = int(len(audio) * 1000 / params.sample_rate)
        vad_model = self._vad_for(model)
        if vad_model is None:
            return [(0, duration_ms)] if duration_ms > 0 else []
        try:
            import torch
            from funasr.models.fsmn_vad_streaming.dynamic_vad import (
                DynamicStreamingVAD,
            )

            dynamic = DynamicStreamingVAD(
                vad_model,
                sample_rate=params.sample_rate,
                speech_noise_thres=params.speech_noise_threshold,
                silence_schedule=[(float("inf"), params.max_sentence_silence_ms)],
            )
            with self._vad_lock:
                segments = dynamic.process(torch.from_numpy(audio).float())
            return [(int(seg[0]), int(seg[1])) for seg in segments]
        except Exception:  # noqa: BLE001 - VAD is an optimization for offline
            logger.exception("offline VAD failed; decoding the full audio")
            return [(0, duration_ms)] if duration_ms > 0 else []


class FunasrASRStream(ASRStream):
    def __init__(
        self,
        engine: FunasrEngine,
        model_name: str,
        model: ModelConfig,
        params: StreamParams,
    ) -> None:
        self._engine = engine
        self._model_name = model_name
        self._model = model
        self._params = params
        self._buffer = bytearray()
        self._last_partial_ms = 0
        self._last_text = ""
        self._partials_enabled = params.partial_interval_ms > 0

    async def feed(self, pcm: bytes) -> list[Partial]:
        self._buffer.extend(pcm)
        if not self._partials_enabled:
            return []
        duration_ms = len(self._buffer) // PCM_BYTES_PER_MS
        if duration_ms - self._last_partial_ms < self._params.partial_interval_ms:
            return []
        if duration_ms < 200:
            return []
        self._last_partial_ms = duration_ms
        if (
            self._params.partial_window_s > 0
            and len(self._buffer)
            > int(self._params.partial_window_s * self._params.sample_rate * 2)
        ):
            # Stop emitting partials once the segment is longer than the
            # window instead of returning a tail-only text that would look
            # like a truncated sentence to the client. The final still decodes
            # the complete segment.
            return []
        window = np.frombuffer(
            bytes(self._buffer), dtype="<i2"
        ).astype(np.float32) / 32768.0
        text, _ = await self._engine._decode(
            self._model_name, self._model, window, self._params
        )
        text = text.strip()
        if not text or text == self._last_text:
            return []
        self._last_text = text
        begin = int(self._params.extra.get("begin_time_ms", 0))
        return [Partial(text=text, begin_time=begin, end_time=None)]

    async def finish(self) -> Final | None:
        if not self._buffer:
            return None
        audio = np.frombuffer(bytes(self._buffer), dtype="<i2").astype(
            np.float32
        ) / 32768.0
        text, words = await self._engine._decode(
            self._model_name, self._model, audio, self._params
        )
        duration_ms = len(self._buffer) // PCM_BYTES_PER_MS
        begin = int(self._params.extra.get("begin_time_ms", 0))
        return Final(
            text=text.strip(),
            begin_time=begin,
            end_time=begin + duration_ms,
            words=_shift_words(words, begin),
        )


class FunasrVadStream(VadStream):
    def __init__(
        self,
        engine: FunasrEngine,
        vad_model: Any,
        params: StreamParams,
    ) -> None:
        from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

        self._engine = engine
        self._vad = DynamicStreamingVAD(
            vad_model,
            sample_rate=params.sample_rate,
            speech_noise_thres=params.speech_noise_threshold,
            silence_schedule=[(float("inf"), params.max_sentence_silence_ms)],
        )
        self._fed_ms = 0
        self._speaking = False

    async def feed(self, pcm: bytes) -> list[SpeechEnd | SpeechStart]:
        return await self._engine._run(self._feed_sync, pcm)

    async def finish(self) -> list[SpeechEnd | SpeechStart]:
        return await self._engine._run(self._finish_sync)

    def _feed_sync(self, pcm: bytes) -> list[SpeechEnd | SpeechStart]:
        import torch

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        with self._engine._vad_lock:
            segments = self._vad.feed(torch.from_numpy(audio), is_final=False)
        self._fed_ms += len(pcm) // PCM_BYTES_PER_MS
        events: list[SpeechEnd | SpeechStart] = []
        # Emit a start before every confirmed segment. If the session already
        # has the segment open it ignores the duplicate start; otherwise this
        # recovers a short utterance that started and ended inside one chunk.
        for segment in segments:
            events.append(SpeechStart(int(segment[0])))
            events.append(SpeechEnd(int(segment[1])))
        current = self._vad.current_speech_start
        if current is not None and (not segments or int(current) > int(segments[-1][1])):
            events.append(SpeechStart(int(current)))
        self._speaking = current is not None
        return events

    def _finish_sync(self) -> list[SpeechEnd | SpeechStart]:
        with self._engine._vad_lock:
            segments = self._vad.finalize()
        events: list[SpeechEnd | SpeechStart] = []
        for segment in segments:
            events.append(SpeechStart(int(segment[0])))
            events.append(SpeechEnd(int(segment[1])))
        if self._vad.current_speech_start is not None:
            events.append(SpeechStart(int(self._vad.current_speech_start)))
            events.append(SpeechEnd(self._fed_ms, "flush"))
            self._vad.current_speech_start = None
        self._speaking = False
        return events


def _model_name(models: dict[str, ModelConfig], model: ModelConfig) -> str:
    for name, candidate in models.items():
        if candidate is model:
            return name
    raise service_unavailable("model is not registered with the FunASR engine")


def _filter_kwargs(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    if target is None:
        return kwargs
    try:
        signature = inspect.signature(target.generate)
    except (AttributeError, TypeError, ValueError):
        return kwargs
    parameters = signature.parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _extract_result(results: Any) -> tuple[str, tuple[Word, ...]]:
    if not results:
        return "", ()
    first = results[0] if isinstance(results, list) else results
    if not isinstance(first, dict):
        return "", ()
    text = _clean_text(str(first.get("text") or ""))
    if not text:
        return "", ()
    stamps = first.get("timestamp") or first.get("timestamps")
    words: tuple[Word, ...] = ()
    if isinstance(stamps, list) and len(stamps) == len(text):
        parsed = []
        for char, stamp in zip(text, stamps, strict=False):
            if not isinstance(stamp, (list, tuple)) or len(stamp) < 2:
                parsed = []
                break
            try:
                begin, end = int(stamp[0]), int(stamp[1])
            except (TypeError, ValueError):
                parsed = []
                break
            parsed.append(
                Word(
                    begin_time=begin,
                    end_time=end,
                    text=char,
                    punctuation=char if char in "，。！？、,.!?" else "",
                )
            )
        if parsed:
            words = tuple(parsed)
    return text, words


def _clean_text(text: str) -> str:
    return _TAG_RE.sub("", text).strip()


def _shift_words(words: tuple[Word, ...], offset_ms: int) -> tuple[Word, ...]:
    if not words or not offset_ms:
        return words
    return tuple(
        Word(
            begin_time=word.begin_time + offset_ms,
            end_time=word.end_time + offset_ms,
            text=word.text,
            punctuation=word.punctuation,
        )
        for word in words
    )


def _hotword_list(value: str) -> list[str]:
    if not value:
        return []
    return [
        part.strip()
        for part in re.split(r"[,，、;；\n\s]+", value)
        if part.strip()
    ]


def _split_segment(
    begin_ms: int, end_ms: int, max_segment_ms: int
) -> list[tuple[int, int]]:
    if end_ms <= begin_ms:
        return []
    if max_segment_ms <= 0 or end_ms - begin_ms <= max_segment_ms:
        return [(begin_ms, end_ms)]
    chunks = []
    cursor = begin_ms
    while cursor < end_ms:
        stop = min(end_ms, cursor + max_segment_ms)
        chunks.append((cursor, stop))
        cursor = stop
    return chunks


def _params_for_warmup() -> StreamParams:
    return StreamParams(partial_interval_ms=0, partial_window_s=0.0)
