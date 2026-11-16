from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cosyvoice_common.errors import (
    internal_error,
    service_unavailable,
    timeout_error,
)

from ..core.audio import PCM_BYTES_PER_MS, pcm_duration_ms
from ..core.vad import EnergyVadStream, PassthroughVadStream, VadStream
from ..schemas import Final, Partial, StreamParams, Transcript, TranscriptSegment
from ..settings import EngineConfig, ModelConfig
from .base import ASREngine, ASRStream

logger = logging.getLogger("funasr_runtime.engine.llamacpp")

_DEFAULT_BINARIES = {
    "fun-asr-nano": "llama-funasr-cli",
    "sensevoice": "llama-funasr-sensevoice",
    "paraformer": "llama-funasr-paraformer",
}
_SRT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


class LlamaCppEngine(ASREngine):
    """GGUF/llama.cpp adapter.

    The upstream binaries are single-shot file transcribers, so realtime mode
    uses an energy VAD and decodes completed segments in a subprocess. Interim
    partials are disabled by default: every decode pays process startup and
    model load. Offline transcription is the primary use of this engine.
    """

    def __init__(
        self,
        config: EngineConfig,
        models: dict[str, ModelConfig],
        *,
        runner: Callable[[list[str], float], subprocess.CompletedProcess] | None = None,
    ) -> None:
        self._cfg = config.llamacpp
        self._vad_cfg = config.vad
        self._models = models
        self._runner = runner
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(self._cfg.max_processes)),
            thread_name_prefix="funasr-llamacpp",
        )
        self._closed = False

    async def startup(self) -> None:
        if self._runner is not None:
            return
        await self._run(self._check_models_sync)

    def _check_models_sync(self) -> None:
        for name, model in self._models.items():
            binary = self._binary_for(model)
            if not _executable(binary):
                raise service_unavailable(
                    f"llama.cpp binary not found: {binary}; "
                    "mount the funasr-llamacpp release or set engine.llamacpp.binary_dir"
                )
            if model.architecture == "fun-asr-nano" and not model.encoder_artifact:
                raise service_unavailable(
                    f"models.{name}.encoder_artifact is required for fun-asr-nano GGUF"
                )
            if (
                model.architecture == "fun-asr-nano"
                and not Path(model.encoder_artifact).expanduser().is_file()
            ):
                raise service_unavailable(
                    f"GGUF encoder not found: {model.encoder_artifact}"
                )
            if not Path(model.artifact).expanduser().is_file():
                raise service_unavailable(
                    f"GGUF model not found: {model.artifact}"
                )
            if model.vad_artifact and not Path(
                model.vad_artifact
            ).expanduser().is_file():
                raise service_unavailable(
                    f"GGUF VAD model not found: {model.vad_artifact}"
                )

    async def shutdown(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def ready(self) -> bool:
        return not self._closed

    async def warmup(self) -> None:
        return None

    def create_stream(self, model: ModelConfig, params: StreamParams) -> ASRStream:
        return LlamaCppASRStream(self, model, params)

    def create_vad_stream(
        self, model: ModelConfig, params: StreamParams
    ) -> VadStream:
        if self._vad_cfg.type == "passthrough":
            return PassthroughVadStream()
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
        return await self._run(self._transcribe_sync, model, pcm, params, True)

    async def _run(self, fn, *args):
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        return await loop.run_in_executor(
            self._executor,
            context.run,
            lambda: fn(*args),
        )

    def _transcribe_sync(
        self,
        model: ModelConfig,
        pcm: bytes,
        params: StreamParams,
        use_vad: bool,
    ) -> Transcript:
        duration_ms = pcm_duration_ms(pcm)
        if duration_ms < 100:
            return Transcript(
                text="",
                language=params.language or None,
                duration_ms=duration_ms,
            )

        wav_path = _write_temp_wav(pcm)
        try:
            command = self._build_command(model, wav_path, use_vad, params.timestamps)
            process = self._run_process(command, self._cfg.timeout_s)
            if process.returncode != 0:
                detail = (process.stderr or "").strip().splitlines()
                message = detail[-1] if detail else f"exit code {process.returncode}"
                raise internal_error(f"llama.cpp failed: {message[:300]}")
            stdout = (process.stdout or "").strip()
            segments = _parse_srt(stdout) if params.timestamps else []
            if segments:
                text = "".join(segment.text for segment in segments)
            else:
                text = stdout
                segments = [
                    TranscriptSegment(
                        text=text,
                        begin_time=0,
                        end_time=duration_ms,
                    )
                ] if text else []
            return Transcript(
                text=text.strip(),
                language=params.language or None,
                duration_ms=duration_ms,
                segments=segments,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(wav_path)

    def _run_process(
        self, command: list[str], timeout_s: float
    ) -> subprocess.CompletedProcess:
        try:
            if self._runner is not None:
                return self._runner(command, timeout_s)
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise timeout_error("llama.cpp transcription timed out") from exc
        except OSError as exc:
            raise service_unavailable(f"llama.cpp could not start: {exc}") from exc

    def _binary_for(self, model: ModelConfig) -> str:
        configured = self._cfg.binaries.get(model.architecture)
        name = configured or _DEFAULT_BINARIES.get(
            model.architecture, "llama-funasr-cli"
        )
        path = Path(name)
        if path.is_absolute():
            return str(path)
        candidate = Path(self._cfg.binary_dir) / name
        if candidate.exists():
            return str(candidate)
        found = shutil.which(name)
        return found or str(candidate)

    def _build_command(
        self,
        model: ModelConfig,
        wav_path: str,
        use_vad: bool,
        timestamps: bool,
    ) -> list[str]:
        binary = self._binary_for(model)
        values = {
            "binary": binary,
            "model": model.artifact,
            "encoder": model.encoder_artifact,
            "vad": model.vad_artifact,
            "audio": wav_path,
        }
        template = list(self._cfg.args)
        if not template or "{binary}" not in template[0]:
            template.insert(0, "{binary}")
        if model.architecture == "fun-asr-nano" and not any(
            "{encoder}" in token for token in template
        ):
            index = template.index("{binary}") + 1
            template[index:index] = ["--enc", "{encoder}"]
        command = [token.format(**values) for token in template]
        if model.vad_artifact and use_vad and "--vad" not in command:
            command.extend(["--vad", model.vad_artifact])
        if timestamps and "--srt" not in command:
            command.append("--srt")
        command.extend(model.args)
        return command


class LlamaCppASRStream(ASRStream):
    def __init__(
        self,
        engine: LlamaCppEngine,
        model: ModelConfig,
        params: StreamParams,
    ) -> None:
        self._engine = engine
        self._model = model
        self._params = params
        self._buffer = bytearray()
        self._last_partial_ms = 0
        self._decode_seconds = 0.0
        self._partials_enabled = params.partial_interval_ms > 0

    async def feed(self, pcm: bytes) -> list[Partial]:
        self._buffer.extend(pcm)
        if not self._partials_enabled:
            return []
        duration_ms = len(self._buffer) // PCM_BYTES_PER_MS
        if duration_ms - self._last_partial_ms < self._params.partial_interval_ms:
            return []
        if duration_ms < 500:
            return []
        self._last_partial_ms = duration_ms
        if (
            self._params.partial_window_s > 0
            and len(self._buffer)
            > int(self._params.partial_window_s * self._params.sample_rate * 2)
        ):
            # A tail-only preview would look like a truncated sentence. Stop
            # partials after the window; the final decodes the full segment.
            return []
        window = bytes(self._buffer)
        started = time.monotonic()
        result = await self._engine._run(
            self._engine._transcribe_sync,
            self._model,
            window,
            self._params,
            False,
        )
        self._decode_seconds += time.monotonic() - started
        text = result.text.strip()
        if not text:
            return []
        begin = int(self._params.extra.get("begin_time_ms", 0))
        return [Partial(text=text, begin_time=begin, end_time=None)]

    async def finish(self) -> Final | None:
        if not self._buffer:
            return None
        started = time.monotonic()
        result = await self._engine._run(
            self._engine._transcribe_sync,
            self._model,
            bytes(self._buffer),
            self._params,
            False,
        )
        self._decode_seconds += time.monotonic() - started
        begin = int(self._params.extra.get("begin_time_ms", 0))
        end = begin + len(self._buffer) // PCM_BYTES_PER_MS
        return Final(
            text=result.text.strip(),
            begin_time=begin,
            end_time=end,
            decode_seconds=self._decode_seconds,
        )


def _write_temp_wav(pcm: bytes) -> str:
    fd, path = tempfile.mkstemp(prefix="funasr-", suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as handle:
            with wave.open(handle, "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16_000)
                writer.writeframes(pcm)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise
    return path


def _parse_srt(text: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        match = None
        for index, line in enumerate(lines):
            match = _SRT_TIME.search(line)
            if match:
                body = " ".join(lines[index + 1 :]).strip()
                break
        if match is None or not body:
            continue
        begin = _srt_ms(
            match.group(1), match.group(2), match.group(3), match.group(4)
        )
        end = _srt_ms(
            match.group(5), match.group(6), match.group(7), match.group(8)
        )
        segments.append(
            TranscriptSegment(text=body, begin_time=begin, end_time=end)
        )
    return segments


def _srt_ms(hours: str, minutes: str, seconds: str, millis: str) -> int:
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis)
    )


def _executable(path: str) -> bool:
    candidate = Path(path)
    return candidate.is_file() and os.access(candidate, os.X_OK)
