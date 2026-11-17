"""Qwen3-ASR-0.6B runtime.

The ``qwen-asr`` transformers backend is offline: it transcribes a complete
audio buffer. The vLLM backend has true streaming but is Linux-only for now.
To keep the existing live-transcript UX on Windows, each session re-transcribes
the accumulated utterance in a background thread at a throttled interval and
returns the latest completed text from ``feed()``. ``finalize()`` always runs a
full pass over the utterance, so the text sent to the conversation engine is the
accurate one.

Third-party imports stay inside :meth:`load`; this module is importable from the
CPU-only gateway venv for configuration/tests.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Qwen3ASRConfig:
    model: str = "Qwen/Qwen3-ASR-0.6B"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    language: str | None = None       # None = automatic language detection
    partial_interval_ms: int = 800
    max_partial_seconds: float = 8.0  # stop re-transcribing long buffers
    max_new_tokens: int = 512
    sample_rate: int = 16000
    warmup: bool = True


def _torch_dtype(torch, name: str):
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(name).lower(), torch.bfloat16)


class Qwen3ASR:
    def __init__(self, cfg: Qwen3ASRConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.model = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        log.info("loading Qwen3-ASR from %s (%s, %s)", self.cfg.model, self.cfg.device, self.cfg.dtype)
        self.model = Qwen3ASRModel.from_pretrained(
            self.cfg.model,
            dtype=_torch_dtype(torch, self.cfg.dtype),
            device_map=self.cfg.device,
            max_inference_batch_size=1,
            max_new_tokens=self.cfg.max_new_tokens,
        )
        if self.cfg.warmup:
            try:
                self.transcribe(np.zeros(int(self.cfg.sample_rate * 0.3), dtype=np.float32))
            except Exception as exc:
                log.warning("Qwen3-ASR warmup failed: %r", exc)
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            log.info("Qwen3-ASR ready (VRAM %.2f/%.2f GB)", alloc, peak)
        self._ready = True

    def transcribe(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or self.model is None:
            return ""
        with self.lock:
            results = self.model.transcribe(
                audio=(audio, int(self.cfg.sample_rate)),
                language=self.cfg.language,
            )
        if not results:
            return ""
        return str(getattr(results[0], "text", "") or "").strip()

    def create_session(self) -> Qwen3ASRSession:
        return Qwen3ASRSession(self)

    def close(self) -> None:
        with self.lock:
            self.model = None
            self._ready = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class Qwen3ASRSession:
    """Per-utterance buffer with throttled background partial transcription."""

    def __init__(self, engine: Qwen3ASR):
        self.engine = engine
        self._lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._samples = 0
        self._text = ""
        self._last_infer_samples = 0
        self._gen = 0
        self._thread: threading.Thread | None = None

    @property
    def _interval_samples(self) -> int:
        return max(1, int(self.engine.cfg.sample_rate * self.engine.cfg.partial_interval_ms / 1000))

    @property
    def _max_partial_samples(self) -> int:
        return int(self.engine.cfg.sample_rate * self.engine.cfg.max_partial_seconds)

    def reset(self, seed: np.ndarray | None = None) -> None:
        with self._lock:
            self._buffer.clear()
            self._samples = 0
            self._text = ""
            self._last_infer_samples = 0
            self._gen += 1
        if seed is not None and len(seed):
            self.feed(np.asarray(seed, dtype=np.float32))

    def feed(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return self._text
        with self._lock:
            self._buffer.append(audio)
            self._samples += audio.size
            busy = self._thread is not None and self._thread.is_alive()
            should = (
                not busy
                and self._samples <= self._max_partial_samples
                and self._samples - self._last_infer_samples >= self._interval_samples
            )
            if should:
                self._last_infer_samples = self._samples
                snapshot = np.concatenate(self._buffer)
                gen = self._gen
                self._thread = threading.Thread(
                    target=self._infer, args=(snapshot, gen), daemon=True, name="qwen3-asr-partial"
                )
                self._thread.start()
            return self._text

    def _infer(self, audio: np.ndarray, gen: int) -> None:
        try:
            text = self.engine.transcribe(audio)
        except Exception as exc:  # partials are best-effort
            log.debug("Qwen3-ASR partial failed: %r", exc)
            return
        with self._lock:
            if gen == self._gen:
                self._text = text

    def finalize(self) -> str:
        with self._lock:
            self._gen += 1
            audio = np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.float32)
            cached = self._text
        if audio.size == 0:
            return cached
        text = self.engine.transcribe(audio)
        with self._lock:
            self._text = text
        return text

    def close(self) -> None:
        with self._lock:
            self._gen += 1
            self._buffer.clear()
            self._thread = None
