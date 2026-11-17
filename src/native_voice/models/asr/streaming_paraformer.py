"""Shared streaming Paraformer runtime.

The FunASR model object mutates shared kwargs and calls ``torch.cuda.empty_cache``
after each generate, so all inference on one loaded model goes through a lock.
Per-utterance state lives in ``StreamingSession``.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from .config import ASRModelConfig
from .streaming_session import StreamingSession

log = logging.getLogger(__name__)


class StreamingParaformer:
    def __init__(self, cfg: ASRModelConfig):
        self.cfg = cfg
        self.chunk_ms = max(240, int(round(cfg.chunk_ms / 60.0)) * 60)
        n = self.chunk_ms // 60
        self.chunk_size = [0, n, max(1, n // 2)]
        self.chunk_samples = n * 960
        self.lock = threading.Lock()
        self.model = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        from funasr import AutoModel

        model = self.cfg.model
        local = Path.home() / ".cache" / "modelscope" / "hub" / "models" / model
        if local.exists():
            model = str(local)
        device = self.cfg.device
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"
        log.info("loading streaming Paraformer (%s, %sms chunks)", device, self.chunk_ms)
        self.model = AutoModel(
            model=model,
            device=device,
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
        )
        try:
            self.generate(np.zeros(self.chunk_samples, dtype=np.float32), final=False, cache={})
        except Exception as exc:  # pragma: no cover
            log.warning("ASR warmup failed: %r", exc)
        self._ready = True
        log.info("streaming Paraformer ready")

    def generate(self, audio: np.ndarray, final: bool, cache: dict) -> str:
        with self.lock:
            result = self.model.generate(
                input=np.asarray(audio, dtype=np.float32),
                cache=cache,
                is_final=final,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
            )
        if not result:
            return ""
        return str(result[0].get("text", ""))

    def create_session(self) -> StreamingSession:
        return StreamingSession(self)

    def close(self) -> None:
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
