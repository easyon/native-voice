"""Per-utterance streaming ASR state (shared model, independent cache)."""
from __future__ import annotations

import numpy as np


class StreamingSession:
    def __init__(self, engine):
        self.engine = engine
        self.cache: dict = {}
        self.buf = np.zeros(0, dtype=np.float32)
        self.parts: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def reset(self, seed: np.ndarray | None = None) -> None:
        self.cache = {}
        self.parts = []
        self.buf = np.zeros(0, dtype=np.float32)
        if seed is not None and len(seed):
            self.buf = np.asarray(seed, dtype=np.float32).reshape(-1)
            self._drain()

    def _append(self, piece: str) -> None:
        if piece:
            self.parts.append(piece)

    def _drain(self) -> None:
        chunk = self.engine.chunk_samples
        while len(self.buf) >= chunk:
            piece = self.buf[:chunk]
            self.buf = self.buf[chunk:]
            self._append(self.engine.generate(piece, final=False, cache=self.cache))

    def feed(self, audio: np.ndarray) -> str:
        self.buf = np.concatenate([self.buf, np.asarray(audio, dtype=np.float32).reshape(-1)])
        self._drain()
        return self.text

    def finalize(self) -> str:
        if len(self.buf) > 0:
            self._append(self.engine.generate(self.buf, final=True, cache=self.cache))
            self.buf = np.zeros(0, dtype=np.float32)
        else:
            silence = np.zeros(int(0.2 * self.engine.cfg.rate), dtype=np.float32)
            self._append(self.engine.generate(silence, final=True, cache=self.cache))
        return self.text
