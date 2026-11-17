"""PCM16 little-endian <-> float32 mono."""
from __future__ import annotations

import numpy as np


def pcm16_to_float32(data: bytes) -> np.ndarray:
    usable = len(data) - (len(data) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
