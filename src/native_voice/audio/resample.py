"""Sample-rate conversion helpers."""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or audio.size == 0:
        return audio
    g = gcd(int(src_rate), int(dst_rate))
    return resample_poly(audio, int(dst_rate) // g, int(src_rate) // g).astype(np.float32)
