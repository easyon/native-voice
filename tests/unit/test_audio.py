from __future__ import annotations

import numpy as np

from native_voice.audio.encode import encode_audio, wav_header
from native_voice.audio.pcm import float32_to_pcm16, pcm16_to_float32
from native_voice.audio.resample import resample


def test_pcm_roundtrip():
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    data = float32_to_pcm16(audio)
    back = pcm16_to_float32(data)
    assert len(data) == len(audio) * 2
    assert np.allclose(back, audio, atol=1 / 32767)


def test_resample_length():
    x = np.zeros(16000, dtype=np.float32)
    assert len(resample(x, 16000, 24000)) == 24000
    assert len(resample(x, 16000, 8000)) == 8000


def test_encode_pcm_and_wav():
    x = np.zeros(2400, dtype=np.float32)
    assert len(encode_audio(x, 24000, "pcm", 24000)) == 4800
    wav = encode_audio(x, 24000, "wav", 24000)
    assert wav.startswith(b"RIFF")
    assert len(wav) == 44 + 4800
    assert wav_header(4800, 24000).startswith(b"RIFF")
