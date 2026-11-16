from __future__ import annotations

import pytest
from cosyvoice_common.errors import APIError
from funasr_runtime.core.audio import WavStreamReader, decode_to_pcm16k

from tests.asr_runtime.conftest import tone_pcm, wav_bytes


def test_wav_header_can_span_frames() -> None:
    pcm = tone_pcm(200)
    wav = wav_bytes(pcm)
    reader = WavStreamReader()
    out = b""
    for start, stop in ((0, 10), (10, 30), (30, len(wav))):
        out += reader.feed(wav[start:stop])
    assert out == pcm
    assert reader.finish() == b""


def test_wav_rejects_stereo() -> None:
    import struct

    # Minimal fmt chunk declaring two channels.
    fmt = struct.pack("<HHIIHH", 1, 2, 16_000, 64_000, 4, 16)
    header = (
        b"RIFF"
        + struct.pack("<I", 36)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", 4)
    )
    reader = WavStreamReader()
    with pytest.raises(APIError, match="mono"):
        reader.feed(header + b"\x00\x00\x00\x00")


def test_pcm_passthrough_and_odd_length() -> None:
    assert decode_to_pcm16k(b"\x00\x00" * 10, "pcm") == b"\x00\x00" * 10
    with pytest.raises(APIError, match="16-bit"):
        decode_to_pcm16k(b"\x00", "pcm")
