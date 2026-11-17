"""Output audio encoding: raw PCM16, WAV, MP3/Opus via ffmpeg.

PCM is the streaming-friendly format.  Compressed formats buffer one sentence
at a time because their container/encoder needs a complete frame.
"""
from __future__ import annotations

import shutil
import struct
import subprocess

import numpy as np

from .pcm import float32_to_pcm16
from .resample import resample


class AudioFormatError(ValueError):
    pass


def is_streamable(fmt: str) -> bool:
    return (fmt or "").lower() == "pcm"


def wav_header(data_len: int, sample_rate: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    byte_rate = sample_rate * channels * sampwidth
    block_align = channels * sampwidth
    return b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, sampwidth * 8
    ) + b"data" + struct.pack("<I", data_len)


def _ffmpeg_encode(pcm: bytes, rate: int, fmt: str) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioFormatError(f"ffmpeg is required for {fmt} output")
    if fmt == "mp3":
        args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(rate), "-ac", "1",
                "-i", "pipe:0", "-f", "mp3", "-b:a", "32k", "pipe:1"]
    elif fmt == "opus":
        args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(rate), "-ac", "1",
                "-i", "pipe:0", "-c:a", "libopus", "-f", "ogg", "pipe:1"]
    else:
        raise AudioFormatError(f"unsupported format: {fmt}")
    try:
        proc = subprocess.run(args, input=pcm, capture_output=True, timeout=15)
    except subprocess.TimeoutExpired as exc:
        raise AudioFormatError("ffmpeg timeout") from exc
    if proc.returncode != 0:
        raise AudioFormatError(proc.stderr.decode("utf-8", "replace")[:300] or "ffmpeg failed")
    return proc.stdout


def encode_audio(
    audio: np.ndarray,
    src_rate: int,
    fmt: str,
    out_rate: int,
    volume: float = 1.0,
) -> bytes:
    fmt = (fmt or "pcm").lower()
    audio = resample(np.asarray(audio, dtype=np.float32).reshape(-1), src_rate, out_rate)
    if volume != 1.0:
        audio = np.clip(audio * float(volume), -1.0, 1.0)
    if fmt == "pcm":
        return float32_to_pcm16(audio)
    if fmt == "wav":
        data = float32_to_pcm16(audio)
        return wav_header(len(data), out_rate) + data
    if fmt in ("mp3", "opus"):
        raw = audio.astype("<f4").tobytes()
        return _ffmpeg_encode(raw, out_rate, fmt)
    raise AudioFormatError(f"unsupported format: {fmt}")
