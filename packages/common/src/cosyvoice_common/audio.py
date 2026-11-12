from __future__ import annotations

import array
import asyncio
import io
import shutil
import struct
import subprocess
import sys
import wave
from functools import lru_cache

from .errors import unsupported_format, unsupported_parameter

SUPPORTED_FORMATS = {"pcm", "wav", "mp3", "opus"}
SUPPORTED_SAMPLE_RATES = {8_000, 12_000, 16_000, 22_050, 24_000, 44_100, 48_000}

try:  # numpy is optional for the gateway, but present in the CosyVoice image.
    import numpy as _np
except Exception:  # pragma: no cover - depends on deployment image
    _np = None


def validate_format(fmt: str) -> str:
    fmt = (fmt or "wav").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise unsupported_format(
            f"unsupported format: {fmt}; supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    return fmt


def validate_sample_rate(sample_rate: int) -> int:
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise unsupported_parameter(
            f"unsupported sample_rate: {sample_rate}; "
            f"supported: {', '.join(str(x) for x in sorted(SUPPORTED_SAMPLE_RATES))}"
        )
    return sample_rate


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int,
    *,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buf.getvalue()


def apply_gain(pcm: bytes, volume: int) -> bytes:
    volume = max(0, min(100, int(volume)))
    if volume == 50:
        return pcm
    gain = volume / 50.0
    if _np is not None:
        samples = _np.frombuffer(pcm, dtype="<i2").astype(_np.float32) * gain
        _np.clip(samples, -32768, 32767, out=samples)
        return samples.astype("<i2").tobytes()

    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    for i, sample in enumerate(samples):
        samples[i] = max(-32768, min(32767, int(sample * gain)))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


@lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _run_ffmpeg(pcm: bytes, sample_rate: int, args: list[str]) -> bytes:
    exe = ffmpeg_path()
    if not exe:
        raise unsupported_parameter("ffmpeg is required for this format/rate")
    cmd = [
        exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        *args,
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        input=pcm,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace").strip()
        raise unsupported_parameter(f"ffmpeg failed: {message or proc.returncode}")
    return proc.stdout


def resample_pcm(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if source_rate == target_rate:
        return pcm
    return _run_ffmpeg(
        pcm,
        source_rate,
        ["-f", "s16le", "-ar", str(target_rate), "-ac", "1"],
    )


def pitch_shift_pcm(pcm: bytes, sample_rate: int, pitch: float) -> bytes:
    if abs(pitch - 1.0) < 1e-6:
        return pcm
    factor = max(0.5, min(2.0, float(pitch)))
    filter_expr = (
        f"asetrate={int(sample_rate * factor)},"
        f"aresample={sample_rate},"
        f"atempo={1.0 / factor:.6f}"
    )
    return _run_ffmpeg(pcm, sample_rate, ["-af", filter_expr, "-f", "s16le"])


def encode_audio(
    pcm: bytes,
    source_rate: int,
    fmt: str,
    *,
    sample_rate: int,
    volume: int = 50,
    pitch: float = 1.0,
    bit_rate: int | None = None,
) -> bytes:
    fmt = validate_format(fmt)
    validate_sample_rate(sample_rate)
    pcm = apply_gain(pcm, volume)
    if abs(pitch - 1.0) >= 1e-6:
        pcm = pitch_shift_pcm(pcm, source_rate, pitch)
    pcm = resample_pcm(pcm, source_rate, sample_rate)

    if fmt == "pcm":
        return pcm
    if fmt == "wav":
        return pcm_to_wav(pcm, sample_rate)
    if fmt == "mp3":
        rate = int(bit_rate or 128_000)
        return _run_ffmpeg(
            pcm,
            sample_rate,
            ["-f", "mp3", "-b:a", str(rate)],
        )
    if fmt == "opus":
        rate = int(bit_rate or 64_000)
        return _run_ffmpeg(
            pcm,
            sample_rate,
            ["-f", "opus", "-b:a", str(rate), "-ac", "1"],
        )
    raise unsupported_format(f"unsupported format: {fmt}")


def encode_chunk(
    pcm: bytes,
    source_rate: int,
    fmt: str,
    *,
    sample_rate: int,
    volume: int = 50,
    pitch: float = 1.0,
    bit_rate: int | None = None,
) -> bytes:
    """Encode one streaming chunk.

    WAV chunks are individually framed; PCM is raw s16le. MP3/Opus chunks are
    encoded independently and require ffmpeg; clients should prefer PCM/MP3 for
    long streaming sessions.
    """
    return encode_audio(
        pcm,
        source_rate,
        fmt,
        sample_rate=sample_rate,
        volume=volume,
        pitch=pitch,
        bit_rate=bit_rate,
    )


def wav_stream_header(sample_rate: int) -> bytes:
    """WAV header with unknown (streaming) RIFF/data sizes.

    Appending raw PCM after this header produces one continuous WAV stream,
    unlike concatenating independently framed WAV chunks.
    """
    channels = 1
    sample_width = 2
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8,
        b"data",
        0xFFFFFFFF,
    )


def encode_stream_chunk(
    pcm: bytes,
    source_rate: int,
    fmt: str,
    *,
    sample_rate: int,
    volume: int = 50,
    pitch: float = 1.0,
    bit_rate: int | None = None,
    first: bool = False,
) -> bytes:
    """Encode a chunk for a long-lived HTTP/WebSocket audio stream.

    ``pcm`` is raw s16le. ``wav`` sends a streaming header only on the first
    chunk and raw PCM afterwards, so concatenating all binary frames yields one
    playable WAV stream. ``opus`` is rejected for streaming because independently
    encoded Ogg/Opus chunks are not safely concatenable; use full-audio
    synthesis for Opus.
    """
    fmt = validate_format(fmt)
    validate_sample_rate(sample_rate)
    if fmt == "opus":
        raise unsupported_parameter(
            "opus is not supported for streaming synthesis; use pcm, wav or mp3"
        )
    if abs(pitch - 1.0) >= 1e-6:
        raise unsupported_parameter("pitch is not supported for streaming synthesis")

    pcm = apply_gain(pcm, volume)
    pcm = resample_pcm(pcm, source_rate, sample_rate)
    if fmt == "pcm":
        return pcm
    if fmt == "wav":
        return (wav_stream_header(sample_rate) if first else b"") + pcm
    if fmt == "mp3":
        rate = int(bit_rate or 128_000)
        return _run_ffmpeg(
            pcm,
            sample_rate,
            ["-f", "mp3", "-b:a", str(rate)],
        )
    raise unsupported_format(f"unsupported format: {fmt}")


def content_type(fmt: str) -> str:
    return {
        "pcm": "audio/L16",
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
    }.get(fmt.lower(), "application/octet-stream")


async def encode_audio_async(
    pcm: bytes,
    source_rate: int,
    fmt: str,
    *,
    sample_rate: int,
    volume: int = 50,
    pitch: float = 1.0,
    bit_rate: int | None = None,
) -> bytes:
    """Run CPU/ffmpeg encoding off the event loop."""
    return await asyncio.to_thread(
        encode_audio,
        pcm,
        source_rate,
        fmt,
        sample_rate=sample_rate,
        volume=volume,
        pitch=pitch,
        bit_rate=bit_rate,
    )


async def encode_chunk_async(
    pcm: bytes,
    source_rate: int,
    fmt: str,
    *,
    sample_rate: int,
    volume: int = 50,
    pitch: float = 1.0,
    bit_rate: int | None = None,
) -> bytes:
    return await asyncio.to_thread(
        encode_chunk,
        pcm,
        source_rate,
        fmt,
        sample_rate=sample_rate,
        volume=volume,
        pitch=pitch,
        bit_rate=bit_rate,
    )


async def encode_stream_chunk_async(
    pcm: bytes,
    source_rate: int,
    fmt: str,
    *,
    sample_rate: int,
    volume: int = 50,
    pitch: float = 1.0,
    bit_rate: int | None = None,
    first: bool = False,
) -> bytes:
    return await asyncio.to_thread(
        encode_stream_chunk,
        pcm,
        source_rate,
        fmt,
        sample_rate=sample_rate,
        volume=volume,
        pitch=pitch,
        bit_rate=bit_rate,
        first=first,
    )
