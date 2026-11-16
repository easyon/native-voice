from __future__ import annotations

import json
import subprocess

from cosyvoice_common.audio import ffmpeg_path
from cosyvoice_common.errors import invalid_parameter, unsupported_format

PCM_SAMPLE_RATE = 16_000
PCM_BYTES_PER_MS = 32  # 16 kHz * mono * 16 bit
_SUPPORTED_STREAM_FORMATS = {"pcm", "wav"}
_COMPRESSED_FORMATS = {
    "aac",
    "amr",
    "flac",
    "m4a",
    "mp3",
    "mp4",
    "ogg",
    "opus",
    "speex",
    "wav",
    "webm",
    "wma",
}


def pcm_duration_ms(pcm: bytes) -> int:
    return len(pcm) // PCM_BYTES_PER_MS


def validate_stream_format(fmt: str) -> str:
    normalized = (fmt or "pcm").lower().lstrip(".")
    if normalized not in _SUPPORTED_STREAM_FORMATS:
        raise unsupported_format(
            f"unsupported realtime audio format: {fmt!r}; "
            "supported: pcm, wav"
        )
    return normalized


class WavStreamReader:
    """Incremental WAV parser for realtime uploads.

    A WAV header may be split across WebSocket binary frames, so the reader
    buffers until it has the RIFF/fmt/data chunks, validates PCM mono 16 kHz,
    then passes the payload through. Compressed or multi-channel WAV is
    rejected instead of being fed to the model as garbage.
    """

    def __init__(self, sample_rate: int = PCM_SAMPLE_RATE) -> None:
        self._buffer = bytearray()
        self._header_done = False
        self._data_remaining: int | None = None
        self._expected_rate = sample_rate
        self._fmt_seen = False

    def feed(self, data: bytes) -> bytes:
        if self._header_done:
            return self._take(data)

        self._buffer.extend(data)
        while not self._header_done:
            payload = self._parse_header()
            if payload is None:
                return b""
            leftover, size = payload
            self._header_done = True
            self._data_remaining = size
            self._buffer.clear()
            return self._take(leftover)
        return b""

    def finish(self) -> bytes:
        if not self._header_done:
            raise unsupported_format("incomplete WAV header")
        if self._data_remaining is not None and self._data_remaining > 0:
            # A truncated data chunk is tolerated: the client may omit the
            # RIFF size or stop early; we return what we have.
            self._data_remaining = 0
        return b""

    def _take(self, data: bytes) -> bytes:
        if self._data_remaining is None:
            return data
        if len(data) > self._data_remaining:
            data = data[: self._data_remaining]
        self._data_remaining -= len(data)
        return data

    def _parse_header(self) -> tuple[bytes, int | None] | None:
        buf = self._buffer
        if len(buf) < 12:
            return None
        if bytes(buf[:4]) != b"RIFF" or bytes(buf[8:12]) != b"WAVE":
            raise unsupported_format("stream is not a RIFF/WAVE file")

        pos = 12
        while True:
            if len(buf) < pos + 8:
                return None
            chunk_id = bytes(buf[pos : pos + 4])
            size = int.from_bytes(buf[pos + 4 : pos + 8], "little")
            body_start = pos + 8
            if chunk_id == b"fmt ":
                if len(buf) < body_start + size:
                    return None
                self._validate_fmt(bytes(buf[body_start : body_start + size]))
                self._fmt_seen = True
                pos = body_start + size + (size & 1)
                continue
            if chunk_id == b"data":
                if not self._fmt_seen:
                    raise unsupported_format("WAV data chunk precedes fmt chunk")
                leftover = bytes(buf[body_start:])
                # 0xFFFFFFFF means "streaming/unknown size".
                remaining = None if size in (0, 0xFFFFFFFF) else size
                return leftover, remaining
            if len(buf) < body_start + size:
                return None
            pos = body_start + size + (size & 1)

    def _validate_fmt(self, fmt: bytes) -> None:
        if len(fmt) < 16:
            raise unsupported_format("WAV fmt chunk is too short")
        audio_format = int.from_bytes(fmt[0:2], "little")
        channels = int.from_bytes(fmt[2:4], "little")
        sample_rate = int.from_bytes(fmt[4:8], "little")
        bits = int.from_bytes(fmt[14:16], "little")
        if audio_format != 1:
            raise unsupported_format("only PCM WAV is supported")
        if channels != 1:
            raise unsupported_format("only mono audio is supported")
        if bits != 16:
            raise unsupported_format("only 16-bit audio is supported")
        if sample_rate != self._expected_rate:
            raise unsupported_format(
                f"unsupported WAV sample rate {sample_rate}; "
                f"expected {self._expected_rate}"
            )


def decode_wav(data: bytes) -> bytes:
    reader = WavStreamReader()
    out = reader.feed(data)
    return out + reader.finish()


def decode_pcm(data: bytes) -> bytes:
    if len(data) % 2:
        raise unsupported_format("PCM data must contain complete 16-bit samples")
    return data


def decode_to_pcm16k(data: bytes, fmt: str | None = None) -> bytes:
    """Decode an uploaded file to 16 kHz / mono / s16le.

    ``fmt`` is the file suffix (``wav``, ``mp3``, ...). PCM is passed through;
    WAV uses the strict streaming parser first and falls back to ffmpeg for
    non-16k/multi-channel files; everything else goes through ffmpeg.
    """
    normalized = (fmt or "").lower().lstrip(".")
    if normalized == "pcm":
        return decode_pcm(data)
    if normalized in {"wav", "wave"}:
        try:
            return decode_wav(data)
        except Exception:
            return _ffmpeg_decode(data)
    if normalized and normalized not in _COMPRESSED_FORMATS:
        raise unsupported_format(f"unsupported audio format: {fmt!r}")
    if not normalized and data[:4] == b"RIFF":
        try:
            return decode_wav(data)
        except Exception:
            return _ffmpeg_decode(data)
    return _ffmpeg_decode(data)


def _ffmpeg_decode(data: bytes) -> bytes:
    exe = ffmpeg_path()
    if not exe:
        raise unsupported_format(
            "ffmpeg is required to decode this audio format; "
            "upload pcm or wav instead"
        )
    command = [
        exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ar",
        str(PCM_SAMPLE_RATE),
        "-ac",
        "1",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            command,
            input=data,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise unsupported_format(f"ffmpeg failed: {exc}") from exc
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace").strip()
        raise unsupported_format(f"ffmpeg failed: {message or proc.returncode}")
    return proc.stdout


class TextFilter:
    """Bailian ``special_word_filter`` post-processing.

    The value is a JSON string::

        {"filter_with_signed":{"word_list":["项目"]},
         "filter_with_empty":{"word_list":["报告"]},
         "system_reserved_filter":false}

    Matching words are replaced by same-length ``*`` or removed. This is a
    text post-process, never a model hotword.
    """

    def __init__(self, signed: list[str] | None = None, removed: list[str] | None = None):
        self._signed = [word for word in (signed or []) if word]
        self._removed = [word for word in (removed or []) if word]

    @classmethod
    def parse(cls, spec: str | None) -> TextFilter:
        if not spec or not spec.strip():
            return cls()
        try:
            payload = json.loads(spec)
        except json.JSONDecodeError as exc:
            raise invalid_parameter(
                f"parameters.special_word_filter is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise invalid_parameter("parameters.special_word_filter must be an object")
        signed = _word_list(payload.get("filter_with_signed"))
        removed = _word_list(payload.get("filter_with_empty"))
        if len(signed) + len(removed) > 32:
            raise invalid_parameter(
                "parameters.special_word_filter supports at most 32 words"
            )
        return cls(signed, removed)

    def apply(self, text: str) -> str:
        for word in self._removed:
            text = text.replace(word, "")
        for word in self._signed:
            text = text.replace(word, "*" * len(word))
        return text


def _word_list(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    words = value.get("word_list")
    if not isinstance(words, list):
        return []
    return [str(word) for word in words if isinstance(word, (str, int, float))]
