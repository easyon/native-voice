from __future__ import annotations

from typing import Any

from ..schemas import Transcript, TranscriptSegment


def error_body(
    message: str,
    *,
    type_: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": type_,
            "param": param,
            "code": code,
        }
    }


def verbose_json(transcript: Transcript, language: str | None = None) -> dict[str, Any]:
    return {
        "task": "transcribe",
        "language": language or transcript.language or "unknown",
        "duration": round(transcript.duration_ms / 1000.0, 3),
        "text": transcript.text,
        "segments": [
            segment.to_dict(index)
            for index, segment in enumerate(transcript.segments)
        ],
    }


def srt(segments: list[TranscriptSegment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{_timestamp(segment.begin_time)} --> {_timestamp(segment.end_time)}",
                    segment.text,
                    "",
                ]
            )
        )
    return "\n".join(blocks)


def vtt(segments: list[TranscriptSegment]) -> str:
    blocks = ["WEBVTT", ""]
    for segment in segments:
        blocks.append(
            "\n".join(
                [
                    f"{_timestamp(segment.begin_time, vtt=True)} --> "
                    f"{_timestamp(segment.end_time, vtt=True)}",
                    segment.text,
                    "",
                ]
            )
        )
    return "\n".join(blocks)


def _timestamp(ms: int, *, vtt: bool = False) -> str:
    ms = max(0, int(ms))
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"
