from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Word:
    begin_time: int
    end_time: int
    text: str
    punctuation: str = ""


@dataclass(frozen=True, slots=True)
class Partial:
    text: str
    begin_time: int
    end_time: int | None = None
    words: tuple[Word, ...] = ()


@dataclass(frozen=True, slots=True)
class Final:
    text: str
    begin_time: int
    end_time: int
    words: tuple[Word, ...] = ()
    # Total model time spent on this segment, including interim partial decodes.
    decode_seconds: float = 0.0


@dataclass(slots=True)
class StreamParams:
    """Recognition parameters normalized for the engines.

    Times are milliseconds on the task timeline. ``speech_noise_threshold``
    follows FunASR's 0..1 convention; the Bailian -1..1 value is converted in
    ``protocol.bailian``.
    """

    sample_rate: int = 16_000
    language: str = "auto"
    hotwords: str = ""
    itn: bool = True
    max_sentence_silence_ms: int = 1_300
    speech_noise_threshold: float = 0.5
    partial_interval_ms: int = 960
    partial_window_s: float = 8.0
    max_segment_ms: int = 30_000
    timestamps: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranscriptSegment:
    text: str
    begin_time: int
    end_time: int
    words: list[Word] = field(default_factory=list)

    def to_dict(self, index: int) -> dict[str, Any]:
        return {
            "id": index,
            "start": round(self.begin_time / 1000.0, 3),
            "end": round(self.end_time / 1000.0, 3),
            "text": self.text,
        }


@dataclass(slots=True)
class Transcript:
    text: str
    language: str | None
    duration_ms: int
    segments: list[TranscriptSegment] = field(default_factory=list)
