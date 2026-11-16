from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .audio import PCM_BYTES_PER_MS


@dataclass(frozen=True, slots=True)
class SpeechStart:
    time_ms: int


@dataclass(frozen=True, slots=True)
class SpeechEnd:
    time_ms: int
    reason: Literal["silence", "max_duration", "flush"] = "silence"


VadEvent = SpeechStart | SpeechEnd


class VadStream(ABC):
    @abstractmethod
    async def feed(self, pcm: bytes) -> list[VadEvent]: ...

    @abstractmethod
    async def finish(self) -> list[VadEvent]: ...

    async def close(self) -> None:
        return None


class PassthroughVadStream(VadStream):
    """Treat the entire task as one utterance."""

    def __init__(self) -> None:
        self._started = False
        self._fed_ms = 0

    async def feed(self, pcm: bytes) -> list[VadEvent]:
        self._fed_ms += len(pcm) // PCM_BYTES_PER_MS
        if not self._started and pcm:
            self._started = True
            return [SpeechStart(0)]
        return []

    async def finish(self) -> list[VadEvent]:
        if not self._started:
            return []
        self._started = False
        return [SpeechEnd(self._fed_ms, "flush")]


class MockVadStream(VadStream):
    """Deterministic segmentation for protocol tests and mock deployments."""

    def __init__(self, segment_ms: int = 1_800) -> None:
        self._segment_ms = max(200, int(segment_ms))
        self._fed_ms = 0
        self._started = False
        self._next_end = self._segment_ms

    async def feed(self, pcm: bytes) -> list[VadEvent]:
        self._fed_ms += len(pcm) // PCM_BYTES_PER_MS
        events: list[VadEvent] = []
        if not self._started and self._fed_ms > 0:
            self._started = True
            events.append(SpeechStart(0))
        while self._started and self._fed_ms >= self._next_end:
            events.append(SpeechEnd(self._next_end))
            self._next_end += self._segment_ms
            events.append(SpeechStart(self._next_end - self._segment_ms))
        return events

    async def finish(self) -> list[VadEvent]:
        if not self._started:
            return []
        self._started = False
        return [SpeechEnd(self._fed_ms, "flush")]


class EnergyVadStream(VadStream):
    """RMS endpointer with hysteresis and a hard segment-length cap.

    Used by the GGUF engine (no torch) and as a fallback when the FunASR VAD
    model cannot be loaded. It is not a replacement for fsmn-vad in noisy
    far-field audio, but it keeps the realtime protocol functional.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        threshold_db: float = -40.0,
        frame_ms: int = 20,
        min_speech_ms: int = 200,
        silence_ms: int = 1_300,
        max_segment_ms: int = 30_000,
        pre_roll_ms: int = 200,
    ) -> None:
        self._sample_rate = sample_rate
        self._threshold = float(threshold_db)
        self._frame_ms = max(10, int(frame_ms))
        self._frame_bytes = int(sample_rate * self._frame_ms / 1000) * 2
        self._min_speech_ms = max(self._frame_ms, int(min_speech_ms))
        self._silence_ms = max(self._frame_ms, int(silence_ms))
        self._max_segment_ms = max(200, int(max_segment_ms))
        self._pre_roll_ms = max(0, int(pre_roll_ms))

        self._pending = bytearray()
        self._fed_bytes = 0
        self._processed_ms = 0
        self._speech_run_ms = 0
        self._silence_run_ms = 0
        self._speaking = False
        self._start_ms = 0
        self._last_speech_ms = 0

    async def feed(self, pcm: bytes) -> list[VadEvent]:
        self._fed_bytes += len(pcm)
        self._pending.extend(pcm)
        events: list[VadEvent] = []
        while len(self._pending) >= self._frame_bytes:
            frame = bytes(self._pending[: self._frame_bytes])
            del self._pending[: self._frame_bytes]
            events.extend(self._consume_frame(frame))
        return events

    async def finish(self) -> list[VadEvent]:
        events: list[VadEvent] = []
        if len(self._pending) >= 320:
            frame = bytes(self._pending[: self._frame_bytes]).ljust(
                self._frame_bytes, b"\x00"
            )
            self._pending.clear()
            events.extend(self._consume_frame(frame))
        if self._speaking:
            end_ms = max(self._last_speech_ms, self._fed_ms())
            events.append(SpeechEnd(end_ms, "flush"))
            self._speaking = False
        return events

    def _fed_ms(self) -> int:
        return self._fed_bytes // PCM_BYTES_PER_MS

    def _consume_frame(self, frame: bytes) -> list[VadEvent]:
        frame_start = self._processed_ms
        frame_end = frame_start + self._frame_ms
        self._processed_ms = frame_end

        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        level_db = 20.0 * math.log10(rms + 1e-9)
        is_speech = level_db >= self._threshold
        events: list[VadEvent] = []

        if not self._speaking:
            if is_speech:
                self._speech_run_ms += self._frame_ms
                if self._speech_run_ms >= self._min_speech_ms:
                    self._speaking = True
                    self._start_ms = max(
                        0,
                        frame_end - self._speech_run_ms - self._pre_roll_ms,
                    )
                    self._last_speech_ms = frame_end
                    self._silence_run_ms = 0
                    events.append(SpeechStart(self._start_ms))
            else:
                self._speech_run_ms = 0
            return events

        if is_speech:
            self._last_speech_ms = frame_end
            self._silence_run_ms = 0
        else:
            self._silence_run_ms += self._frame_ms
            if self._silence_run_ms >= self._silence_ms:
                events.append(SpeechEnd(self._last_speech_ms, "silence"))
                self._speaking = False
                self._speech_run_ms = 0
                self._silence_run_ms = 0
                return events

        if frame_end - self._start_ms >= self._max_segment_ms:
            events.append(SpeechEnd(frame_end, "max_duration"))
            if is_speech:
                self._start_ms = frame_end
                self._last_speech_ms = frame_end
                self._silence_run_ms = 0
                events.append(SpeechStart(frame_end))
            else:
                self._speaking = False
                self._speech_run_ms = 0
                self._silence_run_ms = 0
        return events
