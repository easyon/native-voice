"""Energy-based sentence segmenter for the Bailian-compatible ASR stream.

It is intentionally separate from the conversational VAD: the realtime ASR API
needs `max_sentence_silence` semantics and absolute stream timestamps, while the
conversation endpoint tunes endpointing for barge-in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentEvent:
    kind: str          # "start" | "end"
    begin_ms: float = 0.0
    end_ms: float = 0.0


class SentenceSegmenter:
    def __init__(
        self,
        rate: int = 16000,
        frame_ms: int = 20,
        max_silence_ms: int = 1300,
        noise_threshold: float = 0.0,
    ):
        self.frame_ms = frame_ms
        self.frame_len = int(rate * frame_ms / 1000)
        self.max_silence_ms = max(200, min(6000, int(max_silence_ms)))
        self.offset = 6.0 * max(-1.0, min(1.0, float(noise_threshold)))
        self.residual = np.zeros(0, dtype=np.float32)
        self.ms = 0.0
        self.noise_db = -55.0
        self.warmup = 0
        self._warm_levels: list[float] = []
        self.in_speech = False
        self.start_run = 0
        self.silence_ms = 0.0
        self.begin_ms = 0.0
        self.last_voiced_ms = 0.0
        self.speech_started_ms = 0.0

    @property
    def audio_seconds(self) -> float:
        return self.ms / 1000.0

    @staticmethod
    def _db(frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
        return 20.0 * math.log10(max(rms, 1e-7))

    def _thresholds(self) -> tuple[float, float]:
        start = max(self.noise_db + 7.0, -46.0) + self.offset
        return start, start - 5.0

    def push(self, audio: np.ndarray) -> list[SegmentEvent]:
        events: list[SegmentEvent] = []
        self.residual = np.concatenate(
            [self.residual, np.asarray(audio, dtype=np.float32).reshape(-1)]
        )
        while len(self.residual) >= self.frame_len:
            frame = self.residual[: self.frame_len]
            self.residual = self.residual[self.frame_len:]
            frame_start = self.ms
            self.ms += self.frame_ms
            db = self._db(frame)

            if self.warmup < 15:
                self.warmup += 1
                self._warm_levels.append(db)
                if self.warmup == 15 and self._warm_levels:
                    floor = float(np.percentile(self._warm_levels, 30))
                    # If the stream starts with speech, calibration must not set
                    # the noise floor to the speech level (that would mask it).
                    if floor > -30.0:
                        floor = -45.0
                    self.noise_db = float(np.clip(floor, -70, -20))
                continue

            start_db, end_db = self._thresholds()
            if not self.in_speech:
                self._update_noise(db)
                if db >= start_db:
                    self.start_run += 1
                else:
                    self.start_run = 0
                if self.start_run >= 3:
                    self.in_speech = True
                    self.begin_ms = max(0.0, frame_start - (self.start_run - 1) * self.frame_ms)
                    self.last_voiced_ms = self.ms
                    self.silence_ms = 0.0
                    self.start_run = 0
                    events.append(SegmentEvent("start", begin_ms=self.begin_ms))
                continue

            if db >= end_db:
                self.last_voiced_ms = self.ms
                self.silence_ms = 0.0
            else:
                self.silence_ms += self.frame_ms
            if self.silence_ms >= self.max_silence_ms or (self.ms - self.begin_ms) > 30_000:
                self.in_speech = False
                events.append(SegmentEvent("end", begin_ms=self.begin_ms, end_ms=self.last_voiced_ms))
                self.start_run = 0
                self.silence_ms = 0.0
        return events

    def _update_noise(self, db: float) -> None:
        if db < self.noise_db:
            self.noise_db = 0.8 * self.noise_db + 0.2 * db
        else:
            self.noise_db = 0.995 * self.noise_db + 0.005 * db
        self.noise_db = float(np.clip(self.noise_db, -75.0, -20.0))
