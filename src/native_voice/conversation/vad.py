"""Energy VAD with adaptive noise floor, endpointing and barge-in detection.

The VAD is intentionally lightweight: it runs on the capture path, decides when a
user turn starts/ends, and raises a start event early enough to interrupt the
assistant.  During playback the trigger threshold is raised so that speaker echo
is unlikely to count as speech (headphones are still recommended).
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .params import VadParams


@dataclass
class VadEvent:
    kind: str                 # "start" | "end"
    audio: np.ndarray | None = None
    duration: float = 0.0
    db: float = -120.0
    last_voiced_time: float = 0.0   # monotonic time of the last voiced frame


def _dbfs(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
    return 20.0 * math.log10(max(rms, 1e-7))


class EndpointVad:
    def __init__(self, cfg: VadParams, speech_gate=None):
        self.cfg = cfg
        self.speech_gate = speech_gate
        self.rate = cfg.input_rate
        self.frame_len = int(cfg.input_rate * cfg.frame_ms / 1000)
        self.frame_s = cfg.frame_ms / 1000.0

        # Conservative floor before calibration; it only moves down during warmup.
        self.noise_db: float | None = -55.0
        self.level_db = -120.0
        self._warmup = 0
        self._warm_levels: list[float] = []

        self.active = False
        self._frames: list[np.ndarray] = []
        self._voiced_ms = 0
        self._silence_ms = 0
        self._noise_ms = 0
        self._start_run = 0
        self._preroll: deque[np.ndarray] = deque()
        self._residual = np.zeros(0, dtype=np.float32)
        self._barge = False
        self._echo_until = 0.0
        self._last_voiced_time = 0.0
        self.echo_level_db: float | None = None
        self._playback_active = False
        self._grace_remaining = 0
        self._grace_levels: list[float] = []

    # ------------------------------------------------------------- controls
    def set_agent_speaking(self, speaking: bool):
        if speaking == self._barge:
            return
        self._barge = speaking
        self._start_run = 0
        if speaking:
            self._echo_until = 0.0
            self._playback_active = False
            self.echo_level_db = None
        else:
            self._echo_until = time.monotonic() + self.cfg.echo_tail_s
            self._playback_active = False
            self._grace_remaining = 0

    def on_playback_started(self):
        """Called when the first TTS audio chunk is actually handed to the speaker."""
        self._playback_active = True
        self._start_run = 0
        # Measure the speaker->mic echo for a moment before allowing barge-in.
        self._grace_remaining = max(1, int(0.35 / self.frame_s))
        self._grace_levels = []
        self.echo_level_db = (self.noise_db if self.noise_db is not None else -55.0) + 10.0

    def _echo_mode(self) -> bool:
        return self._barge or time.monotonic() < self._echo_until

    # ------------------------------------------------------------- internals
    def _thresholds(self) -> tuple[float, float, int]:
        noise = self.noise_db if self.noise_db is not None else -60.0
        if self._echo_mode():
            start = max(noise + 12.0, self.cfg.barge_db)
            if self.echo_level_db is not None:
                start = max(start, self.echo_level_db + 6.0)
            sustain = max(1, int(round(self.cfg.barge_sustain_ms / self.cfg.frame_ms)))
            if self.active:
                # Once the user is talking over the assistant, endpoint on their
                # voice rather than on the (echo-inflated) barge threshold.
                end = max(noise + 4.0, -40.0)
            else:
                end = start - 5.0
        else:
            start = max(noise + 7.0, self.cfg.silence_db_abs)
            sustain = 3
            end = start - 5.0
        return start, end, sustain

    def _update_noise(self, db: float):
        if self.active or self._echo_mode():
            return
        if self.noise_db is None:
            self.noise_db = db
            return
        if db < self.noise_db:
            self.noise_db = 0.8 * self.noise_db + 0.2 * db
        else:
            self.noise_db = 0.995 * self.noise_db + 0.005 * db
        self.noise_db = float(np.clip(self.noise_db, -75.0, -20.0))

    def _trim_preroll(self):
        max_frames = max(1, int(self.cfg.preroll_ms / self.cfg.frame_ms))
        while len(self._preroll) > max_frames:
            self._preroll.popleft()

    def _finalize(self) -> VadEvent | None:
        audio = np.concatenate(self._frames) if self._frames else np.zeros(0, dtype=np.float32)
        voiced = self._voiced_ms
        self.active = False
        self._frames = []
        self._silence_ms = 0
        self._noise_ms = 0
        self._voiced_ms = 0
        self._start_run = 0
        self._preroll.clear()
        if voiced < self.cfg.min_speech_ms or audio.size == 0:
            return None
        return VadEvent(
            "end",
            audio=audio,
            duration=audio.size / self.rate,
            last_voiced_time=self._last_voiced_time,
        )

    # ----------------------------------------------------------------- main
    def feed(self, audio: np.ndarray) -> list[VadEvent]:
        """Feed arbitrary-length 16 kHz mono audio, return start/end events."""
        events: list[VadEvent] = []
        self._residual = np.concatenate([self._residual, np.asarray(audio, dtype=np.float32).reshape(-1)])
        n = self.frame_len
        while len(self._residual) >= n:
            frame = self._residual[:n]
            self._residual = self._residual[n:]
            ev = self._process_frame(frame)
            if ev is not None:
                events.append(ev)
        return events

    def _process_frame(self, frame: np.ndarray) -> VadEvent | None:
        db = _dbfs(frame)
        prob = self.speech_gate.push(frame) if self.speech_gate is not None else 1.0
        start_prob = getattr(self.speech_gate, "start_threshold", 0.5) if self.speech_gate else 0.0
        silence_prob = getattr(self.speech_gate, "silence_threshold", 0.3) if self.speech_gate else 0.0
        self.level_db = db
        if self._warmup < self.cfg.vad_warmup_frames:
            self._warmup += 1
            # Ignore the first ~200 ms: drivers/AGC often produce a settling transient.
            if self._warmup > 10:
                self._warm_levels.append(db)
            self._preroll.append(frame)
            self._trim_preroll()
            if self._warmup == self.cfg.vad_warmup_frames and self._warm_levels:
                floor = float(np.percentile(np.asarray(self._warm_levels), 30))
                self.noise_db = float(np.clip(floor, -70.0, -20.0))
            return None

        if self._echo_mode() and self._playback_active and not self.active:
            if self._grace_remaining > 0:
                self._grace_levels.append(db)
                self._grace_remaining -= 1
                if self._grace_remaining == 0 and self._grace_levels:
                    self.echo_level_db = float(np.percentile(np.asarray(self._grace_levels), 60))
                self._preroll.append(frame)
                self._trim_preroll()
                return None
            if self.echo_level_db is None:
                self.echo_level_db = db
            else:
                self.echo_level_db = 0.98 * self.echo_level_db + 0.02 * db

        start_db, end_db, sustain = self._thresholds()

        if not self.active:
            self._update_noise(db)
            if db >= start_db and prob >= start_prob:
                self._start_run += 1
            else:
                self._start_run = 0
            if self._start_run >= sustain:
                self.active = True
                self._frames = list(self._preroll)
                self._frames.append(frame)
                self._preroll.clear()
                self._voiced_ms = self._start_run * self.cfg.frame_ms
                self._silence_ms = 0
                self._start_run = 0
                self._last_voiced_time = time.monotonic()
                audio = np.concatenate(self._frames)
                return VadEvent("start", audio=audio, duration=audio.size / self.rate, db=db)
            self._preroll.append(frame)
            self._trim_preroll()
            return None

        # active utterance
        self._frames.append(frame)
        energy_voiced = db >= end_db
        if energy_voiced:
            self._voiced_ms += self.cfg.frame_ms
            self._last_voiced_time = time.monotonic()
        # Energy silence ends quickly. High energy with a low speech probability
        # is usually room noise; use the gate but require a longer window so a
        # brief probability dip in real speech does not cut the user off.
        if not energy_voiced:
            self._silence_ms += self.cfg.frame_ms
            self._noise_ms = 0
        elif prob >= silence_prob:
            self._silence_ms = 0
            self._noise_ms = 0
        else:
            self._silence_ms = 0
            self._noise_ms += self.cfg.frame_ms

        total_ms = len(self._frames) * self.cfg.frame_ms
        if (
            self._silence_ms >= self.cfg.endpoint_silence_ms
            or self._noise_ms >= self.cfg.noise_override_ms
        ):
            return self._finalize()
        if total_ms >= self.cfg.max_utterance_s * 1000:
            return self._finalize()
        return None

    @property
    def noise_floor_db(self) -> float:
        return self.noise_db if self.noise_db is not None else float("nan")
