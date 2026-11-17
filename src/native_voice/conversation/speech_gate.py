"""Silero VAD speech-probability gate (ONNX, CPU, no torch).

The energy VAD is fast but cannot tell a cough/keyboard bang from speech. This
gate consumes 512-sample windows (with the model's 64-sample context) and
returns a speech probability, which the conversation VAD uses to start/keep/
end utterances and to require a longer confirmation before barge-in.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class SpeechGate:
    def __init__(
        self,
        model_path: str | Path,
        enabled: bool = True,
        start_threshold: float = 0.5,
        silence_threshold: float = 0.3,
    ):
        self.start_threshold = start_threshold
        self.silence_threshold = silence_threshold
        self.enabled = False
        self.session = None
        self.prob = 0.0
        self._buf = np.zeros(0, dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        if not enabled:
            return
        path = Path(model_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            log.warning("Silero VAD model not found at %s; using energy-only VAD", path)
            return
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"], sess_options=opts
            )
            self.enabled = True
            log.info("Silero speech gate enabled")
        except Exception as exc:  # pragma: no cover - optional dependency
            log.warning("Silero speech gate unavailable: %r", exc)

    def reset(self) -> None:
        self.prob = 0.0
        self._buf = np.zeros(0, dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def push(self, frame: np.ndarray) -> float:
        if not self.enabled:
            return 1.0
        self._buf = np.concatenate([self._buf, np.asarray(frame, dtype=np.float32).reshape(-1)])
        while len(self._buf) >= 512:
            window = self._buf[:512]
            self._buf = self._buf[512:]
            inp = np.concatenate([self._context, window[None, :]], axis=1).astype(np.float32)
            out, state = self.session.run(
                None,
                {
                    "input": inp,
                    "state": self._state,
                    "sr": np.array(16000, dtype=np.int64),
                },
            )
            self._state = state
            self._context = inp[:, -64:]
            self.prob = float(out[0, 0])
        return self.prob
