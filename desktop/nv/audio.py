"""PortAudio audio I/O: continuous microphone capture and interruptible playback."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd


def list_devices() -> str:
    return str(sd.query_devices())


class Mic:
    """Continuous mono float32 capture at 16 kHz.

    The PortAudio callback only copies frames into a bounded queue; all VAD work
    happens on the conversation thread, so a slow consumer drops the oldest audio
    instead of blocking the audio engine.
    """

    def __init__(self, device: int | None, rate: int = 16000, frame_ms: int = 20):
        self.device = device
        self.rate = rate
        self.blocksize = int(rate * frame_ms / 1000)
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=400)
        self._stream: sd.InputStream | None = None
        self.overruns = 0
        self.latency = 0.0

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.overruns += 1
        frame = indata[:, 0].copy()
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

    def start(self):
        self._stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.rate,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()
        self.latency = float(self._stream.latency)

    def read(self, timeout: float = 0.05) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class WavMic:
    """Replay a wav file as a stand-in microphone, paced in real time.

    Used by `--wav` to exercise the full pipeline (VAD/ASR/LLM/TTS/playback)
    without needing to speak. After the file ends it keeps emitting silence.
    """

    def __init__(self, path: str | Path, rate: int = 16000, frame_ms: int = 20):
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != rate:
            from scipy.signal import resample_poly

            g = np.gcd(int(sr), int(rate))
            data = resample_poly(data, rate // g, sr // g)
        # A short lead-in so the VAD can calibrate its noise floor.
        lead = np.zeros(int(rate * 1.0), dtype=np.float32)
        self.data = np.concatenate([lead, np.asarray(data, dtype=np.float32)])
        self.rate = rate
        self.block = int(rate * frame_ms / 1000)
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=400)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.overruns = 0
        self.latency = 0.0

    def _run(self):
        t0 = time.time()
        pos = 0
        while not self._stop.is_set():
            if pos < len(self.data):
                block = self.data[pos:pos + self.block]
            else:
                block = np.zeros(self.block, dtype=np.float32)
            pos += self.block
            due = t0 + pos / self.rate
            delay = due - time.time()
            if delay > 0:
                time.sleep(delay)
            try:
                self._q.put_nowait(block)
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(block)
                except queue.Full:
                    pass

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="wavmic")
        self._thread.start()

    def read(self, timeout: float = 0.05) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self._stop.set()


class Speaker:
    """Mono float32 playback at 24 kHz with a cleared queue on barge-in."""

    def __init__(self, device: int | None, rate: int = 24000, gain: float = 1.0):
        self.device = device
        self.rate = rate
        self.gain = gain
        self._bufs: list[np.ndarray] = []
        self._pos = 0
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self.underruns = 0
        self.latency = 0.0
        self._audible_cb = None
        self._audible_pending = False

    def _callback(self, outdata, frames, time_info, status):
        if status:
            self.underruns += 1
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < frames and self._bufs:
                buf = self._bufs[0]
                take = min(frames - filled, len(buf) - self._pos)
                if take <= 0:
                    self._bufs.pop(0)
                    self._pos = 0
                    continue
                out[filled:filled + take] = buf[self._pos:self._pos + take]
                self._pos += take
                filled += take
                if self._pos >= len(buf):
                    self._bufs.pop(0)
                    self._pos = 0
        np.clip(out, -1.0, 1.0, out=out)
        outdata[:, 0] = out
        cb = None
        if self._audible_pending and np.any(np.abs(out) > 1e-4):
            with self._lock:
                if self._audible_pending:
                    self._audible_pending = False
                    cb = self._audible_cb
                    self._audible_cb = None
        if cb is not None:
            try:
                cb(time.monotonic() + self.latency)
            except Exception:
                pass

    def start(self):
        try:
            self._stream = sd.OutputStream(
                device=self.device,
                channels=1,
                samplerate=self.rate,
                dtype="float32",
                blocksize=480,
                latency="low",
                callback=self._callback,
            )
        except Exception:
            # Some drivers reject a fixed 20 ms block; let PortAudio choose.
            self._stream = sd.OutputStream(
                device=self.device,
                channels=1,
                samplerate=self.rate,
                dtype="float32",
                latency="low",
                callback=self._callback,
            )
        self._stream.start()
        self.latency = float(self._stream.latency)

    def arm_audible(self, callback):
        """Call `callback(monotonic_time)` when the first non-silent audio is heard."""
        with self._lock:
            self._audible_cb = callback
            self._audible_pending = True

    def write(self, chunk: np.ndarray):
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self.gain != 1.0:
            chunk = chunk * self.gain
        if chunk.size == 0:
            return
        with self._lock:
            self._bufs.append(chunk)

    def stop(self, fade_ms: float = 14.0):
        """Drop queued audio immediately, with a tiny fade to avoid a click."""
        with self._lock:
            if not self._bufs:
                return
            head = self._bufs[0][self._pos:]
            n = min(len(head), max(1, int(self.rate * fade_ms / 1000)))
            fade = head[:n].copy()
            if n > 1:
                fade *= np.linspace(1.0, 0.0, n, dtype=np.float32)
            self._bufs = [fade] if n else []
            self._pos = 0

    def is_idle(self) -> bool:
        with self._lock:
            return not self._bufs

    def queued_seconds(self) -> float:
        with self._lock:
            if not self._bufs:
                return 0.0
            total = sum(len(b) for b in self._bufs) - self._pos
            return max(0.0, total / self.rate)

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
