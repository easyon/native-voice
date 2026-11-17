"""Qwen3-TTS served by an external vLLM-Omni process.

The OSS ``qwen-tts`` package generates a whole utterance before returning
audio. vLLM-Omni exposes a real streaming WebSocket endpoint (``/v1/audio/
speech/stream``): text is sent incrementally and the server returns PCM chunks
as the codec frames are produced. This adapter keeps the engine protocol
unchanged and only translates vLLM's PCM16 stream into the float32 chunks the
gateway already understands.

The vLLM-Omni process runs separately (see ``deploy/vllm-omni``), normally in
a Linux Docker container with GPU access. This module therefore has no torch
dependency and can run in the root venv.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from ...config.schema import VoiceProfile

log = logging.getLogger(__name__)


@dataclass
class Qwen3TTSVLLMConfig:
    url: str = "ws://127.0.0.1:8091/v1/audio/speech/stream"
    health_url: str = "http://127.0.0.1:8091/health"
    # Host root that is mounted into the container at ``container_root``.
    root: str | Path = "."
    container_root: str = "/workspace"
    # ``None`` omits the field: Qwen3-TTS accepts a language, CosyVoice3 does not.
    language: str | None = "Chinese"
    # Session fields are model-family specific. Qwen3-TTS needs ``task_type``
    # (``Base`` = voice cloning) and ``x_vector_only_mode``; CosyVoice3 rejects
    # both, so its provider sets ``task_type=None`` and disables the x-vector
    # flag. ``None`` means "do not send this field".
    task_type: str | None = "Base"
    x_vector_only_mode: bool = True
    sample_rate: int = 24000
    response_format: str = "pcm"
    # "none" lets the gateway's clause splitter decide sentence boundaries.
    split_granularity: str | None = None
    # Optional per-request cap. None keeps vLLM's text-scaled Base ceiling.
    max_new_tokens: int | None = None
    startup_timeout_s: float = 600.0
    # vLLM-Omni closes an idle WebSocket after 30 s. Reconnect proactively
    # before that so a stale socket costs no failed request.
    idle_reconnect_s: float = 20.0
    # The first real request warms lazy CUDA/code2wav paths (several seconds).
    # Do it during load so /readyz means the first user turn is already fast.
    warmup: bool = True
    warmup_text: str = "你好。"
    voices: dict[str, VoiceProfile] = field(default_factory=dict)
    prompt_wav: Path | None = None
    prompt_text: str = ""
    description: str = ""


def _plain(text: str) -> str:
    value = (text or "").strip()
    if "<|endofprompt|>" in value:
        value = value.split("<|endofprompt|>", 1)[1].strip()
    return value


def _ref_audio_uri(path: str) -> str:
    """vLLM-Omni only accepts URLs, data: URLs, or file:// URIs."""
    path = (path or "").strip()
    if path.startswith(("http://", "https://", "file://", "data:")):
        return path
    return "file://" + quote(path)


class Qwen3TTSVLLM:
    LABEL = "Qwen3-TTS vLLM"

    def __init__(self, cfg: Qwen3TTSVLLMConfig):
        self.cfg = cfg
        self.sample_rate = int(cfg.sample_rate)
        self._voices: dict[str, tuple[str, str, str]] = {}
        self._voice_names: list[str] = []
        self._lock = threading.Lock()
        # One long-lived WS: vLLM-Omni accepts session.config -> session.done
        # repeatedly on the same connection. Serialise whole utterances.
        self._ws_lock = threading.Lock()
        self._ws = None
        self._ws_last_used = 0.0
        self._ready = False
        self._warm_voice: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        self._wait_for_server()
        voices = self.cfg.voices or {}
        if voices:
            for name, profile in voices.items():
                self._add_voice(name, profile.prompt_text, Path(profile.prompt_wav), profile.description)
        elif self.cfg.prompt_wav and Path(self.cfg.prompt_wav).exists():
            self._add_voice("default", self.cfg.prompt_text, Path(self.cfg.prompt_wav), self.cfg.description)
        if not self._voice_names:
            raise RuntimeError(
                f"{self.LABEL} provider needs at least one reference voice; "
                "configure voices.yaml or models.tts_prompt_wav"
            )
        self._warm_voice = self._voice_names[0]
        if self.cfg.warmup:
            try:
                for _ in self.iter_synthesize(self.cfg.warmup_text, self._warm_voice):
                    pass
            except Exception as exc:  # keep readiness; log the first-turn risk
                log.warning("%s warmup failed: %s", self.LABEL, exc)
        self._ready = True
        log.info(
            "%s ready (url=%s, voices=%s)",
            self.LABEL,
            self.cfg.url,
            ",".join(self._voice_names),
        )

    def _wait_for_server(self) -> None:
        if not self.cfg.health_url:
            return
        deadline = time.monotonic() + max(1.0, float(self.cfg.startup_timeout_s))
        last: Exception | None = None
        # Localhost health checks must not go through a system HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            try:
                with opener.open(self.cfg.health_url, timeout=2.0) as resp:
                    if 200 <= resp.status < 300:
                        return
            except Exception as exc:  # server not up yet
                last = exc
            time.sleep(1.0)
        raise RuntimeError(
            f"vLLM-Omni TTS server is not ready at {self.cfg.health_url} "
            f"after {self.cfg.startup_timeout_s:.0f}s: {last!r}"
        )

    # ----------------------------------------------------------------- voices
    def _add_voice(self, name: str, prompt_text: str, prompt_wav: Path, description: str = "") -> None:
        ref_audio = self._container_path(prompt_wav)
        text = _plain(prompt_text)
        self._voices[name] = (ref_audio, text, description or name)
        if name not in self._voice_names:
            self._voice_names.append(name)

    def _container_path(self, path: Path) -> str:
        root = Path(self.cfg.root).resolve()
        p = Path(path)
        if not p.is_absolute():
            p = root / p
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"reference audio not found: {p}")
        if not self.cfg.container_root:
            return str(p)
        try:
            rel = p.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"reference audio must be inside {root} so it can be mounted "
                f"into the vLLM container: {p}"
            ) from exc
        return str(Path(self.cfg.container_root) / rel).replace("\\", "/")

    def add_voice(self, name: str, prompt_text: str, prompt_wav: Path, description: str = "") -> None:
        with self._lock:
            self._add_voice(name, prompt_text, prompt_wav, description)
        log.info("Qwen3-TTS vLLM voice added: %s", name)

    def remove_voice(self, name: str) -> None:
        with self._lock:
            self._voices.pop(name, None)
            if name in self._voice_names:
                self._voice_names.remove(name)
            if self._warm_voice == name:
                self._warm_voice = self._voice_names[0] if self._voice_names else None
        log.info("Qwen3-TTS vLLM voice removed: %s", name)

    def voices(self) -> list[dict]:
        return [
            {"name": name, "description": self._voices[name][2]}
            for name in self._voice_names
        ]

    def has_voice(self, name: str) -> bool:
        return name in self._voice_names

    # ---------------------------------------------------------------- synthesis
    def _session_config(self, voice: str) -> dict[str, Any]:
        ref_audio, ref_text, _ = self._voices[voice]
        cfg: dict[str, Any] = {
            "response_format": self.cfg.response_format,
            "stream_audio": True,
            "ref_audio": _ref_audio_uri(ref_audio),
            "ref_text": ref_text,
        }
        if self.cfg.task_type:
            cfg["task_type"] = self.cfg.task_type
        if self.cfg.language:
            cfg["language"] = self.cfg.language
        if self.cfg.x_vector_only_mode:
            cfg["x_vector_only_mode"] = not bool(ref_text)
        if self.cfg.split_granularity:
            cfg["split_granularity"] = self.cfg.split_granularity
        if self.cfg.max_new_tokens is not None:
            cfg["max_new_tokens"] = int(self.cfg.max_new_tokens)
        return cfg

    def iter_synthesize(self, text: str, voice: str, cancel=None) -> Iterator[np.ndarray]:
        """Yield 24 kHz float32 mono chunks for one clause."""
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            if voice not in self._voices:
                raise ValueError(f"unknown voice: {voice}")
            session = self._session_config(voice)
        if cancel is not None and cancel.is_set():
            return

        with self._ws_lock:
            attempts = 2
            for attempt in range(attempts):
                completed = False
                yielded = False
                try:
                    if self._ws is not None and (
                        self._ws_closed(self._ws)
                        or time.monotonic() - self._ws_last_used
                        > max(1.0, self.cfg.idle_reconnect_s)
                    ):
                        self._drop_ws()
                    ws = self._ensure_ws()
                    ws.send(json.dumps({"type": "session.config", **session}, ensure_ascii=False))
                    ws.send(json.dumps({"type": "input.text", "text": text}, ensure_ascii=False))
                    ws.send(json.dumps({"type": "input.done"}))
                    for chunk in self._receive(ws, cancel):
                        yielded = True
                        yield chunk
                    completed = not (cancel is not None and cancel.is_set())
                    return
                except Exception as exc:
                    self._drop_ws()
                    if cancel is not None and cancel.is_set():
                        return
                    # A stale idle socket can fail before any audio; reconnect and
                    # retry once. Never retry after audio, to avoid duplicates.
                    if yielded or attempt == attempts - 1:
                        raise RuntimeError(
                            f"vLLM-Omni TTS request failed: {exc!r}"
                        ) from exc
                finally:
                    # An abandoned/cancelled generator leaves the server mid-session;
                    # throw the connection away rather than reuse an ambiguous state.
                    if not completed:
                        self._drop_ws()
                    else:
                        self._ws_last_used = time.monotonic()

    @staticmethod
    def _ws_closed(ws) -> bool:
        """Close code lives on the sans-I/O protocol in websockets>=13."""
        protocol = getattr(ws, "protocol", None)
        if protocol is not None:
            return protocol.close_code is not None
        return getattr(ws, "close_code", None) is not None

    def _ensure_ws(self):
        if self._ws is None:
            from websockets.sync.client import connect

            self._ws = connect(
                self.cfg.url,
                open_timeout=10.0,
                close_timeout=2.0,
                max_size=None,
                compression=None,
            )
            self._ws_last_used = time.monotonic()
        return self._ws

    def _drop_ws(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _receive(self, ws, cancel) -> Iterator[np.ndarray]:
        while True:
            if cancel is not None and cancel.is_set():
                try:
                    ws.send(json.dumps({"type": "session.close"}))
                except Exception:
                    pass
                # The session is no longer reusable; the caller's finally block
                # drops the socket so the next clause reconnects cleanly.
                return
            try:
                msg = ws.recv(timeout=0.25)
            except TimeoutError:
                continue
            if isinstance(msg, (bytes, bytearray)):
                yield from self._pcm16(msg)
                continue
            try:
                obj = json.loads(msg)
            except Exception:
                continue
            kind = obj.get("type")
            if kind == "audio.chunk":
                data = obj.get("audio_b64")
                if data:
                    yield from self._pcm16(base64.b64decode(data))
            elif kind == "audio.done":
                if obj.get("error"):
                    raise RuntimeError(str(obj.get("message") or "vLLM-Omni TTS failed"))
            elif kind == "session.done":
                return
            elif kind == "error":
                raise RuntimeError(str(obj.get("message") or obj))
            # audio.start / unknown JSON frames carry no audio.

    def _pcm16(self, data: bytes | bytearray) -> Iterator[np.ndarray]:
        if len(data) < 2:
            return
        usable = len(data) - (len(data) % 2)
        pcm = np.frombuffer(bytes(data[:usable]), dtype="<i2")
        if not pcm.size:
            return
        yield pcm.astype(np.float32) / 32768.0

    def close(self) -> None:
        with self._ws_lock:
            self._drop_ws()
        with self._lock:
            self._voices.clear()
            self._voice_names.clear()
            self._warm_voice = None
            self._ready = False
