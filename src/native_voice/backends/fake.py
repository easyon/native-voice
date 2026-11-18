"""Deterministic fake backends for protocol tests and UI development."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from ..models.llm.client import StreamDone, TextDelta, ToolCall
from .base import ASRBackend, ASRSession, TTSBackend

_FAKE_ASR_TEXT = "你好，这是一段测试语音"


class FakeTTS(TTSBackend):
    def __init__(self, sample_rate: int = 24000, chunk_ms: int = 120, voice_store=None):
        self.sample_rate = sample_rate
        self.chunk = int(sample_rate * chunk_ms / 1000)
        self.voice_store = voice_store
        self._extra: list[dict] = []
        self._ready = False

    async def start(self) -> None:
        self._ready = True

    async def close(self) -> None:
        self._ready = False

    def voices(self) -> list[dict]:
        return [{"name": "default", "description": "fake"}] + list(self._extra)

    async def add_voice(self, name: str, prompt_wav, prompt_text: str, description: str = "") -> None:
        if name not in {v["name"] for v in self._extra}:
            self._extra.append({"name": name, "description": description or name})

    async def remove_voice(self, name: str) -> None:
        self._extra = [v for v in self._extra if v["name"] != name]

    @property
    def ready(self) -> bool:
        return self._ready

    async def synthesize(self, text: str, voice: str, cancel, priority: int = 1) -> AsyncIterator[np.ndarray]:
        total = max(0.4, len(text) * 0.06)
        samples = int(total * self.sample_rate)
        pos = 0
        while pos < samples:
            if cancel is not None and cancel.is_set():
                return
            n = min(self.chunk, samples - pos)
            t = np.arange(n, dtype=np.float32) + pos
            yield (0.05 * np.sin(2 * np.pi * 440 * t / self.sample_rate)).astype(np.float32)
            pos += n
            await asyncio.sleep(0.02)


class FakeASRSession(ASRSession):
    def __init__(self):
        self.samples = 0
        self._final = False

    async def reset(self, seed=None) -> None:
        self.samples = int(len(seed)) if seed is not None else 0
        self._final = False

    async def feed(self, audio: np.ndarray) -> str:
        self.samples += int(len(audio))
        if self._final:
            return ""
        seconds = self.samples / 16000
        chars = min(len(_FAKE_ASR_TEXT), int(seconds * 5))
        return _FAKE_ASR_TEXT[:chars]

    async def finalize(self) -> str:
        self._final = True
        return _FAKE_ASR_TEXT if self.samples > 1600 else ""

    async def close(self) -> None:
        return None


class FakeLLM:
    """Canned streaming reply for UI/dev without llama-server.

    ``script`` drives multi-round tool conversations.  Each entry is one
    completion round and may contain ``text`` chunks and/or ``tool_calls``::

        FakeLLM(script=[
            {"tool_calls": [{"name": "get_current_time", "arguments": "{}"}]},
            {"text": ["现在是", "下午三点。"]},
        ])

    Without a script every round streams :attr:`chunks`.
    """

    def __init__(self, chunks: list[str] | None = None, script: list[dict[str, Any]] | None = None):
        self.chunks = chunks or ["你好。", "我是本地语音助手，", "有什么可以帮你？"]
        self.script = list(script or [])
        # Introspection helpers for tests.
        self.last_tools: list[dict] | None = None
        self.last_messages: list[dict] = []
        self.rounds = 0

    async def chat_stream_events(self, messages, tools=None, cancel=None, tool_choice=None):
        self.rounds += 1
        self.last_tools = tools
        self.last_messages = list(messages)
        step = self.script.pop(0) if self.script else {"text": self.chunks}
        for index, call in enumerate(step.get("tool_calls") or []):
            yield ToolCall(
                id=str(call.get("id") or f"call_{index}"),
                name=str(call["name"]),
                arguments=str(call.get("arguments") or "{}"),
            )
        for chunk in step.get("text") or []:
            if cancel is not None and cancel.is_set():
                return
            await asyncio.sleep(0.02)
            yield TextDelta(chunk)
        yield StreamDone("stop")

    async def chat_stream(self, messages, cancel=None):
        async for event in self.chat_stream_events(messages, None, cancel):
            if isinstance(event, TextDelta):
                yield event.text


class FakeASR(ASRBackend):
    def __init__(self):
        self._ready = False

    async def start(self) -> None:
        self._ready = True

    async def close(self) -> None:
        self._ready = False

    def create_session(self) -> ASRSession:
        return FakeASRSession()

    @property
    def ready(self) -> bool:
        return self._ready
