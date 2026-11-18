"""Async streaming client for the local llama.cpp server."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from ...config.schema import LLMConfig
from .think_filter import ThinkFilter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextDelta:
    """A piece of speakable assistant text."""

    text: str


@dataclass(frozen=True)
class ToolCall:
    """A complete tool call assembled from streamed fragments."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class StreamDone:
    """End of one completion round; ``finish_reason`` is the server value."""

    finish_reason: str = ""


class ToolCallAccumulator:
    """Collect streamed ``delta.tool_calls`` fragments.

    llama.cpp and the OpenAI API split one call across chunks: the first chunk
    carries ``id`` and ``function.name``, later chunks append ``arguments``
    character by character, keyed by ``index``.
    """

    def __init__(self) -> None:
        self._slots: dict[int, dict[str, str]] = {}

    def feed(self, chunks: list[dict[str, Any]] | None) -> None:
        for chunk in chunks or []:
            try:
                index = int(chunk.get("index", 0))
            except (TypeError, ValueError):
                index = 0
            slot = self._slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if chunk.get("id"):
                slot["id"] = str(chunk["id"])
            function = chunk.get("function") or {}
            if function.get("name"):
                slot["name"] = str(function["name"])
            if function.get("arguments"):
                slot["arguments"] += str(function["arguments"])

    def calls(self) -> list[ToolCall]:
        out: list[ToolCall] = []
        for index in sorted(self._slots):
            slot = self._slots[index]
            if not slot["name"]:
                continue
            out.append(
                ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=slot["arguments"] or "{}",
                )
            )
        return out


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.base = f"http://127.0.0.1:{cfg.port}"
        self.http: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5))

    async def close(self) -> None:
        if self.http is not None:
            await self.http.aclose()
            self.http = None

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                return (await client.get(self.base + "/health")).status_code == 200
        except Exception:
            return False

    async def warmup(self) -> None:
        if self.http is None:
            await self.start()
        payload = {
            "messages": [
                {"role": "system", "content": self.cfg.system_prompt},
                {"role": "user", "content": "你好"},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            resp = await self.http.post(self.base + "/v1/chat/completions", json=payload)
            if resp.status_code == 200:
                log.info("LLM prompt cache warmed")
        except Exception as exc:
            log.warning("LLM warmup failed: %r", exc)

    async def chat_stream(
        self, messages: list[dict], cancel: asyncio.Event | None = None
    ) -> AsyncIterator[str]:
        """Text-only view of :meth:`chat_stream_events`."""
        async for event in self.chat_stream_events(messages, None, cancel):
            if isinstance(event, TextDelta):
                yield event.text

    async def chat_stream_events(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        cancel: asyncio.Event | None = None,
        tool_choice: str | None = None,
    ) -> AsyncIterator[TextDelta | ToolCall | StreamDone]:
        """Stream one completion round.

        Yields text deltas as they arrive, then any completed tool calls, then a
        single :class:`StreamDone`.  A round that calls a tool usually contains
        no text at all; the caller runs the tools and starts another round.
        """
        if self.http is None:
            await self.start()
        payload: dict[str, Any] = {
            "messages": messages,
            "stream": True,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            payload["tools"] = tools
            choice = (tool_choice or self.cfg.tool_choice or "auto").strip().lower()
            payload["tool_choice"] = choice if choice in {"auto", "required", "none"} else "auto"
        filt = ThinkFilter()
        calls = ToolCallAccumulator()
        finish_reason = ""
        async with self.http.stream("POST", self.base + "/v1/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:300]
                raise RuntimeError(f"LLM HTTP {resp.status_code}: {body}")
            async for raw in resp.aiter_lines():
                if cancel is not None and cancel.is_set():
                    return
                if not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                calls.feed(delta.get("tool_calls"))
                piece = delta.get("content") or ""
                if piece:
                    text = filt.feed(piece)
                    if text:
                        yield TextDelta(text)
        tail = filt.flush()
        if tail and (cancel is None or not cancel.is_set()):
            yield TextDelta(tail)
        if cancel is not None and cancel.is_set():
            return
        for call in calls.calls():
            yield call
        yield StreamDone(finish_reason)
