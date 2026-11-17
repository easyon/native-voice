"""Async streaming client for the local llama.cpp server."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import httpx

from ...config.schema import LLMConfig
from .think_filter import ThinkFilter

log = logging.getLogger(__name__)


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
        if self.http is None:
            await self.start()
        payload = {
            "messages": messages,
            "stream": True,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        filt = ThinkFilter()
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
                piece = (choices[0].get("delta") or {}).get("content") or ""
                if piece:
                    text = filt.feed(piece)
                    if text:
                        yield text
        tail = filt.flush()
        if tail and (cancel is None or not cancel.is_set()):
            yield tail
