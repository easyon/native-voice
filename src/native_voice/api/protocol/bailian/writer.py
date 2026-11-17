"""Ordered, bounded WebSocket writer.

JSON control events and their binary audio payloads must stay adjacent, so
both go through a single queue served by one writer task.
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)


class ConnectionWriter:
    def __init__(self, websocket, max_items: int = 128):
        self.websocket = websocket
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_items)
        self._task = asyncio.create_task(self._run(), name="ws-writer")
        self.closed = False

    async def send_json(self, obj: dict) -> None:
        await self._put(("json", obj))

    async def send_bytes(self, data: bytes) -> None:
        await self._put(("bytes", data))

    async def _put(self, item) -> None:
        if self.closed:
            raise ConnectionError("writer closed")
        await self._queue.put(item)

    async def _run(self) -> None:
        try:
            while True:
                kind, data = await self._queue.get()
                if kind == "json":
                    await self.websocket.send_json(data)
                else:
                    await self.websocket.send_bytes(data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # connection died
            self.closed = True
            log.debug("writer stopped: %r", exc)

    async def close(self) -> None:
        self.closed = True
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
