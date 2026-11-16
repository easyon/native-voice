from __future__ import annotations

import asyncio
import logging

from cosyvoice_common.errors import service_unavailable

logger = logging.getLogger("funasr_runtime.lifecycle")


class Lifecycle:
    """Tracks in-flight protocol requests and coordinates graceful shutdown."""

    def __init__(self, shutdown_timeout_s: float) -> None:
        self._lock = asyncio.Lock()
        self._inflight = 0
        self._draining = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._shutdown_timeout_s = max(0.1, float(shutdown_timeout_s))

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def inflight(self) -> int:
        return self._inflight

    async def enter(self) -> None:
        async with self._lock:
            if self._draining:
                raise service_unavailable("service is shutting down")
            self._inflight += 1
            self._idle.clear()

    async def leave(self) -> None:
        async with self._lock:
            if self._inflight <= 0:
                return
            self._inflight -= 1
            if self._inflight == 0:
                self._idle.set()

    async def begin_shutdown(self) -> None:
        async with self._lock:
            self._draining = True
            if self._inflight == 0:
                self._idle.set()

    async def drain(self) -> bool:
        try:
            await asyncio.wait_for(
                self._idle.wait(),
                timeout=self._shutdown_timeout_s,
            )
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "graceful shutdown timed out with %s requests in flight",
                self._inflight,
            )
            return False
