from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from cosyvoice_common.errors import throttled

from ..settings import LimitsConfig
from .tenant import Tenant


class _TokenBucket:
    def __init__(self, capacity: int, per_minute: int) -> None:
        self.capacity = float(max(1, capacity))
        self.rate = max(0.001, float(per_minute) / 60.0)
        self.tokens = self.capacity
        self.updated = time.monotonic()

    def consume(self, amount: float = 1.0) -> bool:
        now = time.monotonic()
        self.tokens = min(
            self.capacity,
            self.tokens + (now - self.updated) * self.rate,
        )
        self.updated = now
        if self.tokens < amount:
            return False
        self.tokens -= amount
        return True


class _Entry:
    def __init__(self, tenant: Tenant, default_max_concurrent: int) -> None:
        limit = tenant.max_concurrent_streams or default_max_concurrent
        self.semaphore = asyncio.Semaphore(max(1, int(limit)))
        self.requests = (
            _TokenBucket(tenant.requests_per_minute, tenant.requests_per_minute)
            if tenant.requests_per_minute
            else None
        )
        self.characters = (
            _TokenBucket(tenant.characters_per_minute, tenant.characters_per_minute)
            if tenant.characters_per_minute
            else None
        )


class TenantSlot:
    def __init__(self, entry: _Entry) -> None:
        self._entry = entry
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._entry.semaphore.release()


class TenantLimiter:
    """Per-tenant concurrency / request-rate / character-rate limits.

    The orchestrator still has a global semaphore. These limits prevent one
    noisy tenant from occupying the whole GPU pool.
    """

    def __init__(self, limits: LimitsConfig) -> None:
        self._default_max = max(1, int(limits.max_concurrent_streams))
        self._acquire_timeout_s = max(0.1, float(limits.acquire_timeout_s))
        self._entries: dict[str, _Entry] = {}

    def _entry(self, tenant: Tenant) -> _Entry:
        entry = self._entries.get(tenant.id)
        if entry is None:
            entry = self._entries[tenant.id] = _Entry(tenant, self._default_max)
        return entry

    async def acquire(self, tenant: Tenant) -> TenantSlot:
        entry = self._entry(tenant)
        try:
            await asyncio.wait_for(
                entry.semaphore.acquire(),
                timeout=self._acquire_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise throttled("tenant concurrency limit exceeded") from exc

        if entry.requests is not None and not entry.requests.consume():
            entry.semaphore.release()
            raise throttled("tenant request rate exceeded")
        return TenantSlot(entry)

    @asynccontextmanager
    async def slot(self, tenant: Tenant) -> AsyncIterator[TenantSlot]:
        handle = await self.acquire(tenant)
        try:
            yield handle
        finally:
            await handle.release()

    def consume_characters(self, tenant: Tenant, characters: int) -> None:
        entry = self._entry(tenant)
        if entry.characters is None:
            return
        if not entry.characters.consume(max(0, int(characters))):
            raise throttled("tenant character rate exceeded")
