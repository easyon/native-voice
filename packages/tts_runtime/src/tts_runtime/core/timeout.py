from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from cosyvoice_common.errors import timeout_error


@asynccontextmanager
async def deadline(seconds: float) -> AsyncIterator[None]:
    """Cancel the current scope after `seconds` and map it to Timeout.

    Uses asyncio.timeout on 3.11+, with a small fallback for 3.10.
    """
    if seconds <= 0:
        yield
        return

    if hasattr(asyncio, "timeout"):
        try:
            async with asyncio.timeout(seconds):
                yield
        except TimeoutError as exc:
            raise timeout_error() from exc
        return

    task = asyncio.current_task()
    if task is None:
        yield
        return
    loop = asyncio.get_running_loop()
    fired = False

    def cancel_task() -> None:
        nonlocal fired
        fired = True
        task.cancel()

    handle = loop.call_later(seconds, cancel_task)
    try:
        yield
    except asyncio.CancelledError:
        if not fired:
            raise
        raise timeout_error() from None
    finally:
        handle.cancel()
