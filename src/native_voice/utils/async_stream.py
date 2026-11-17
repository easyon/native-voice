"""Bridge a blocking generator (GPU model) into an async iterator.

A daemon pump thread consumes the generator and hands chunks to the event loop
through a small bounded queue, so consumers stream immediately while the pump
applies backpressure.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterable


class _Done:
    pass


class _Error:
    def __init__(self, exc: BaseException):
        self.exc = exc


_DONE = _Done()


async def iter_in_thread(
    iterable: Iterable,
    loop: asyncio.AbstractEventLoop,
    queue_max: int = 8,
) -> AsyncIterator:
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)

    def pump():
        try:
            for item in iterable:
                asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            asyncio.run_coroutine_threadsafe(queue.put(_Error(exc)), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(_DONE), loop).result()

    threading.Thread(target=pump, daemon=True, name="model-pump").start()
    while True:
        item = await queue.get()
        if isinstance(item, _Done):
            return
        if isinstance(item, _Error):
            raise item.exc
        yield item
