"""Priority scheduler for one TTS engine model.

Jobs are sentences from many gateway sessions. Lower priority values run first,
so the first sentence of a new turn jumps ahead of later sentences from other
turns. Only one job uses the model at a time.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.async_stream import iter_in_thread

log = logging.getLogger(__name__)


@dataclass
class Job:
    websocket: Any
    text: str
    voice: str
    priority: int
    cancel: Any  # threading.Event
    done: asyncio.Future
    samples: int = 0


class TTSScheduler:
    def __init__(self, model):
        self.model = model
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._seq = 0
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="tts-engine-scheduler")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def submit(self, job: Job) -> None:
        self._seq += 1
        await self.queue.put((job.priority, self._seq, job))

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            priority, seq, job = await self.queue.get()
            if job.cancel.is_set():
                job.done.set_result(None)
                continue
            try:
                await job.websocket.send_json({"op": "start", "priority": priority})
                generator = self.model.iter_synthesize(job.text, job.voice, job.cancel)
                async for chunk in iter_in_thread(generator, loop):
                    if job.cancel.is_set():
                        continue
                    data = np.asarray(chunk, dtype="<f4").tobytes()
                    job.samples += len(chunk)
                    await job.websocket.send_bytes(data)
                if not job.cancel.is_set():
                    await job.websocket.send_json({"op": "done", "samples": job.samples})
            except Exception as exc:  # connection died or model error
                log.debug("TTS job failed: %r", exc)
                try:
                    await job.websocket.send_json({"op": "error", "message": repr(exc)})
                except Exception:
                    pass
            finally:
                if not job.done.done():
                    job.done.set_result(None)
