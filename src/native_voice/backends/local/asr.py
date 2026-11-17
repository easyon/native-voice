"""Local streaming Paraformer backend adapter."""
from __future__ import annotations

import asyncio

import numpy as np

from ...config.schema import ServiceConfig
from ...models.asr.factory import create_asr_engine
from ..base import ASRBackend, ASRSession


class LocalASRSession(ASRSession):
    def __init__(self, engine):
        self._session = engine.create_session()

    async def reset(self, seed: np.ndarray | None = None) -> None:
        await asyncio.to_thread(self._session.reset, seed)

    async def feed(self, audio: np.ndarray) -> str:
        return await asyncio.to_thread(self._session.feed, audio)

    async def finalize(self) -> str:
        return await asyncio.to_thread(self._session.finalize)

    async def close(self) -> None:
        close = getattr(self._session, "close", None)
        if close is not None:
            await asyncio.to_thread(close)


class LocalASRBackend(ASRBackend):
    def __init__(self, cfg: ServiceConfig):
        self.engine = create_asr_engine(cfg)

    async def start(self) -> None:
        await asyncio.to_thread(self.engine.load)

    async def close(self) -> None:
        await asyncio.to_thread(self.engine.close)

    def create_session(self) -> ASRSession:
        return LocalASRSession(self.engine)

    @property
    def ready(self) -> bool:
        return self.engine.ready
