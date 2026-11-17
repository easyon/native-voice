"""Local CosyVoice3 backend adapter."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from ...config.schema import ServiceConfig
from ...models.tts.factory import create_tts_model
from ...service.voice_store import VoiceStore
from ...utils.async_stream import iter_in_thread
from ..base import TTSBackend


class LocalTTSBackend(TTSBackend):
    def __init__(self, cfg: ServiceConfig, voice_store: VoiceStore | None = None):
        self.voice_store = voice_store
        self.model = create_tts_model(cfg)

    async def start(self) -> None:
        await asyncio.to_thread(self.model.load)
        if self.voice_store is not None:
            for rec in self.voice_store.list():
                await asyncio.to_thread(
                    self.model.add_voice,
                    rec["id"],
                    rec["prompt_text"],
                    Path(rec["path"]),
                    rec.get("description", ""),
                )

    async def close(self) -> None:
        await asyncio.to_thread(self.model.close)

    async def synthesize(
        self, text: str, voice: str, cancel, priority: int = 1
    ) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        generator = self.model.iter_synthesize(text, voice, cancel)
        async for chunk in iter_in_thread(generator, loop):
            yield chunk

    def voices(self) -> list[dict]:
        return self.model.voices()

    async def add_voice(self, name: str, prompt_wav, prompt_text: str, description: str = "") -> None:
        await asyncio.to_thread(self.model.add_voice, name, prompt_text, Path(prompt_wav), description)

    async def remove_voice(self, name: str) -> None:
        await asyncio.to_thread(self.model.remove_voice, name)

    @property
    def ready(self) -> bool:
        return self.model.ready
