"""Backend interfaces shared by local models, remote engines and test fakes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np


@dataclass
class Backends:
    tts: TTSBackend
    asr: ASRBackend

    async def start(self) -> None:
        await self.tts.start()
        await self.asr.start()

    async def close(self) -> None:
        await self.tts.close()
        await self.asr.close()


class TTSBackend(ABC):
    """24 kHz float32 mono chunks, async streaming."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def synthesize(
        self, text: str, voice: str, cancel, priority: int = 1
    ) -> AsyncIterator[np.ndarray]: ...

    @abstractmethod
    def voices(self) -> list[dict]: ...

    async def add_voice(self, name: str, prompt_wav, prompt_text: str, description: str = "") -> None:
        raise NotImplementedError("this backend does not support voice cloning")

    async def remove_voice(self, name: str) -> None:
        raise NotImplementedError("this backend does not support voice cloning")

    @property
    @abstractmethod
    def ready(self) -> bool: ...


class ASRSession(ABC):
    """Per-utterance streaming state. Feed 16 kHz mono float32 audio."""

    @abstractmethod
    async def reset(self, seed: np.ndarray | None = None) -> None: ...

    @abstractmethod
    async def feed(self, audio: np.ndarray) -> str:
        """Return cumulative partial text for the current sentence."""

    @abstractmethod
    async def finalize(self) -> str: ...

    @abstractmethod
    async def close(self) -> None: ...


class ASRBackend(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def create_session(self) -> ASRSession: ...

    @property
    @abstractmethod
    def ready(self) -> bool: ...
