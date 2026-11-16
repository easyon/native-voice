from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.vad import VadStream
from ..schemas import Final, Partial, StreamParams, Transcript
from ..settings import ModelConfig


class ASRStream(ABC):
    """One utterance (VAD-delimited segment) decoded incrementally."""

    @abstractmethod
    async def feed(self, pcm: bytes) -> list[Partial]: ...

    @abstractmethod
    async def finish(self) -> Final | None: ...

    async def close(self) -> None:
        return None


class ASREngine(ABC):
    @abstractmethod
    async def startup(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    async def ready(self) -> bool: ...

    @abstractmethod
    async def warmup(self) -> None: ...

    @abstractmethod
    def create_stream(self, model: ModelConfig, params: StreamParams) -> ASRStream: ...

    @abstractmethod
    def create_vad_stream(
        self, model: ModelConfig, params: StreamParams
    ) -> VadStream: ...

    @abstractmethod
    async def transcribe(
        self, model: ModelConfig, pcm: bytes, params: StreamParams
    ) -> Transcript: ...
