from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..schemas import SynthesisRequest


@dataclass(slots=True)
class AudioChunk:
    pcm: bytes
    sample_rate: int
    is_final: bool = False


class TTSEngine(ABC):
    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def ready(self) -> bool:
        return True

    async def warmup(self) -> None:
        """Optional warmup after startup; failure must fail readiness."""
        return None

    @abstractmethod
    def synthesize(self, request: SynthesisRequest) -> AsyncIterator[AudioChunk]:
        """Yield mono s16le PCM chunks."""
        raise NotImplementedError
