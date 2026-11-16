from __future__ import annotations

from .core.vad import VadStream
from .engines.base import ASREngine, ASRStream
from .engines.llamacpp import LlamaCppEngine
from .engines.mock import MockEngine
from .schemas import StreamParams, Transcript
from .settings import ModelConfig, RuntimeConfig


class EngineRouter(ASREngine):
    """Dispatches each model to its configured engine.

    A deployment can expose a mock model for health checks, a PyTorch model for
    realtime, and a GGUF model for offline transcription at the same time.
    Only the engines actually referenced by ``models.*.engine`` are started.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._engines: dict[str, ASREngine] = {}
        kinds = {model.engine for model in config.models.values()}
        self._engines["mock"] = MockEngine(config.engine.mock)
        if "funasr" in kinds:
            from .engines.funasr import FunasrEngine

            self._engines["funasr"] = FunasrEngine(
                config.engine,
                {
                    name: model
                    for name, model in config.models.items()
                    if model.engine == "funasr"
                },
            )
        if "llamacpp" in kinds:
            self._engines["llamacpp"] = LlamaCppEngine(
                config.engine,
                {
                    name: model
                    for name, model in config.models.items()
                    if model.engine == "llamacpp"
                },
            )

    def _for(self, model: ModelConfig) -> ASREngine:
        return self._engines[model.engine or self._config.engine.type]

    async def startup(self) -> None:
        for engine in self._engines.values():
            await engine.startup()

    async def shutdown(self) -> None:
        for engine in self._engines.values():
            await engine.shutdown()

    async def ready(self) -> bool:
        for engine in self._engines.values():
            if not await engine.ready():
                return False
        return True

    async def warmup(self) -> None:
        for engine in self._engines.values():
            await engine.warmup()

    def create_stream(self, model: ModelConfig, params: StreamParams) -> ASRStream:
        return self._for(model).create_stream(model, params)

    def create_vad_stream(
        self, model: ModelConfig, params: StreamParams
    ) -> VadStream:
        return self._for(model).create_vad_stream(model, params)

    async def transcribe(
        self, model: ModelConfig, pcm: bytes, params: StreamParams
    ) -> Transcript:
        return await self._for(model).transcribe(model, pcm, params)


def create_engine(config: RuntimeConfig) -> ASREngine:
    return EngineRouter(config)
