from __future__ import annotations

from collections.abc import Callable

from cosyvoice_common.errors import service_unavailable

from .engines.base import TTSEngine
from .engines.cosyvoice import CosyVoiceEngine
from .engines.mock import MockTTSEngine
from .settings import RuntimeConfig

EngineFactory = Callable[[RuntimeConfig], TTSEngine]
_ENGINES: dict[str, EngineFactory] = {}


def register_engine(name: str, factory: EngineFactory) -> None:
    _ENGINES[name] = factory


def create_engine(config: RuntimeConfig) -> TTSEngine:
    factory = _ENGINES.get(config.engine.type)
    if factory is None:
        known = ", ".join(sorted(_ENGINES)) or "<none>"
        raise service_unavailable(
            f"unknown engine type: {config.engine.type}; registered: {known}"
        )
    return factory(config)


register_engine("mock", lambda cfg: MockTTSEngine(cfg.engine.mock))
register_engine(
    "cosyvoice",
    lambda cfg: CosyVoiceEngine(cfg.engine.cosyvoice, cfg.models, cfg.voices),
)
