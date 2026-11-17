"""Backend selection: fake for tests/dev, local models for the real service."""
from __future__ import annotations

from ..config.schema import ServiceConfig
from .base import Backends


def build_backends(cfg: ServiceConfig, kind: str = "local", voice_store=None) -> Backends:
    if kind == "fake":
        from .fake import FakeASR, FakeTTS

        return Backends(FakeTTS(voice_store=voice_store), FakeASR())

    if kind == "worker":
        from .worker.asr import WorkerASRBackend
        from .worker.tts import WorkerTTSBackend

        return Backends(
            WorkerTTSBackend(
                cfg.engine.tts_ws, cfg.engine.tts_http, cfg.engine.token, cfg.voices, voice_store
            ),
            WorkerASRBackend(cfg.engine.asr_ws, cfg.engine.token, cfg.engine.asr_http),
        )

    from .local.asr import LocalASRBackend
    from .local.tts import LocalTTSBackend

    return Backends(
        LocalTTSBackend(cfg, voice_store),
        LocalASRBackend(cfg),
    )
