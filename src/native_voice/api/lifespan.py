"""Application lifespan: build security, backends and metrics once."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..backends.factory import build_backends
from ..observability.logging import configure_logging
from ..observability.metrics import METRICS
from ..security.auth import Authenticator
from ..security.keystore import KeyStore
from ..security.quotas import QuotaManager
from ..security.tickets import TicketSigner
from ..service.voice_store import VoiceStore

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = app.state.config
    configure_logging(cfg.logging.level, cfg.logging.json_logs)

    store = KeyStore(cfg.auth.db_path)
    if cfg.auth.dev_key:
        store.ensure_dev_key()
    quotas = QuotaManager(cfg.quotas, store)
    authenticator = Authenticator(store)
    voice_store = VoiceStore(cfg.voice_store.dir)

    supervisor = None
    if app.state.backend_kind == "worker" and cfg.engine.mode == "spawn":
        if not app.state.config_path:
            raise RuntimeError("engine.mode=spawn requires a config path")
        from ..service.engine_supervisor import EngineSupervisor

        supervisor = EngineSupervisor(cfg, app.state.config_path)
        await supervisor.start()

    backends = build_backends(cfg, app.state.backend_kind, voice_store)
    await backends.start()

    llm = app.state.injected_llm
    llm_supervisor = None
    if llm is None and cfg.llm.enabled and app.state.backend_kind != "fake":
        from ..models.llm.client import LLMClient
        from ..service.llm_supervisor import LLMSupervisor

        llm_supervisor = LLMSupervisor(cfg)
        await llm_supervisor.start()
        llm = LLMClient(cfg.llm)
        await llm.start()
        if cfg.llm.warm:
            await llm.warmup()

    app.state.store = store
    app.state.quotas = quotas
    app.state.authenticator = authenticator
    app.state.backends = backends
    app.state.voice_store = voice_store
    app.state.supervisor = supervisor
    app.state.llm = llm
    app.state.llm_supervisor = llm_supervisor
    app.state.ticket_signer = TicketSigner(cfg.auth.ticket_secret)
    METRICS.ready.set(1)
    log.info("service ready (backend=%s, engine=%s)", app.state.backend_kind, cfg.engine.mode)
    try:
        yield
    finally:
        METRICS.ready.set(0)
        if llm is not None and llm is not app.state.injected_llm:
            await llm.close()
        if llm_supervisor is not None:
            await llm_supervisor.stop()
        await backends.close()
        if supervisor is not None:
            await supervisor.stop()
        store.close()
        log.info("service stopped")
