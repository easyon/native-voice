"""ASR GPU engine: shared streaming Paraformer, one WS session per utterance."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket

from ..config.schema import ServiceConfig
from ..models.asr.factory import create_asr_engine
from ..observability.logging import configure_logging
from .security import authorized

log = logging.getLogger(__name__)


def create_asr_app(cfg: ServiceConfig, provider: str | None = None) -> FastAPI:
    provider = provider or cfg.engine.asr_provider

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(cfg.logging.level, cfg.logging.json_logs)
        engine = create_asr_engine(cfg, provider)
        log.info("starting ASR provider=%s", provider)
        await asyncio.to_thread(engine.load)
        app.state.engine = engine
        log.info("ASR engine ready (provider=%s)", provider)
        try:
            yield
        finally:
            await asyncio.to_thread(engine.close)

    app = FastAPI(title="Native Voice ASR Engine", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        return {"ready": bool(app.state.engine.ready), "provider": provider}

    @app.websocket("/asr")
    async def asr_session(websocket: WebSocket):
        if not authorized(websocket.headers, cfg):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        session = app.state.engine.create_session()
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    audio = np.frombuffer(msg["bytes"], dtype="<f4").astype(np.float32)
                    text = await asyncio.to_thread(session.feed, audio)
                    await websocket.send_json({"op": "partial", "text": text})
                    continue
                try:
                    obj = json.loads(msg.get("text") or "{}")
                except Exception:
                    continue
                op = obj.get("op")
                if op == "open":
                    await websocket.send_json({"op": "ready"})
                elif op == "finalize":
                    text = await asyncio.to_thread(session.finalize)
                    await websocket.send_json({"op": "final", "text": text})
                elif op == "reset":
                    await asyncio.to_thread(session.reset)
                    await websocket.send_json({"op": "ok"})
                elif op == "close":
                    break
        except Exception as exc:
            log.debug("ASR engine session ended: %r", exc)
        finally:
            close = getattr(session, "close", None)
            if close is not None:
                await asyncio.to_thread(close)

    return app
