"""TTS GPU engine: one CosyVoice3 model, priority-queued sentence jobs."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket

from ..config.schema import ServiceConfig
from ..models.tts.factory import create_tts_model
from ..observability.logging import configure_logging
from ..service.voice_store import VoiceStore
from .scheduler import Job, TTSScheduler
from .security import authorized

log = logging.getLogger(__name__)


def create_tts_app(cfg: ServiceConfig, provider: str | None = None) -> FastAPI:
    provider = provider or cfg.engine.tts_provider

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(cfg.logging.level, cfg.logging.json_logs)
        model = create_tts_model(cfg, provider)
        log.info("starting TTS provider=%s", provider)
        await asyncio.to_thread(model.load)
        store = VoiceStore(cfg.voice_store.dir)
        for rec in store.list():
            try:
                await asyncio.to_thread(
                    model.add_voice,
                    rec["id"],
                    rec["prompt_text"],
                    Path(rec["path"]),
                    rec.get("description", ""),
                )
                log.info("loaded cloned voice %s", rec["id"])
            except Exception as exc:
                log.warning("failed to load cloned voice %s: %r", rec.get("id"), exc)
        scheduler = TTSScheduler(model)
        scheduler.start()
        app.state.model = model
        app.state.scheduler = scheduler
        app.state.store = store
        log.info("TTS engine ready (provider=%s)", provider)
        try:
            yield
        finally:
            await scheduler.stop()
            await asyncio.to_thread(model.close)

    app = FastAPI(title="Native Voice TTS Engine", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        return {"ready": bool(app.state.model.ready), "provider": provider}

    @app.get("/voices")
    async def list_voices(request: Request):
        if not authorized(request.headers, cfg):
            raise HTTPException(status_code=403, detail="forbidden")
        return {"voices": app.state.model.voices()}

    @app.post("/voices")
    async def add_voice(request: Request):
        if not authorized(request.headers, cfg):
            raise HTTPException(status_code=403, detail="forbidden")
        body = await request.json()
        voice_id = str(body.get("id", "")).strip()
        prompt_text = str(body.get("prompt_text", "")).strip()
        prompt_wav = str(body.get("prompt_wav", "")).strip()
        description = str(body.get("description", ""))
        if not voice_id or not prompt_text or not prompt_wav:
            raise HTTPException(status_code=400, detail="id, prompt_wav and prompt_text are required")
        if len(prompt_text) > 3000:
            raise HTTPException(status_code=400, detail="prompt_text too long")
        path = Path(prompt_wav).resolve()
        root = Path(cfg.voice_store.dir).resolve()
        if not path.is_relative_to(root) or not path.exists():
            raise HTTPException(status_code=400, detail="prompt_wav must be inside the voice store")
        await asyncio.to_thread(app.state.model.add_voice, voice_id, prompt_text, path, description)
        return {"ok": True}

    @app.delete("/voices/{voice_id}")
    async def remove_voice(voice_id: str, request: Request):
        if not authorized(request.headers, cfg):
            raise HTTPException(status_code=403, detail="forbidden")
        await asyncio.to_thread(app.state.model.remove_voice, voice_id)
        return {"ok": True}

    @app.websocket("/tts")
    async def tts_job(websocket: WebSocket):
        if not authorized(websocket.headers, cfg):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            first = json.loads((await websocket.receive()).get("text") or "{}")
            if first.get("op") != "synthesize":
                await websocket.close(code=1008)
                return
            cancel = threading.Event()
            job = Job(
                websocket=websocket,
                text=str(first.get("text", "")),
                voice=str(first.get("voice", "default")),
                priority=int(first.get("priority", 1)),
                cancel=cancel,
                done=asyncio.get_running_loop().create_future(),
            )
            await app.state.scheduler.submit(job)
            while not job.done.done():
                try:
                    msg = await asyncio.wait_for(websocket.receive(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if msg.get("type") == "websocket.disconnect":
                    cancel.set()
                    break
                text = msg.get("text")
                if text:
                    try:
                        obj = json.loads(text)
                    except Exception:
                        continue
                    if obj.get("op") == "cancel":
                        cancel.set()
            await job.done
        except Exception as exc:
            log.debug("TTS engine connection ended: %r", exc)

    return app
