from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request):
    state = request.app.state
    backends = getattr(state, "backends", None)
    ready = bool(backends and backends.tts.ready and backends.asr.ready)
    body = {
        "status": "ready" if ready else "not_ready",
        "tts": bool(backends and backends.tts.ready),
        "asr": bool(backends and backends.asr.ready),
    }
    return JSONResponse(body, status_code=200 if ready else 503)
