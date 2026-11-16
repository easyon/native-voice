from __future__ import annotations

from fastapi import APIRouter, Header, Request
from fastapi.responses import PlainTextResponse

from .deps import authenticate_metrics, runtime_from_request

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/livez")
async def livez() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> dict[str, object]:
    runtime = runtime_from_request(request)
    ready = await runtime.ready()
    return {
        "status": "ready" if ready else "not_ready",
        "engine": runtime.config.engine.type,
        "models": sorted(runtime.config.models),
        "draining": runtime.lifecycle.draining,
        "inflight": runtime.lifecycle.inflight,
    }


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(
    request: Request,
    authorization: str | None = Header(default=None),
) -> PlainTextResponse:
    runtime = runtime_from_request(request)
    authenticate_metrics(authorization, runtime.config.auth)
    return PlainTextResponse(
        runtime.metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
