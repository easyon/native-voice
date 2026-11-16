from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .deps import authenticate_metrics, runtime_from_request

router = APIRouter()


@router.get("/healthz", include_in_schema=False)
@router.get("/livez", include_in_schema=False)
async def healthz(request: Request) -> dict[str, str]:
    """Liveness: the process is up and the event loop is serving requests."""
    runtime = runtime_from_request(request)
    return {
        "status": "ok",
        "service": runtime.config.service_name,
        "version": runtime.config.version,
    }


@router.get("/readyz", include_in_schema=False)
async def readyz(request: Request):
    """Readiness: loaded, warmed up and not draining."""
    runtime = runtime_from_request(request)
    try:
        ready = await runtime.ready()
    except Exception:  # noqa: BLE001 - readiness must never raise
        ready = False
    body = {
        "status": "ready" if ready else "not_ready",
        "engine": runtime.config.engine.type,
        "draining": runtime.lifecycle.draining,
        "inflight": runtime.lifecycle.inflight,
        "models": sorted(runtime.config.models),
        "voices": sorted(runtime.config.voices),
    }
    return body if ready else JSONResponse(body, status_code=503)


@router.get("/metrics", include_in_schema=False)
async def metrics(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> PlainTextResponse:
    runtime = runtime_from_request(request)
    authenticate_metrics(authorization, runtime.config.auth)
    return PlainTextResponse(
        runtime.metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
