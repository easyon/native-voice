from __future__ import annotations

from fastapi import APIRouter, Request, Response

from ...observability.metrics import render_metrics

router = APIRouter()


@router.get("/metrics")
async def prometheus() -> Response:
    payload, content_type = render_metrics()
    return Response(payload, media_type=content_type)


@router.get("/v1/metrics/summary")
async def summary(request: Request) -> dict:
    state = request.app.state
    backends = getattr(state, "backends", None)
    quotas = getattr(state, "quotas", None)
    return {
        "ready": bool(backends and backends.tts.ready and backends.asr.ready),
        "quota": quotas.stats() if quotas else {},
    }
