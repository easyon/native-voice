"""Bailian-compatible inference WebSocket endpoint."""
from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket

from ...security.auth import AuthError
from ..protocol.bailian.connection import BailianConnection

log = logging.getLogger(__name__)
router = APIRouter()


async def _authenticate(websocket: WebSocket):
    try:
        return websocket.app.state.authenticator.authenticate_headers(websocket.headers)
    except AuthError as exc:
        log.info("WS authentication failed: %s", exc)
        try:
            await websocket.close(code=1008)
        except Exception:
            pass
        return None


async def _serve(websocket: WebSocket, forced_kind: str | None = None) -> None:
    principal = await _authenticate(websocket)
    if principal is None:
        return
    state = websocket.app.state
    conn = BailianConnection(
        websocket=websocket,
        principal=principal,
        cfg=state.config,
        backends=state.backends,
        quotas=state.quotas,
        authenticator=state.authenticator,
        forced_kind=forced_kind,
    )
    await conn.run()


@router.websocket("/api-ws/v1/inference")
async def inference(websocket: WebSocket) -> None:
    await _serve(websocket)


@router.websocket("/ws/asr")
async def asr_alias(websocket: WebSocket) -> None:
    await _serve(websocket, forced_kind="asr")


@router.websocket("/ws/tts")
async def tts_alias(websocket: WebSocket) -> None:
    await _serve(websocket, forced_kind="tts")
