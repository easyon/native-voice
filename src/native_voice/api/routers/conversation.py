"""Full-duplex conversation WebSocket for the browser application."""
from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, WebSocket

from ...conversation.engine import ConversationEngine
from ...observability.logging import session_id_var
from ...security.auth import AuthError, Principal
from ...security.quotas import QuotaExceeded
from ..protocol.bailian.writer import ConnectionWriter

log = logging.getLogger(__name__)
router = APIRouter()


async def _principal(websocket: WebSocket):
    """Bearer for native clients; first-message ticket for browsers."""
    state = websocket.app.state
    header = websocket.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        try:
            principal = state.authenticator.authenticate_headers(websocket.headers)
            if not principal.has("conversation"):
                raise AuthError("missing conversation scope", code="Forbidden", status=403)
        except AuthError:
            await websocket.close(code=1008)
            return None
        await websocket.accept()
        return principal

    await websocket.accept()
    try:
        first = await websocket.receive_json()
    except Exception as exc:
        log.info("conversation auth message failed: %r", exc)
        await websocket.close(code=1008)
        return None
    if first.get("type") != "auth":
        await websocket.close(code=1008)
        return None
    payload = state.ticket_signer.verify(str(first.get("ticket", "")))
    if not payload:
        await websocket.close(code=1008)
        return None
    record = state.store.get(str(payload.get("k", "")))
    if record is None or not record.enabled:
        await websocket.close(code=1008)
        return None
    scopes = tuple(payload.get("s") or record.scopes)
    principal = Principal(record.key_id, scopes, record.rpm, record.max_concurrent)
    if not principal.has("conversation"):
        await websocket.close(code=1008)
        return None
    return principal


async def _serve(websocket: WebSocket, principal: Principal) -> None:
    state = websocket.app.state
    if state.llm is None:
        await websocket.send_json(
            {"type": "error", "code": "LLMUnavailable", "message": "conversation LLM is not configured"}
        )
        await websocket.close(code=1011)
        return
    try:
        lease = state.quotas.acquire(principal)
    except QuotaExceeded as exc:
        await websocket.send_json({"type": "error", "code": "Throttling", "message": str(exc)})
        await websocket.close(code=1013)
        return

    session_id_var.set(uuid.uuid4().hex[:12])
    writer = ConnectionWriter(websocket)
    engine = ConversationEngine(state.config, state.backends, state.llm, writer)
    try:
        await engine.start()
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                data = message["bytes"]
                if len(data) > state.config.server.max_message_bytes:
                    await writer.send_json(
                        {"type": "error", "code": "InvalidParameter", "message": "audio frame too large"}
                    )
                    continue
                await engine.feed_audio(data)
                continue
            text = message.get("text")
            if text is None:
                continue
            try:
                event = json.loads(text)
            except Exception:
                await writer.send_json(
                    {"type": "error", "code": "InvalidParameter", "message": "invalid JSON"}
                )
                continue
            kind = event.get("type")
            if kind == "input.text":
                await engine.input_text(str(event.get("text", "")))
            elif kind == "control.interrupt":
                await engine.interrupt()
            elif kind == "playback.started":
                await engine.playback_started(str(event.get("turn_id", "")))
            elif kind == "playback.done":
                await engine.playback_done(str(event.get("turn_id", "")))
            elif kind == "session.start":
                if event.get("voice"):
                    engine.voice = str(event["voice"])
                if event.get("endpoint_ms") is not None:
                    engine.set_endpoint_ms(int(event["endpoint_ms"]))
                if event.get("system_prompt") is not None:
                    engine.set_system_prompt(str(event["system_prompt"]))
            elif kind == "session.config":
                if event.get("endpoint_ms") is not None:
                    engine.set_endpoint_ms(int(event["endpoint_ms"]))
                if event.get("system_prompt") is not None:
                    engine.set_system_prompt(str(event["system_prompt"]))
                if event.get("voice"):
                    engine.voice = str(event["voice"])
            elif kind == "session.end":
                break
    except Exception as exc:
        log.warning("conversation session ended: %r", exc)
    finally:
        await engine.close()
        await writer.close()
        state.quotas.release(lease)


@router.websocket("/ws/conversation")
async def conversation(websocket: WebSocket) -> None:
    principal = await _principal(websocket)
    if principal is None:
        return
    await _serve(websocket, principal)
