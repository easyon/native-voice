from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...security.auth import AuthError

router = APIRouter()


@router.post("/v1/realtime/ticket")
async def create_ticket(request: Request):
    state = request.app.state
    try:
        principal = state.authenticator.authenticate_headers(request.headers)
    except AuthError as exc:
        return JSONResponse({"error": exc.code, "message": str(exc)}, status_code=exc.status)
    origin = request.headers.get("origin", "")
    ticket = state.ticket_signer.issue(
        principal.key_id, list(principal.scopes), state.config.auth.ticket_ttl_s, origin
    )
    return {"ticket": ticket, "expires_in": state.config.auth.ticket_ttl_s}
