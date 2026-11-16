from __future__ import annotations

import hmac

from cosyvoice_common.errors import unauthorized
from fastapi import Request, WebSocket

from ..core.tenant import Tenant
from ..runtime import Runtime
from ..settings import AuthConfig


def _bearer(authorization: str | None) -> str:
    if not authorization:
        raise unauthorized("missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise unauthorized("Authorization must be Bearer <api_key>")
    return token


def authenticate(authorization: str | None, auth: AuthConfig) -> Tenant:
    if not auth.enabled:
        return Tenant("default")
    token = _bearer(authorization)

    # Compare against every configured key in constant time. The number of
    # keys is small and configured at deploy time, so this is cheap and avoids
    # a timing oracle on the API key.
    for item in auth.tenants:
        if item.api_key and hmac.compare_digest(token, item.api_key):
            return Tenant(
                id=item.id or token[:12],
                max_concurrent_streams=item.max_concurrent_streams,
                requests_per_minute=item.requests_per_minute,
                characters_per_minute=item.characters_per_minute,
            )
    matched = False
    for key in auth.api_keys:
        if key and hmac.compare_digest(token, key):
            matched = True
    if matched:
        return Tenant(token[:12])
    raise unauthorized()


def authenticate_metrics(authorization: str | None, auth: AuthConfig) -> None:
    if not auth.enabled or auth.metrics_public:
        return
    if auth.metrics_token:
        token = _bearer(authorization)
        if not hmac.compare_digest(token, auth.metrics_token):
            raise unauthorized()
        return
    authenticate(authorization, auth)


def runtime_from_request(request: Request) -> Runtime:
    return request.app.state.runtime


def runtime_from_websocket(websocket: WebSocket) -> Runtime:
    return websocket.app.state.runtime
