"""Internal engine authentication (localhost + optional shared token)."""
from __future__ import annotations

import hmac

from ..config.schema import ServiceConfig


def authorized(headers, cfg: ServiceConfig) -> bool:
    token = cfg.engine.token
    if not token:
        return True
    supplied = headers.get("authorization", "")
    return hmac.compare_digest(supplied, f"Bearer {token}")
