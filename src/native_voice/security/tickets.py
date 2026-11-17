"""Short-lived HMAC tickets for browser WebSocket authentication."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class TicketSigner:
    def __init__(self, secret: str | bytes | None = None):
        if secret is None or secret == "":
            self.secret = secrets.token_bytes(32)
        elif isinstance(secret, str):
            self.secret = secret.encode()
        else:
            self.secret = secret

    def issue(self, key_id: str, scopes: list[str], ttl_s: int = 60, origin: str = "") -> str:
        payload = {
            "k": key_id,
            "s": list(scopes),
            "e": int(time.time()) + max(1, int(ttl_s)),
            "o": origin,
            "n": secrets.token_urlsafe(12),
        }
        body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
        sig = _b64(hmac.new(self.secret, body.encode(), hashlib.sha256).digest())
        return f"{body}.{sig}"

    def verify(self, ticket: str) -> dict | None:
        try:
            body, sig = ticket.split(".", 1)
            expected = _b64(hmac.new(self.secret, body.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(sig, expected):
                return None
            payload = json.loads(_unb64(body))
            if int(payload.get("e", 0)) < time.time():
                return None
            return payload
        except Exception:
            return None
