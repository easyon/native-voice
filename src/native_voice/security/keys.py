"""API key format and password hashing (stdlib only)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

PREFIX = "nv_"
_ITERATIONS = 200_000


def generate_key() -> tuple[str, str, str]:
    """Return (key_id, secret, full_token). Only the token is shown to the user."""
    key_id = secrets.token_hex(8)
    secret = secrets.token_urlsafe(32)
    return key_id, secret, f"{PREFIX}{key_id}_{secret}"


def parse_token(token: str) -> tuple[str, str] | None:
    if not token.startswith(PREFIX):
        return None
    rest = token[len(PREFIX):]
    if "_" not in rest:
        return None
    key_id, secret = rest.split("_", 1)
    if not key_id or not secret:
        return None
    return key_id, secret


def hash_secret(secret: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", secret.encode(), salt, _ITERATIONS)
    salt_b64 = base64.urlsafe_b64encode(salt).decode()
    digest_b64 = base64.urlsafe_b64encode(digest).decode()
    return f"pbkdf2_sha256${_ITERATIONS}${salt_b64}${digest_b64}"


def verify_secret(secret: str, encoded: str) -> bool:
    try:
        scheme, iterations, salt_b64, digest_b64 = encoded.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_b64.encode())
        expected = base64.urlsafe_b64decode(digest_b64.encode())
        actual = hashlib.pbkdf2_hmac("sha256", secret.encode(), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False
