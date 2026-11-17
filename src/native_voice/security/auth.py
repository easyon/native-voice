"""Bearer authentication and scope checks."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .keystore import ApiKeyRecord, KeyStore


class AuthError(Exception):
    def __init__(self, message: str, code: str = "Unauthorized", status: int = 401):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class Principal:
    key_id: str
    scopes: tuple[str, ...]
    rpm: int | None
    max_concurrent: int | None

    def has(self, scope: str) -> bool:
        return scope in self.scopes or "admin" in self.scopes or "*" in self.scopes


def _bearer(headers: Mapping[str, str]) -> str | None:
    for key in ("authorization", "Authorization"):
        value = headers.get(key)
        if value:
            value = value.strip()
            if value.lower().startswith("bearer "):
                return value[7:].strip()
    return None


class Authenticator:
    def __init__(self, store: KeyStore):
        self.store = store

    def authenticate_token(self, token: str) -> Principal:
        record: ApiKeyRecord | None = self.store.verify_token(token)
        if record is None:
            raise AuthError("invalid or expired API key")
        return Principal(record.key_id, record.scopes, record.rpm, record.max_concurrent)

    def authenticate_headers(self, headers: Mapping[str, str]) -> Principal:
        token = _bearer(headers)
        if not token:
            raise AuthError("missing Authorization: Bearer <api_key>")
        return self.authenticate_token(token)

    @staticmethod
    def require(principal: Principal, scope: str) -> None:
        if not principal.has(scope):
            raise AuthError(f"API key lacks scope '{scope}'", code="Forbidden", status=403)
