"""SQLite-backed API key and usage store (single-node default)."""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .keys import generate_key, hash_secret, parse_token, verify_secret

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiKeyRecord:
    key_id: str
    name: str
    scopes: tuple[str, ...]
    rpm: int | None
    max_concurrent: int | None
    enabled: bool
    expires_at: float | None
    created_at: float


class KeyStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    secret_hash TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    scopes TEXT NOT NULL DEFAULT '',
                    rpm INTEGER,
                    max_concurrent INTEGER,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    expires_at REAL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS usage_daily (
                    day TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    amount REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (day, key_id, kind)
                );
                """
            )

    @staticmethod
    def _row(row: sqlite3.Row) -> ApiKeyRecord:
        return ApiKeyRecord(
            key_id=row["key_id"],
            name=row["name"],
            scopes=tuple(s for s in (row["scopes"] or "").split(",") if s),
            rpm=row["rpm"],
            max_concurrent=row["max_concurrent"],
            enabled=bool(row["enabled"]),
            expires_at=row["expires_at"],
            created_at=row["created_at"],
        )

    def create_key(
        self,
        name: str,
        scopes: list[str],
        rpm: int | None = None,
        max_concurrent: int | None = None,
        expires_at: float | None = None,
    ) -> tuple[ApiKeyRecord, str]:
        key_id, secret, token = generate_key()
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO api_keys(key_id, secret_hash, name, scopes, rpm, max_concurrent,"
                " enabled, expires_at, created_at) VALUES (?,?,?,?,?,?,1,?,?)",
                (key_id, hash_secret(secret), name, ",".join(scopes), rpm, max_concurrent, expires_at, now),
            )
            row = self._conn.execute("SELECT * FROM api_keys WHERE key_id=?", (key_id,)).fetchone()
        return self._row(row), token

    def get(self, key_id: str) -> ApiKeyRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM api_keys WHERE key_id=?", (key_id,)).fetchone()
        return self._row(row) if row else None

    def list_keys(self) -> list[ApiKeyRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM api_keys ORDER BY created_at").fetchall()
        return [self._row(r) for r in rows]

    def set_enabled(self, key_id: str, enabled: bool) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE api_keys SET enabled=? WHERE key_id=?",
                (1 if enabled else 0, key_id),
            )
        return cur.rowcount > 0

    def verify_token(self, token: str) -> ApiKeyRecord | None:
        parsed = parse_token(token)
        if not parsed:
            return None
        key_id, secret = parsed
        with self._lock:
            row = self._conn.execute("SELECT * FROM api_keys WHERE key_id=?", (key_id,)).fetchone()
        if not row or not row["enabled"]:
            return None
        if row["expires_at"] is not None and float(row["expires_at"]) < time.time():
            return None
        if not verify_secret(secret, row["secret_hash"]):
            return None
        return self._row(row)

    def record_usage(self, key_id: str, kind: str, amount: float) -> None:
        day = date.today().isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO usage_daily(day, key_id, kind, amount) VALUES (?,?,?,?) "
                "ON CONFLICT(day, key_id, kind) DO UPDATE SET amount = amount + excluded.amount",
                (day, key_id, kind, float(amount)),
            )

    def usage_today(self, key_id: str, kind: str) -> float:
        day = date.today().isoformat()
        with self._lock:
            row = self._conn.execute(
                "SELECT amount FROM usage_daily WHERE day=? AND key_id=? AND kind=?", (day, key_id, kind)
            ).fetchone()
        return float(row["amount"]) if row else 0.0

    def ensure_dev_key(self) -> str | None:
        """In dev, create and log a full-scope key if the store is empty."""
        if self.list_keys():
            return None
        _, token = self.create_key(
            "dev",
            ["asr", "tts", "conversation", "admin"],
            rpm=600,
            max_concurrent=4,
        )
        log.warning("DEV API KEY (store it now; it will not be shown again): %s", token)
        print(f"\n[DEV API KEY] {token}\n", flush=True)
        return token

    def close(self) -> None:
        with self._lock:
            self._conn.close()
