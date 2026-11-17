"""In-memory concurrency/RPM/daily quotas (Redis-backed later)."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from ..config.schema import QuotaConfig
from ..observability.metrics import METRICS
from .auth import Principal
from .keystore import KeyStore


class QuotaExceeded(Exception):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class QuotaLease:
    key_id: str
    kind: str
    released: bool = False


class QuotaManager:
    def __init__(self, cfg: QuotaConfig, store: KeyStore):
        self.cfg = cfg
        self.store = store
        self._lock = threading.Lock()
        self._active_key: dict[str, int] = defaultdict(int)
        self._active_total = 0
        self._rpm: dict[str, deque[float]] = defaultdict(deque)

    def acquire(self, principal: Principal) -> QuotaLease:
        key_limit = principal.max_concurrent or self.cfg.max_concurrent_per_key
        rpm_limit = principal.rpm or self.cfg.rpm_per_key
        now = time.monotonic()
        with self._lock:
            if self._active_total >= self.cfg.max_concurrent_tasks:
                METRICS.quota_rejections.labels("concurrency_total").inc()
                raise QuotaExceeded("concurrency", "service concurrency limit reached")
            if self._active_key[principal.key_id] >= key_limit:
                METRICS.quota_rejections.labels("concurrency_key").inc()
                raise QuotaExceeded("concurrency", "per-key concurrency limit reached")
            window = self._rpm[principal.key_id]
            while window and now - window[0] > 60.0:
                window.popleft()
            if len(window) >= rpm_limit:
                METRICS.quota_rejections.labels("rpm").inc()
                raise QuotaExceeded("rpm", "requests per minute limit reached")
            window.append(now)
            self._active_key[principal.key_id] += 1
            self._active_total += 1
        return QuotaLease(principal.key_id, "task")

    def check_daily(self, principal: Principal, kind: str) -> None:
        if kind == "tts":
            limit, unit = self.cfg.tts_chars_per_day, "characters"
        elif kind == "asr":
            limit, unit = self.cfg.asr_seconds_per_day, "seconds"
        else:
            return
        if limit <= 0:
            return
        used = self.store.usage_today(principal.key_id, unit)
        if used >= limit:
            METRICS.quota_rejections.labels("daily").inc()
            raise QuotaExceeded("daily", f"daily {unit} quota reached")

    def release(self, lease: QuotaLease) -> None:
        if lease.released:
            return
        with self._lock:
            lease.released = True
            self._active_key[lease.key_id] = max(0, self._active_key[lease.key_id] - 1)
            self._active_total = max(0, self._active_total - 1)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"active_total": self._active_total, "active_keys": len(self._active_key)}
