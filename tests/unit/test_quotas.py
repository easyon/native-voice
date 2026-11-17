from __future__ import annotations

from pathlib import Path

import pytest

from native_voice.config.schema import QuotaConfig
from native_voice.security.auth import Principal
from native_voice.security.keystore import KeyStore
from native_voice.security.quotas import QuotaExceeded, QuotaManager


def make_quota(tmp_path: Path, **kw) -> QuotaManager:
    cfg = QuotaConfig(max_concurrent_tasks=1, max_concurrent_per_key=1, rpm_per_key=2, **kw)
    return QuotaManager(cfg, KeyStore(tmp_path / "q.db"))


def test_concurrency_lease(tmp_path: Path):
    q = make_quota(tmp_path)
    p = Principal("k", ("asr",), None, None)
    lease = q.acquire(p)
    with pytest.raises(QuotaExceeded):
        q.acquire(p)
    q.release(lease)
    q.release(lease)  # idempotent
    q.acquire(p)


def test_rpm(tmp_path: Path):
    q = make_quota(tmp_path)
    p = Principal("k", ("asr",), rpm=2, max_concurrent=None)
    for _ in range(2):
        q.release(q.acquire(p))
    with pytest.raises(QuotaExceeded) as exc:
        q.acquire(p)
    assert exc.value.reason == "rpm"


def test_daily_quota(tmp_path: Path):
    q = make_quota(tmp_path)
    q.store.record_usage("k", "characters", 100)
    q.cfg.tts_chars_per_day = 10
    with pytest.raises(QuotaExceeded) as exc:
        q.check_daily(Principal("k", ("tts",), None, None), "tts")
    assert exc.value.reason == "daily"
