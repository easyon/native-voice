from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from native_voice.api.app import create_app
from native_voice.config.schema import ServiceConfig
from native_voice.security.keystore import KeyStore


@pytest.fixture
def cfg(tmp_path: Path) -> ServiceConfig:
    c = ServiceConfig(root=Path(".").resolve())
    c.auth.db_path = tmp_path / "keys.db"
    c.logging.level = "WARNING"
    c.logging.json_logs = False
    c.server.ws_idle_timeout_s = 20
    c.tts.default_format = "pcm"
    c.tts.default_sample_rate = 24000
    c.models.root = c.root
    c.voice_store.dir = tmp_path / "voices"
    return c


@pytest.fixture
def token(cfg: ServiceConfig) -> str:
    store = KeyStore(cfg.auth.db_path)
    _, tok = store.create_key("test", ["asr", "tts", "conversation"])
    store.close()
    return tok


@pytest.fixture
def client(cfg: ServiceConfig):
    app = create_app(cfg, backend_kind="fake")
    with TestClient(app) as c:
        yield c
