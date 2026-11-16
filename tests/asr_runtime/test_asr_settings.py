from __future__ import annotations

from pathlib import Path

import pytest
from funasr_runtime.settings import RuntimeConfig, load_config
from pydantic import ValidationError

_ENV_VARS = [
    "FUNASR_RUNTIME_CONFIG",
    "FUNASR_ENGINE_TYPE",
    "FUNASR_API_KEYS",
    "FUNASR_METRICS_TOKEN",
    "FUNASR_MODELS_DIR",
    "FUNASR_DEVICE",
    "FUNASR_PORT",
    "FUNASR_MAX_CONCURRENT_STREAMS",
    "FUNASR_PARTIAL_INTERVAL_MS",
    "FUNASR_PARTIAL_WINDOW_S",
    "FUNASR_OFFLINE_MAX_UPLOAD_BYTES",
    "FUNASR_LOG_LEVEL",
    "FUNASR_LOG_FORMAT",
    "FUNASR_ENVIRONMENT",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "funasr.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_default_config() -> None:
    config = RuntimeConfig.default()
    assert config.engine.type == "mock"
    assert "fun-asr-realtime" in config.models
    assert config.aliases["whisper-1"] == "fun-asr-realtime"
    assert config.offline.default_model == "fun-asr-realtime"


def test_load_yaml_and_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(
        tmp_path,
        """
engine:
  type: mock
models:
  fun-asr-realtime:
    engine: mock
limits:
  max_concurrent_streams: 4
""",
    )
    monkeypatch.setenv("FUNASR_API_KEYS", "k1, k2")
    monkeypatch.setenv("FUNASR_DEVICE", "cuda:1")
    monkeypatch.setenv("FUNASR_MODELS_DIR", "/models")
    monkeypatch.setenv("FUNASR_PORT", "9001")
    monkeypatch.setenv("FUNASR_MAX_CONCURRENT_STREAMS", "12")
    monkeypatch.setenv("FUNASR_PARTIAL_INTERVAL_MS", "500")
    monkeypatch.setenv("FUNASR_PARTIAL_WINDOW_S", "4")
    monkeypatch.setenv("FUNASR_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("FUNASR_ENVIRONMENT", "prod")

    config = load_config(path)
    assert config.auth.enabled is True
    assert config.auth.api_keys == ["k1", "k2"]
    assert config.engine.funasr.device == "cuda:1"
    assert config.engine.funasr.model_root == "/models"
    assert config.engine.funasr.partial_interval_ms == 500
    assert config.engine.funasr.partial_window_s == 4.0
    assert config.server.port == 9001
    assert config.limits.max_concurrent_streams == 12
    assert config.logging.level == "DEBUG"
    assert config.environment == "prod"


def test_invalid_env_integer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(
        tmp_path,
        """
models:
  fun-asr-realtime:
    engine: mock
""",
    )
    monkeypatch.setenv("FUNASR_PORT", "not-a-port")
    with pytest.raises(ValueError, match="FUNASR_PORT"):
        load_config(path)


def test_model_engine_inherits_top_level() -> None:
    config = RuntimeConfig.model_validate(
        {
            "engine": {"type": "mock"},
            "models": {"m": {"artifact": ""}},
        }
    )
    assert config.models["m"].engine == "mock"


def test_unknown_model_engine_rejected() -> None:
    with pytest.raises(ValidationError, match="not supported"):
        RuntimeConfig.model_validate(
            {
                "engine": {"type": "mock"},
                "models": {"m": {"engine": "whisper"}},
            }
        )


def test_alias_must_target_model() -> None:
    with pytest.raises(ValidationError, match="does not name a configured model"):
        RuntimeConfig.model_validate(
            {
                "engine": {"type": "mock"},
                "models": {"m": {"engine": "mock"}},
                "aliases": {"x": "missing"},
            }
        )


def test_funasr_model_requires_artifact() -> None:
    with pytest.raises(ValidationError, match="artifact is required"):
        RuntimeConfig.model_validate(
            {
                "engine": {"type": "funasr"},
                "models": {"m": {"engine": "funasr"}},
            }
        )


@pytest.mark.parametrize(
    "name",
    [
        "funasr.yaml",
        "funasr.cpu.yaml",
        "funasr.gpu.yaml",
        "funasr.llamacpp.yaml",
    ],
)
def test_config_templates_load(name: str) -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / name
    config = load_config(path)
    assert config.models
    assert config.offline.default_model in config.models

