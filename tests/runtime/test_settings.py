from __future__ import annotations

from pathlib import Path

import pytest
from cosyvoice_runtime.settings import load_config

WORKER_ENV = (
    "COSYVOICE_MAX_CONCURRENT",
    "COSYVOICE_TRT_CONCURRENT",
    "COSYVOICE_LOAD_TRT",
    "COSYVOICE_LOAD_VLLM",
)


@pytest.fixture(autouse=True)
def _clear_worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in WORKER_ENV:
        monkeypatch.delenv(name, raising=False)


def _write_config(path: Path) -> Path:
    path.write_text(
        "engine:\n"
        "  type: mock\n"
        "  cosyvoice:\n"
        "    max_concurrent: 2\n"
        "    load_trt: false\n",
        encoding="utf-8",
    )
    return path


def test_worker_env_overrides_engine_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path / "runtime.yaml")
    monkeypatch.setenv("COSYVOICE_MAX_CONCURRENT", "1")
    monkeypatch.setenv("COSYVOICE_TRT_CONCURRENT", "4")
    monkeypatch.setenv("COSYVOICE_LOAD_TRT", "true")
    monkeypatch.setenv("COSYVOICE_LOAD_VLLM", "0")

    config = load_config(config_path)

    assert config.engine.cosyvoice.max_concurrent == 1
    assert config.engine.cosyvoice.trt_concurrent == 4
    assert config.engine.cosyvoice.load_trt is True
    assert config.engine.cosyvoice.load_vllm is False


def test_unset_worker_env_keeps_yaml_values(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path / "runtime.yaml"))

    assert config.engine.cosyvoice.max_concurrent == 2
    assert config.engine.cosyvoice.trt_concurrent == 1
    assert config.engine.cosyvoice.load_trt is False
    assert config.engine.cosyvoice.load_vllm is False


def test_invalid_worker_env_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path / "runtime.yaml")
    monkeypatch.setenv("COSYVOICE_MAX_CONCURRENT", "many")

    with pytest.raises(ValueError, match="COSYVOICE_MAX_CONCURRENT"):
        load_config(config_path)
