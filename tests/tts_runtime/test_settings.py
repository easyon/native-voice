from __future__ import annotations

from pathlib import Path

import pytest
from tts_runtime.settings import load_config

WORKER_ENV = (
    "COSYVOICE_MAX_CONCURRENT",
    "COSYVOICE_TRT_CONCURRENT",
    "COSYVOICE_LOAD_TRT",
    "COSYVOICE_LOAD_VLLM",
    "COSYVOICE_VLLM_GPU_MEMORY_UTILIZATION",
    "COSYVOICE_VLLM_MAX_MODEL_LEN",
    "COSYVOICE_VLLM_MAX_NUM_SEQS",
    "COSYVOICE_VLLM_MAX_NUM_BATCHED_TOKENS",
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


def test_vllm_sizing_defaults_are_applied(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path / "runtime.yaml"))

    vllm = config.engine.cosyvoice.vllm
    assert vllm.gpu_memory_utilization == 0.5
    assert vllm.max_model_len == 8192
    assert vllm.max_num_seqs == 32
    assert vllm.max_num_batched_tokens == 8192
    assert vllm.overrides()["gpu_memory_utilization"] == 0.5


def test_vllm_env_overrides_sizing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path / "runtime.yaml")
    monkeypatch.setenv("COSYVOICE_VLLM_GPU_MEMORY_UTILIZATION", "0.35")
    monkeypatch.setenv("COSYVOICE_VLLM_MAX_MODEL_LEN", "4096")
    monkeypatch.setenv("COSYVOICE_VLLM_MAX_NUM_SEQS", "16")
    monkeypatch.setenv("COSYVOICE_VLLM_MAX_NUM_BATCHED_TOKENS", "4096")

    config = load_config(config_path)

    vllm = config.engine.cosyvoice.vllm
    assert vllm.gpu_memory_utilization == 0.35
    assert vllm.max_model_len == 4096
    assert vllm.max_num_seqs == 16
    assert vllm.max_num_batched_tokens == 4096


def test_invalid_vllm_env_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path / "runtime.yaml")
    monkeypatch.setenv("COSYVOICE_VLLM_GPU_MEMORY_UTILIZATION", "lots")

    with pytest.raises(
        ValueError, match="COSYVOICE_VLLM_GPU_MEMORY_UTILIZATION"
    ):
        load_config(config_path)


def test_invalid_worker_env_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path / "runtime.yaml")
    monkeypatch.setenv("COSYVOICE_MAX_CONCURRENT", "many")

    with pytest.raises(ValueError, match="COSYVOICE_MAX_CONCURRENT"):
        load_config(config_path)
