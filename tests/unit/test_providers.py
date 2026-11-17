from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from native_voice.config import load_config
from native_voice.config.schema import EngineConfig, ProviderSpec, ServiceConfig
from native_voice.models.asr.factory import create_asr_engine
from native_voice.models.asr.qwen3_asr import Qwen3ASRConfig, Qwen3ASRSession
from native_voice.models.tts.factory import create_tts_model
from native_voice.service.engine_supervisor import EngineSupervisor, _interpreter


def test_provider_registry_and_factories():
    cfg = load_config("configs/service.yaml")

    assert {"cosyvoice3", "qwen3tts", "qwen3tts_vllm", "cosyvoice3_vllm"} <= set(cfg.providers["tts"])
    assert {"paraformer", "qwen3asr"} <= set(cfg.providers["asr"])
    assert cfg.providers["tts"]["qwen3tts"].python.endswith(("python.exe", "python"))

    assert type(create_tts_model(cfg, "cosyvoice3")).__name__ == "CosyVoice3"
    assert type(create_tts_model(cfg, "qwen3tts")).__name__ == "Qwen3TTS"
    assert type(create_tts_model(cfg, "qwen3tts_vllm")).__name__ == "Qwen3TTSVLLM"
    assert type(create_tts_model(cfg, "cosyvoice3_vllm")).__name__ == "CosyVoice3VLLM"
    assert type(create_asr_engine(cfg, "paraformer")).__name__ == "StreamingParaformer"
    assert type(create_asr_engine(cfg, "qwen3asr")).__name__ == "Qwen3ASR"

    with pytest.raises(ValueError):
        create_tts_model(cfg, "no-such-provider")


def test_env_selects_provider(monkeypatch):
    monkeypatch.setenv("NV_TTS_PROVIDER", "qwen3tts")
    monkeypatch.setenv("NV_ASR_PROVIDER", "qwen3asr")
    cfg = load_config("configs/service.yaml")
    assert cfg.engine.tts_provider == "qwen3tts"
    assert cfg.engine.asr_provider == "qwen3asr"


def test_env_places_llm_on_cpu(monkeypatch, tmp_path):
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"")
    monkeypatch.setenv("NV_LLM_GPU_LAYERS", "0")
    monkeypatch.setenv("NV_LLM_SERVER", str(server))
    cfg = load_config("configs/service.yaml")
    assert cfg.llm.n_gpu_layers == 0
    assert cfg.llm.server == server


class _FakeASREngine:
    def __init__(self):
        self.cfg = Qwen3ASRConfig(partial_interval_ms=0, max_partial_seconds=8.0)

    def transcribe(self, audio):
        return f"samples={len(audio)}"


def test_qwen3_asr_session_finalize_uses_full_buffer():
    session = Qwen3ASRSession(_FakeASREngine())
    session.feed(np.zeros(1600, dtype=np.float32))
    session.feed(np.zeros(800, dtype=np.float32))
    assert session.finalize() == "samples=2400"

    session.reset()
    assert session.finalize() == ""


def test_qwen3tts_token_cap_bounds_runaway_generation():
    from native_voice.models.tts.qwen3tts import Qwen3TTS, Qwen3TTSConfig

    model = Qwen3TTS(
        Qwen3TTSConfig(
            model_dir="unused",
            prompt_wav=Path("unused.wav"),
            prompt_text="unused",
            max_new_tokens=4096,
        )
    )
    short = model._max_tokens("你好。")
    long = model._max_tokens("这是一句明显更长的话，用来验证上限随文本增长。" * 3)
    assert 32 <= short < long < 4096
    assert model._max_tokens("x" * 5000) == 4096


def test_supervisor_maps_venv_layout_across_platforms(tmp_path):
    env = tmp_path / "envs" / "qwen3tts"
    linux_py = env / "bin" / "python"
    linux_py.parent.mkdir(parents=True)
    linux_py.write_text("", encoding="utf-8")
    assert _interpreter(str(env / "Scripts" / "python.exe")) == str(linux_py)

    windows_py = tmp_path / "envs" / "qwen3asr" / "Scripts" / "python.exe"
    windows_py.parent.mkdir(parents=True)
    windows_py.write_text("", encoding="utf-8")
    assert _interpreter(str(windows_py.parent.parent / "bin" / "python")) == str(windows_py)

    missing = str(tmp_path / "nope" / "bin" / "python")
    assert _interpreter(missing) == missing


def test_supervisor_spawns_selected_provider_python(tmp_path, monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            self.pid = 1234

        def poll(self):
            return None

    monkeypatch.setattr(
        "native_voice.service.engine_supervisor.subprocess.Popen", FakePopen
    )
    cfg = ServiceConfig(
        root=tmp_path,
        engine=EngineConfig(mode="spawn", tts_provider="qwen3tts"),
    )
    cfg.providers["tts"] = {
        "qwen3tts": ProviderSpec(
            name="qwen3tts",
            python=sys.executable,
            module="native_voice.engine.cli",
        )
    }
    supervisor = EngineSupervisor(cfg, tmp_path / "service.yaml")
    supervisor._spawn("tts", 8102)
    supervisor.procs[0][1].close()

    args = captured["args"]
    assert args[0] == sys.executable
    assert args[1:3] == ["-m", "native_voice.engine.cli"]
    assert args[3] == "tts"
    assert args[-2:] == ["--provider", "qwen3tts"]
    assert captured["kwargs"]["cwd"] == str(tmp_path)
