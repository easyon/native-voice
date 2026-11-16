from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest
from cosyvoice_common.errors import APIError
from pydantic import ValidationError
from tts_runtime.engines.cosyvoice import CosyVoiceEngine
from tts_runtime.settings import CosyVoiceEngineConfig, ModelConfig


def _install_fake_cosyvoice(
    monkeypatch: pytest.MonkeyPatch, factory: object
) -> None:
    fake_cv = types.ModuleType("cosyvoice.cli.cosyvoice")
    fake_cv.CosyVoice3 = factory
    fake_cv.CosyVoice2 = factory
    fake_cli = types.ModuleType("cosyvoice.cli")
    fake_cli.cosyvoice = fake_cv
    fake_pkg = types.ModuleType("cosyvoice")
    fake_pkg.cli = fake_cli
    monkeypatch.setitem(sys.modules, "cosyvoice", fake_pkg)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", fake_cv)


def _engine(
    tmp_path: Path, config: CosyVoiceEngineConfig
) -> tuple[CosyVoiceEngine, ModelConfig]:
    model_cfg = ModelConfig(engine="cosyvoice3", artifact=str(tmp_path))
    engine = CosyVoiceEngine(config, {"m": model_cfg})
    return engine, model_cfg


def test_engine_passes_acceleration_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """load_vllm/load_trt/trt_concurrent must reach the upstream constructor."""
    seen: dict[str, object] = {}

    class FakeModel:
        sample_rate = 24000

    # The engine filters kwargs by the upstream constructor signature, so the
    # fake must declare them explicitly (a bare **kwargs is not enough).
    def factory(
        model_dir: str,
        load_trt: bool = False,
        load_vllm: bool = False,
        fp16: bool = False,
        trt_concurrent: int = 1,
    ) -> FakeModel:
        seen.update(
            load_trt=load_trt,
            load_vllm=load_vllm,
            fp16=fp16,
            trt_concurrent=trt_concurrent,
        )
        return FakeModel()

    _install_fake_cosyvoice(monkeypatch, factory)
    monkeypatch.setattr(
        CosyVoiceEngine, "_module_available", staticmethod(lambda name: True)
    )
    (tmp_path / "flow.decoder.estimator.fp32.onnx").write_bytes(b"onnx")

    engine, _ = _engine(
        tmp_path,
        CosyVoiceEngineConfig(
            preload=True,
            load_vllm=True,
            load_trt=True,
            trt_concurrent=3,
        ),
    )
    asyncio.run(engine.startup())

    assert seen["load_vllm"] is True
    assert seen["load_trt"] is True
    assert seen["trt_concurrent"] == 3
    assert seen["fp16"] is False


def test_load_vllm_requires_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeModel:
        sample_rate = 24000

    _install_fake_cosyvoice(monkeypatch, lambda model_dir, **kwargs: FakeModel())
    monkeypatch.setattr(
        CosyVoiceEngine, "_module_available", staticmethod(lambda name: False)
    )
    engine, _ = _engine(
        tmp_path, CosyVoiceEngineConfig(preload=True, load_vllm=True)
    )

    with pytest.raises(APIError) as excinfo:
        asyncio.run(engine.startup())
    assert excinfo.value.status_code == 503
    assert "vllm is not installed" in excinfo.value.message


def test_load_trt_requires_onnx_or_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeModel:
        sample_rate = 24000

    _install_fake_cosyvoice(monkeypatch, lambda model_dir, **kwargs: FakeModel())
    monkeypatch.setattr(
        CosyVoiceEngine, "_module_available", staticmethod(lambda name: True)
    )
    engine, _ = _engine(
        tmp_path, CosyVoiceEngineConfig(preload=True, load_trt=True)
    )

    with pytest.raises(APIError) as excinfo:
        asyncio.run(engine.startup())
    assert excinfo.value.status_code == 503
    assert "fp32.onnx" in excinfo.value.message


def test_acceleration_off_by_default_does_not_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeModel:
        sample_rate = 24000

    _install_fake_cosyvoice(monkeypatch, lambda model_dir, **kwargs: FakeModel())
    engine, _ = _engine(tmp_path, CosyVoiceEngineConfig(preload=True))
    asyncio.run(engine.startup())


def test_trt_concurrent_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        CosyVoiceEngineConfig(trt_concurrent=0)
