from __future__ import annotations

import sys
import types

import pytest
from tts_runtime.engines import cosyvoice_compat as compat


class _FakeEngineArgs:
    """Mimics vllm.EngineArgs for the fields the runtime rewrites."""

    gpu_memory_utilization = 0.2
    max_model_len = 32768
    max_num_seqs = 256
    max_num_batched_tokens = 8192


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch):
    seen: list[object] = []

    class FakeLLMEngine:
        @classmethod
        def from_engine_args(cls, engine_args: object) -> str:
            seen.append(engine_args)
            return "engine"

    module = types.ModuleType("vllm")
    module.LLMEngine = FakeLLMEngine
    monkeypatch.setitem(sys.modules, "vllm", module)
    # The patch is process-wide; reset it so each test starts from the
    # unpatched class method.
    monkeypatch.setattr(compat, "_VLLM_PATCHED", False)
    monkeypatch.setattr(compat, "_VLLM_OVERRIDES", {})
    return FakeLLMEngine, seen


def test_install_applies_sizing_to_engine_args(fake_vllm) -> None:
    cls, seen = fake_vllm
    assert compat.install_vllm_engine_overrides(
        {
            "gpu_memory_utilization": 0.45,
            "max_model_len": 4096,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 8192,
        }
    )

    args = _FakeEngineArgs()
    assert cls.from_engine_args(args) == "engine"
    assert seen == [args]
    assert args.gpu_memory_utilization == 0.45
    assert args.max_model_len == 4096
    assert args.max_num_seqs == 16
    assert args.max_num_batched_tokens == 8192


def test_later_install_replaces_values_without_repatching(fake_vllm) -> None:
    cls, _ = fake_vllm
    compat.install_vllm_engine_overrides({"max_model_len": 4096})
    compat.install_vllm_engine_overrides({"max_model_len": 2048})

    args = _FakeEngineArgs()
    cls.from_engine_args(args)
    assert args.max_model_len == 2048


def test_none_values_are_ignored(fake_vllm) -> None:
    cls, _ = fake_vllm
    assert compat.install_vllm_engine_overrides(
        {"max_model_len": None, "max_num_seqs": 8}
    )

    args = _FakeEngineArgs()
    cls.from_engine_args(args)
    assert args.max_model_len == 32768
    assert args.max_num_seqs == 8


def test_no_overrides_or_missing_vllm_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compat, "_VLLM_PATCHED", False)
    monkeypatch.setattr(compat, "_VLLM_OVERRIDES", {})
    assert compat.install_vllm_engine_overrides({}) is False

    monkeypatch.setitem(sys.modules, "vllm", None)
    assert compat.install_vllm_engine_overrides({"max_model_len": 4096}) is False
