from __future__ import annotations

import sys
import types

import tts_runtime.engines.cosyvoice_compat as compat


class FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_vllm_sampling_gets_request_seed(monkeypatch) -> None:
    fake = types.SimpleNamespace(SamplingParams=FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setattr(compat, "_VLLM_SAMPLING_PATCHED", False)

    compat.set_current_seed(7)
    assert compat.install_vllm_sampling_defaults() is True

    params = fake.SamplingParams(top_k=25, stop_token_ids=[1], max_tokens=10)
    assert params.kwargs["top_k"] == 25
    assert params.kwargs["seed"] == 7
    assert "top_p" not in params.kwargs


def test_optional_top_p_cap(monkeypatch) -> None:
    fake = types.SimpleNamespace(SamplingParams=FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setattr(compat, "_VLLM_SAMPLING_PATCHED", False)

    compat.set_current_seed(7)
    compat.install_vllm_sampling_defaults(top_p=0.9)
    params = fake.SamplingParams(top_k=25)

    assert params.kwargs["top_p"] == 0.9
    assert params.kwargs["seed"] == 7


def test_explicit_sampling_seed_wins(monkeypatch) -> None:
    fake = types.SimpleNamespace(SamplingParams=FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setattr(compat, "_VLLM_SAMPLING_PATCHED", False)

    compat.set_current_seed(7)
    compat.install_vllm_sampling_defaults()
    params = fake.SamplingParams(seed=42)

    assert params.kwargs["seed"] == 42
    assert "top_p" not in params.kwargs


def test_sampling_patch_is_a_noop_without_vllm(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setattr(compat, "_VLLM_SAMPLING_PATCHED", False)

    assert compat.install_vllm_sampling_defaults() is False
