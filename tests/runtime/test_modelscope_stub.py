from __future__ import annotations

import sys
import types

import pytest
from cosyvoice_runtime.engines import cosyvoice_compat as compat


@pytest.fixture(autouse=True)
def _clean_modules(monkeypatch: pytest.MonkeyPatch):
    """Keep each test isolated from the real (or stubbed) module cache."""
    monkeypatch.delitem(sys.modules, "modelscope", raising=False)


def test_stub_is_registered_when_modelscope_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str, package: str | None = None) -> types.ModuleType:
        raise ImportError(name)

    monkeypatch.setattr(compat.importlib, "import_module", missing)

    assert compat.install_modelscope_stub() is True
    assert "modelscope" in sys.modules


def test_stub_snapshot_download_names_the_misconfiguration() -> None:
    compat.install_modelscope_stub()

    from modelscope import snapshot_download

    with pytest.raises(RuntimeError, match="local checkpoint directories"):
        snapshot_download("iic/CosyVoice3-0.5B")


def test_real_modelscope_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    real = types.ModuleType("modelscope")
    sentinel = object()
    real.snapshot_download = sentinel
    monkeypatch.setitem(sys.modules, "modelscope", real)

    assert compat.install_modelscope_stub() is False
    import modelscope

    assert modelscope.snapshot_download is sentinel


def test_stub_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str, package: str | None = None) -> types.ModuleType:
        raise ImportError(name)

    monkeypatch.setattr(compat.importlib, "import_module", missing)

    assert compat.install_modelscope_stub() is True
    first = sys.modules["modelscope"]
    assert compat.install_modelscope_stub() is False
    assert sys.modules["modelscope"] is first
