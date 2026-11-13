"""Compatibility shims for running the official CosyVoice package in-process.

CosyVoice pins ``torch==2.3.1`` / ``torchaudio==2.3.1``. Newer torchaudio
(>= 2.9) removed the soundfile backend from ``torchaudio.load`` and routes all
I/O through TorchCodec, which is built against a specific torch/CUDA/FFmpeg
combination and often does not match the deployment. Prompt wavs are plain
audio files, so we decode them with soundfile instead of changing the
checkpoint or the caller contract.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from typing import Any

logger = logging.getLogger("cosyvoice_runtime.engine")


def _soundfile_load_wav(wav: Any, target_sr: int, min_sr: int = 16_000) -> Any:
    import soundfile
    import torch
    import torchaudio

    data, sample_rate = soundfile.read(str(wav), dtype="float32", always_2d=True)
    # (channels, time); upstream averages to mono and keeps the channel dim.
    speech = torch.from_numpy(data.T.copy()).mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate >= min_sr, (
            f"wav sample rate {sample_rate} must be greater than {target_sr}"
        )
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(speech)
    return speech


def install_wav_loader() -> bool:
    """Route CosyVoice prompt-wav decoding through soundfile.

    Safe to call repeatedly; returns False when torch/soundfile are absent and
    the upstream loader should be left alone.
    """
    try:
        import soundfile  # noqa: F401
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
    except ImportError:
        return False

    try:
        from cosyvoice.cli import frontend
        from cosyvoice.utils import file_utils
    except ImportError:
        return False

    file_utils.load_wav = _soundfile_load_wav
    frontend.load_wav = _soundfile_load_wav
    logger.debug("installed soundfile-based wav loader for CosyVoice")
    return True


def install_modelscope_stub() -> bool:
    """Make ``import modelscope`` succeed when only local checkpoints are used.

    Upstream ``cosyvoice/cli/cosyvoice.py`` does ``from modelscope import
    snapshot_download`` at import time and calls it only when ``model_dir`` is
    not an existing path. Deployments here always pass a local directory, so
    the import is the only thing that has to work; keeping ModelScope installed
    otherwise costs >100MB and, before ``wetext`` 0.2.1, its authenticated
    snapshot calls were the failure that broke text normalization.

    A real ``modelscope`` installation always wins: the stub is only registered
    when the import fails, and ``snapshot_download`` raises instead of silently
    returning a wrong path.
    """
    if "modelscope" in sys.modules:
        return False
    try:
        importlib.import_module("modelscope")
    except ImportError:
        pass
    else:
        return False

    def snapshot_download(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "modelscope is not installed and this deployment resolves only "
            "local checkpoint directories; pass the extracted model directory "
            "instead of a ModelScope repo id"
        )

    module = types.ModuleType("modelscope")
    module.__doc__ = "placeholder installed by cosyvoice_runtime.engines.cosyvoice_compat"
    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    sys.modules["modelscope"] = module
    logger.debug("registered a modelscope placeholder (local checkpoints only)")
    return True


def install_wetext_assets(directory: str | None = None) -> bool:
    """Point wetext at a pre-provisioned model directory (legacy).

    ``wetext`` 0.0.4 resolves its Kaldifst grammars via
    ``modelscope.snapshot_download("pengzhendong/wetext")`` on first use, which
    is blocked or rate limited in private networks and left CosyVoice without
    text normalization ("no frontend is avaliable"). ``requirements-cosyvoice.lock``
    pins wetext 0.2.1, which bundles every grammar in ``wetext/fsts``, so this
    shim is only needed when someone deliberately downgrades to 0.0.4.

    Setting ``COSYVOICE_WETEXT_DIR`` (or passing ``directory``) to a checkout
    of that repo with ``en/tn`` and ``zh/tn`` grammars removes the runtime
    dependency on ModelScope. Returns False when no usable directory is
    configured, which is the normal case for wetext >= 0.2.
    """
    import os

    path = directory or os.getenv("COSYVOICE_WETEXT_DIR", "")
    path = (path or "").strip()
    if not path or not os.path.isdir(path):
        return False
    try:
        import wetext.wetext as _wetext
    except ImportError:
        return False
    if not getattr(_wetext, "snapshot_download", None):
        return False
    _wetext.snapshot_download = lambda *args, **kwargs: path
    logger.info("wetext assets pinned to %s", path)
    return True


_VLLM_OVERRIDES: dict[str, Any] = {}
_VLLM_PATCHED = False


def install_vllm_engine_overrides(overrides: dict[str, Any] | None = None) -> bool:
    """Force vLLM sizing that upstream's ``load_vllm`` does not expose.

    ``CosyVoiceModel.load_vllm`` constructs ``EngineArgs`` with a hardcoded
    ``gpu_memory_utilization=0.2`` and never sets ``max_model_len`` /
    ``max_num_seqs`` / ``max_num_batched_tokens``. On a 32GB card with the
    flow and HiFi-GAN models resident that budget cannot cover weights plus
    the torch.compile activation peak, and the engine aborts in KV-cache
    initialization. The values are not constructor arguments, so rewrite the
    ``EngineArgs`` just before ``LLMEngine.from_engine_args`` reads them.

    Safe to call repeatedly; a later call replaces the values used by engines
    constructed afterwards. Returns False when vLLM is not importable (the
    engine layer already reports that condition) or nothing is configured.
    """
    global _VLLM_PATCHED
    if overrides:
        _VLLM_OVERRIDES.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
    if not _VLLM_OVERRIDES:
        return False
    try:
        import vllm
    except ImportError:
        return False
    if _VLLM_PATCHED:
        return True

    original = vllm.LLMEngine.from_engine_args

    def from_engine_args(engine_args: Any) -> Any:
        # Guard against a vLLM upgrade renaming a field: setting an unknown
        # attribute on EngineArgs would silently do nothing.
        fields = getattr(type(engine_args), "__dataclass_fields__", {})
        for key, value in _VLLM_OVERRIDES.items():
            if fields and key not in fields:
                logger.warning("vLLM EngineArgs has no field %r; ignoring", key)
                continue
            setattr(engine_args, key, value)
        logger.info("vLLM engine sizing: %s", dict(_VLLM_OVERRIDES))
        return original(engine_args)

    vllm.LLMEngine.from_engine_args = staticmethod(from_engine_args)
    _VLLM_PATCHED = True
    logger.debug("patched vllm.LLMEngine.from_engine_args")
    return True
