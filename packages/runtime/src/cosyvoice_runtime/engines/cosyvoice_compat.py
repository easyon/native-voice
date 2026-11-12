"""Compatibility shims for running the official CosyVoice package in-process.

CosyVoice pins ``torch==2.3.1`` / ``torchaudio==2.3.1``. Newer torchaudio
(>= 2.9) removed the soundfile backend from ``torchaudio.load`` and routes all
I/O through TorchCodec, which is built against a specific torch/CUDA/FFmpeg
combination and often does not match the deployment. Prompt wavs are plain
audio files, so we decode them with soundfile instead of changing the
checkpoint or the caller contract.
"""

from __future__ import annotations

import logging
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


def install_wetext_assets(directory: str | None = None) -> bool:
    """Point wetext at a pre-provisioned model directory.

    ``wetext.Normalizer`` resolves its Kaldifst grammars via
    ``modelscope.snapshot_download("pengzhendong/wetext")`` on first use. In
    private networks that call can be blocked or rate limited, which leaves
    CosyVoice without text normalization ("no frontend is avaliable").

    Setting ``COSYVOICE_WETEXT_DIR`` (or passing ``directory``) to a checkout
    of that repo with ``en/tn`` and ``zh/tn`` grammars removes the runtime
    dependency on ModelScope entirely.
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
