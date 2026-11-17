"""SenseVoiceSmall ASR via FunASR (GPU when available)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .config import Config

# SenseVoice occasionally hallucinates these on near-silence / echo tails.
_JUNK = {
    "谢谢观看", "谢谢大家", "请不吝点赞", "字幕由", "字幕提供", "谢谢", "。", "，", "！", "?",
}
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️‍]+")


class ASR:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None

    def load(self):
        from funasr import AutoModel

        device = self.cfg.asr_device
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"
        model = self.cfg.asr_model
        local = Path.home() / ".cache" / "modelscope" / "hub" / "models" / model
        if local.exists():
            model = str(local)
        print(f"[ASR] 加载 SenseVoiceSmall ({device}) ...", flush=True)
        self.model = AutoModel(
            model=model,
            device=device,
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
        )
        # Warm up kernels so the first real utterance is not several times slower.
        try:
            self.model.generate(
                input=np.zeros(int(0.5 * self.cfg.input_rate), dtype=np.float32),
                fs=self.cfg.input_rate,
                language=self.cfg.asr_language,
                use_itn=True,
            )
        except Exception:
            pass
        print("[ASR] 就绪", flush=True)

    def transcribe(self, audio: np.ndarray) -> str:
        if self.model is None:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < int(0.15 * self.cfg.input_rate):
            return ""
        try:
            result = self.model.generate(
                input=audio,
                fs=self.cfg.input_rate,
                language=self.cfg.asr_language,
                use_itn=True,
                batch_size_s=60,
            )
        except TypeError:
            result = self.model.generate(
                input=audio,
                language=self.cfg.asr_language,
                use_itn=True,
                batch_size_s=60,
            )
        if not result:
            return ""
        text = result[0].get("text", "")
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            text = rich_transcription_postprocess(text)
        except Exception:
            pass
        text = _EMOJI.sub("", str(text))
        text = " ".join(text.replace("<|", " ").replace("|>", " ").split()).strip()
        if text in _JUNK or len(text.replace("。", "").replace("，", "").strip()) == 0:
            return ""
        return text
