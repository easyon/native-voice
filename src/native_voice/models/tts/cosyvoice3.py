"""CosyVoice3 streaming TTS runtime (one instance per process)."""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import numpy as np

from .config import TTSModelConfig
from .graph import GraphDecoder
from .sampling import patch_sampling

log = logging.getLogger(__name__)


class CosyVoice3:
    def __init__(self, cfg: TTSModelConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.tts = None
        self.sample_rate = 24000
        self._ready = False
        self._voice_names: list[str] = []
        self._voice_desc: dict[str, str] = {}
        self._warm_voice: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        import torch

        third_party = self.cfg.third_party_dir
        for p in (str(third_party), str(third_party / "third_party" / "Matcha-TTS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        # This service never uses CosyVoice's network-backed text normaliser.
        sys.modules.setdefault("wetext", None)

        import cosyvoice.cli.cosyvoice as cvcli
        from cosyvoice.cli.cosyvoice import AutoModel

        cvcli.tqdm = lambda it, *a, **k: it
        log.info("loading CosyVoice3 from %s", self.cfg.model_dir)
        self.tts = AutoModel(model_dir=str(self.cfg.model_dir), fp16=True)
        self.sample_rate = int(self.tts.sample_rate)

        voices = self.cfg.voices or {}
        if not voices:
            self._add_voice("default", self.cfg.prompt_text, self.cfg.prompt_wav, "默认音色")
        else:
            for name, profile in voices.items():
                self._add_voice(name, profile.prompt_text, profile.prompt_wav, profile.description)
        self._warm_voice = next(iter(self._voice_names), None)

        for name in ("llm", "flow", "hift"):
            getattr(self.tts.model, name).half()
        patch_sampling(self.tts.model.llm)
        if self.cfg.use_cuda_graph:
            try:
                GraphDecoder(self.tts.model.llm, self.cfg.max_cache_len, self.cfg.verbose)
            except Exception as exc:
                log.warning("CUDA graph unavailable, using eager decode: %r", exc)

        if self.cfg.warmup and self._warm_voice:
            self._warmup()
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            log.info("CosyVoice3 ready (VRAM %.2f/%.2f GB)", alloc, peak)
        self._ready = True

    def _add_voice(self, name: str, prompt_text: str, prompt_wav: Path, description: str = "") -> None:
        self.tts.add_zero_shot_spk(prompt_text, str(prompt_wav), name)
        if name not in self._voice_names:
            self._voice_names.append(name)
        self._voice_desc[name] = description or name

    def add_voice(self, name: str, prompt_text: str, prompt_wav: Path, description: str = "") -> None:
        """Register a zero-shot voice at runtime (serialized with synthesis)."""
        with self.lock:
            self._add_voice(name, prompt_text, prompt_wav, description)
        log.info("voice added: %s", name)

    def remove_voice(self, name: str) -> None:
        with self.lock:
            if name in self._voice_names:
                self._voice_names.remove(name)
            self._voice_desc.pop(name, None)
            try:
                self.tts.frontend.spk2info.pop(name, None)
            except Exception:
                pass
        log.info("voice removed: %s", name)

    def _warmup(self) -> None:
        log.info("warming up TTS")
        for _ in self.iter_synthesize("你好。", self._warm_voice, None):
            pass

    def voices(self) -> list[dict]:
        return [
            {"name": name, "description": self._voice_desc.get(name, name)}
            for name in self._voice_names
        ]

    def has_voice(self, name: str) -> bool:
        return name in self._voice_names

    def iter_synthesize(self, text: str, voice: str, cancel):
        """Yield 24 kHz float32 mono chunks. Drains silently on cancel."""
        text = (text or "").strip()
        if not text:
            return
        if not self.has_voice(voice):
            raise ValueError(f"unknown voice: {voice}")
        with self.lock:
            model = self.tts.model
            model.token_hop_len = self.cfg.hop
            try:
                for out in self.tts.inference_zero_shot(
                    text,
                    "",
                    "",
                    zero_shot_spk_id=voice,
                    stream=True,
                    text_frontend=False,
                ):
                    if cancel is not None and cancel.is_set():
                        continue
                    chunk = out["tts_speech"]
                    yield chunk.squeeze(0).float().cpu().numpy().astype(np.float32)
            finally:
                model.token_hop_len = self.cfg.hop

    def close(self) -> None:
        self.tts = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
