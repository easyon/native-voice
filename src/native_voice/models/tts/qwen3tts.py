"""Qwen3-TTS-12Hz Base runtime (voice clone, one instance per process).

The open-source ``qwen-tts`` package exposes whole-utterance generation, not a
token-level audio stream. The gateway already splits LLM output into short
clauses, so this engine synthesizes one clause per job and yields the resulting
24 kHz waveform in small chunks. That keeps first-audio latency close to the
first-clause generation time without depending on vLLM-Omni.

This module is imported only in a venv that has ``qwen-tts`` installed; the
package itself is imported lazily inside :meth:`load`.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...audio.resample import resample
from ...config.schema import VoiceProfile

log = logging.getLogger(__name__)


@dataclass
class Qwen3TTSConfig:
    model_dir: str | Path
    prompt_wav: Path
    prompt_text: str
    voices: dict[str, VoiceProfile] = field(default_factory=dict)
    language: str = "Chinese"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    chunk_ms: int = 100
    sample_rate: int = 24000
    max_new_tokens: int = 4096
    non_streaming_mode: bool = False
    do_sample: bool = True
    warmup: bool = True


def _torch_dtype(torch, name: str):
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(name).lower(), torch.bfloat16)


def _plain(text: str) -> str:
    value = (text or "").strip()
    if "<|endofprompt|>" in value:
        value = value.split("<|endofprompt|>", 1)[1].strip()
    return value


class Qwen3TTS:
    def __init__(self, cfg: Qwen3TTSConfig):
        self.cfg = cfg
        self.model = None
        self.lock = threading.Lock()
        self.sample_rate = int(cfg.sample_rate)
        self.prompts: dict[str, Any] = {}
        self._voice_names: list[str] = []
        self._voice_desc: dict[str, str] = {}
        self._warm_voice: str | None = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        import torch
        try:
            from qwen_tts import Qwen3TTSModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "qwen3tts 需要独立 venv（envs/qwen3tts）。请用 worker 模式启动："
                "BACKEND=worker scripts/serve_dev.sh（或 "
                "scripts\\serve_dev.ps1 -Backend worker -Tts qwen3tts），"
                "不要在 local 模式下加载。"
            ) from exc

        kwargs: dict[str, Any] = {
            "device_map": self.cfg.device,
            "dtype": _torch_dtype(torch, self.cfg.dtype),
        }
        if self.cfg.attn_implementation:
            kwargs["attn_implementation"] = self.cfg.attn_implementation

        log.info("loading Qwen3-TTS from %s (%s, %s)", self.cfg.model_dir, self.cfg.device, self.cfg.dtype)
        self.model = Qwen3TTSModel.from_pretrained(str(self.cfg.model_dir), **kwargs)

        voices = self.cfg.voices or {}
        if voices:
            for name, profile in voices.items():
                self._add_voice(name, profile.prompt_text, Path(profile.prompt_wav), profile.description)
        elif self.cfg.prompt_wav and Path(self.cfg.prompt_wav).exists():
            self._add_voice("default", self.cfg.prompt_text, Path(self.cfg.prompt_wav), "默认音色")

        if not self._voice_names:
            raise RuntimeError(
                "Qwen3-TTS Base needs at least one reference voice; "
                "configure voices.yaml or models.tts_prompt_wav"
            )
        self._warm_voice = self._voice_names[0]

        if self.cfg.warmup:
            self._warmup()
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            log.info("Qwen3-TTS ready (VRAM %.2f/%.2f GB)", alloc, peak)
        self._ready = True

    def _warmup(self) -> None:
        log.info("warming up Qwen3-TTS")
        try:
            for _ in self.iter_synthesize("你好。", self._warm_voice, None):
                pass
        except Exception as exc:  # warmup must not block startup
            log.warning("Qwen3-TTS warmup failed: %r", exc)

    # ----------------------------------------------------------------- voices
    def _add_voice(self, name: str, prompt_text: str, prompt_wav: Path, description: str = "") -> None:
        if self.model is None:
            raise RuntimeError("model is not loaded")
        text = _plain(prompt_text)
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=str(prompt_wav),
            ref_text=text or None,
            x_vector_only_mode=not bool(text),
        )
        self.prompts[name] = prompt
        if name not in self._voice_names:
            self._voice_names.append(name)
        self._voice_desc[name] = description or name

    def add_voice(self, name: str, prompt_text: str, prompt_wav: Path, description: str = "") -> None:
        with self.lock:
            self._add_voice(name, prompt_text, prompt_wav, description)
        log.info("Qwen3-TTS voice added: %s", name)

    def remove_voice(self, name: str) -> None:
        with self.lock:
            self.prompts.pop(name, None)
            if name in self._voice_names:
                self._voice_names.remove(name)
            self._voice_desc.pop(name, None)
        log.info("Qwen3-TTS voice removed: %s", name)

    def voices(self) -> list[dict]:
        return [
            {"name": name, "description": self._voice_desc.get(name, name)}
            for name in self._voice_names
        ]

    def has_voice(self, name: str) -> bool:
        return name in self._voice_names

    def _max_tokens(self, text: str) -> int:
        """Bound generation so a bad clone prompt cannot produce minutes of audio.

        12.5 codec frames per second; 0.35 s/char plus 1.5 s is a generous
        upper bound for Chinese/English speech at normal pace. If the model
        fails to emit EOS, the cap stops it instead of running to 4096 frames.
        """
        seconds = max(2.0, len(text) * 0.35 + 1.5)
        return max(32, min(self.cfg.max_new_tokens, int(seconds * 12.5) + 16))

    # ------------------------------------------------------------- synthesize
    def iter_synthesize(self, text: str, voice: str, cancel):
        """Yield 24 kHz float32 mono chunks for one clause."""
        text = (text or "").strip()
        if not text:
            return
        if not self.has_voice(voice):
            raise ValueError(f"unknown voice: {voice}")

        with self.lock:
            prompt = self.prompts.get(voice) or self.prompts.get(self._warm_voice or "")
            if prompt is None:
                raise ValueError(f"unknown voice: {voice}")
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=self.cfg.language,
                voice_clone_prompt=prompt,
                non_streaming_mode=self.cfg.non_streaming_mode,
                do_sample=self.cfg.do_sample,
                max_new_tokens=self._max_tokens(text),
            )
            if not wavs:
                return
            wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            wav = resample(wav, int(sr), self.cfg.sample_rate)
            chunk = max(1, int(self.cfg.sample_rate * self.cfg.chunk_ms / 1000))
            for start in range(0, wav.size, chunk):
                if cancel is not None and cancel.is_set():
                    return
                yield wav[start : start + chunk]

    def close(self) -> None:
        with self.lock:
            self.model = None
            self.prompts.clear()
            self._voice_names.clear()
            self._voice_desc.clear()
            self._ready = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
