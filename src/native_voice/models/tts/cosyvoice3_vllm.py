"""CosyVoice3 served by an external vLLM-Omni process.

CosyVoice3 exposes the same OpenAI-compatible speech API and WebSocket stream
as Qwen3-TTS, so the proxy in :mod:`qwen3tts_vllm` is reused. Only the session
fields differ: CosyVoice3 clones from ``ref_audio`` + ``ref_text`` and rejects
Qwen's ``task_type`` / ``x_vector_only_mode``; ``language`` is Qwen-only as
well.
"""
from __future__ import annotations

from dataclasses import dataclass

from .qwen3tts_vllm import Qwen3TTSVLLM, Qwen3TTSVLLMConfig


@dataclass
class CosyVoice3VLLMConfig(Qwen3TTSVLLMConfig):
    url: str = "ws://127.0.0.1:8092/v1/audio/speech/stream"
    health_url: str = "http://127.0.0.1:8092/health"
    task_type: str | None = None
    language: str | None = None
    x_vector_only_mode: bool = False


class CosyVoice3VLLM(Qwen3TTSVLLM):
    LABEL = "CosyVoice3 vLLM"
