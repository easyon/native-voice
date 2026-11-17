"""TTS provider factory (selection happens at startup, one model per process)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config.schema import ProviderSpec, ServiceConfig
from .config import TTSModelConfig
from .cosyvoice3 import CosyVoice3

_ALIASES = {
    "cosyvoice": "cosyvoice3",
    "qwen3-tts": "qwen3tts",
    "qwen3_tts": "qwen3tts",
    "qwen3tts-vllm": "qwen3tts_vllm",
    "qwen3-tts-vllm": "qwen3tts_vllm",
    "qwen3tts_vllm": "qwen3tts_vllm",
    "cosyvoice3-vllm": "cosyvoice3_vllm",
    "cosyvoice3_vllm": "cosyvoice3_vllm",
}


def normalize_provider(provider: str | None, default: str) -> str:
    name = (provider or default or "").strip().lower()
    return _ALIASES.get(name, name)


def provider_spec(cfg: ServiceConfig, kind: str, provider: str) -> ProviderSpec | None:
    return (cfg.providers.get(kind) or {}).get(provider)


def _model(cfg: ServiceConfig, spec: ProviderSpec | None, default_dir: str, hub_id: str) -> str:
    if spec is not None and spec.model:
        value = spec.model
        if value.startswith(("models/", "models\\")):
            local = (Path(cfg.root) / value).resolve()
            return str(local) if local.exists() else hub_id
        return value
    local = (Path(cfg.root) / default_dir).resolve()
    return str(local) if local.exists() else hub_id


def _plain_prompt_text(text: str) -> str:
    """CosyVoice prompts carry a control prefix; clone models want the transcript."""
    value = (text or "").strip()
    if "<|endofprompt|>" in value:
        value = value.split("<|endofprompt|>", 1)[1].strip()
    return value


def create_tts_model(cfg: ServiceConfig, provider: str | None = None):
    name = normalize_provider(provider, cfg.engine.tts_provider)
    spec = provider_spec(cfg, "tts", name)
    opts = dict(spec.options) if spec is not None else {}

    if name == "cosyvoice3":
        model_dir = opts.get("model") or (
            spec.model if spec is not None and spec.model else cfg.models.tts_model
        )
        return CosyVoice3(
            TTSModelConfig(
                model_dir=Path(model_dir),
                prompt_wav=Path(opts.get("prompt_wav", cfg.models.tts_prompt_wav)),
                prompt_text=str(opts.get("prompt_text", cfg.models.tts_prompt_text)),
                voices=cfg.voices,
                hop=int(opts.get("hop", cfg.models.tts_hop)),
                max_cache_len=int(opts.get("max_cache_len", cfg.models.tts_max_cache_len)),
                use_cuda_graph=bool(opts.get("use_cuda_graph", cfg.models.use_cuda_graph)),
                warmup=bool(opts.get("warmup", cfg.models.warmup)),
                third_party_dir=(Path(cfg.root) / "third_party" / "CosyVoice-main").resolve(),
            )
        )

    if name == "qwen3tts":
        from .qwen3tts import Qwen3TTS, Qwen3TTSConfig

        model_dir = _model(
            cfg, spec, "models/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        )
        return Qwen3TTS(
            Qwen3TTSConfig(
                model_dir=model_dir,
                prompt_wav=Path(str(opts.get("prompt_wav", cfg.models.tts_prompt_wav))),
                prompt_text=_plain_prompt_text(str(opts.get("prompt_text", cfg.models.tts_prompt_text))),
                voices=cfg.voices,
                language=str(opts.get("language", "Chinese")),
                device=str(opts.get("device", "cuda:0")),
                dtype=str(opts.get("dtype", "bfloat16")),
                attn_implementation=str(opts.get("attn_implementation", "sdpa")),
                chunk_ms=int(opts.get("chunk_ms", 100)),
                max_new_tokens=int(opts.get("max_new_tokens", 4096)),
                non_streaming_mode=bool(opts.get("non_streaming_mode", False)),
                do_sample=bool(opts.get("do_sample", True)),
                warmup=bool(opts.get("warmup", cfg.models.warmup)),
            )
        )

    if name == "qwen3tts_vllm":
        from .qwen3tts_vllm import Qwen3TTSVLLM, Qwen3TTSVLLMConfig

        return Qwen3TTSVLLM(
            Qwen3TTSVLLMConfig(
                **_vllm_kwargs(
                    cfg,
                    opts,
                    url="ws://127.0.0.1:8091/v1/audio/speech/stream",
                    health_url="http://127.0.0.1:8091/health",
                ),
                language=str(opts.get("language", "Chinese")),
            )
        )

    if name == "cosyvoice3_vllm":
        from .cosyvoice3_vllm import CosyVoice3VLLM, CosyVoice3VLLMConfig

        return CosyVoice3VLLM(
            CosyVoice3VLLMConfig(
                **_vllm_kwargs(
                    cfg,
                    opts,
                    url="ws://127.0.0.1:8092/v1/audio/speech/stream",
                    health_url="http://127.0.0.1:8092/health",
                ),
                language=(str(opts["language"]) if opts.get("language") else None),
            )
        )

    known = ", ".join(
        sorted(
            {
                *cfg.providers.get("tts", {}),
                "cosyvoice3",
                "cosyvoice3_vllm",
                "qwen3tts",
                "qwen3tts_vllm",
            }
        )
    )
    raise ValueError(f"unknown TTS provider {name!r}; configured: {known}")


def _vllm_kwargs(
    cfg: ServiceConfig, opts: dict[str, Any], url: str, health_url: str
) -> dict[str, Any]:
    """Options shared by the two vLLM-Omni WebSocket proxies."""
    return dict(
        url=str(opts.get("url", url)),
        health_url=str(opts.get("health_url", health_url)),
        root=cfg.root,
        container_root=str(opts.get("container_root", "/workspace")),
        sample_rate=int(opts.get("sample_rate", 24000)),
        response_format=str(opts.get("response_format", "pcm")),
        split_granularity=(
            str(opts["split_granularity"]) if opts.get("split_granularity") else None
        ),
        max_new_tokens=(
            int(opts["max_new_tokens"]) if opts.get("max_new_tokens") is not None else None
        ),
        startup_timeout_s=float(opts.get("startup_timeout_s", 600)),
        idle_reconnect_s=float(opts.get("idle_reconnect_s", 20)),
        warmup=bool(opts.get("warmup", True)),
        warmup_text=str(opts.get("warmup_text", "你好。")),
        voices=cfg.voices,
        prompt_wav=Path(str(opts.get("prompt_wav", cfg.models.tts_prompt_wav))),
        prompt_text=_plain_prompt_text(str(opts.get("prompt_text", cfg.models.tts_prompt_text))),
        description=str(opts.get("description", "默认音色")),
    )
