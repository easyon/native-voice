"""ASR provider factory.

The provider is chosen at startup (``engine.asr_provider`` / ``NV_ASR_PROVIDER``)
and exactly one implementation is loaded into the ASR engine process. Heavy
third-party imports stay inside each branch so the gateway's CPU-only venv never
needs Qwen/FunASR dependencies.
"""
from __future__ import annotations

from pathlib import Path

from ...config.schema import ProviderSpec, ServiceConfig
from .config import ASRModelConfig
from .streaming_paraformer import StreamingParaformer

_ALIASES = {
    "funasr": "paraformer",
    "paraformer-streaming": "paraformer",
    "qwen3-asr": "qwen3asr",
    "qwen3_asr": "qwen3asr",
}


def normalize_provider(provider: str | None, default: str) -> str:
    name = (provider or default or "").strip().lower()
    return _ALIASES.get(name, name)


def provider_spec(cfg: ServiceConfig, kind: str, provider: str) -> ProviderSpec | None:
    return (cfg.providers.get(kind) or {}).get(provider)


def _model(cfg: ServiceConfig, spec: ProviderSpec | None, default_dir: str, hub_id: str) -> str:
    if spec is not None and spec.model:
        value = spec.model
        # A configured local directory that has not been downloaded yet: fall
        # back to the hub id so from_pretrained can fetch it.
        if value.startswith(("models/", "models\\")):
            local = (Path(cfg.root) / value).resolve()
            return str(local) if local.exists() else hub_id
        return value
    local = (Path(cfg.root) / default_dir).resolve()
    return str(local) if local.exists() else hub_id


def create_asr_engine(cfg: ServiceConfig, provider: str | None = None):
    name = normalize_provider(provider, cfg.engine.asr_provider)
    spec = provider_spec(cfg, "asr", name)
    opts = dict(spec.options) if spec is not None else {}

    if name == "paraformer":
        model = opts.get("model") or (
            spec.model if spec is not None and spec.model else cfg.models.asr_stream_model
        )
        return StreamingParaformer(
            ASRModelConfig(
                model=str(model),
                device=str(opts.get("device", cfg.models.asr_device)),
                chunk_ms=int(opts.get("chunk_ms", cfg.models.asr_chunk_ms)),
            )
        )

    if name == "qwen3asr":
        from .qwen3_asr import Qwen3ASR, Qwen3ASRConfig

        return Qwen3ASR(
            Qwen3ASRConfig(
                model=_model(cfg, spec, "models/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-0.6B"),
                device=str(opts.get("device", cfg.models.asr_device)),
                dtype=str(opts.get("dtype", "bfloat16")),
                language=opts.get("language"),
                partial_interval_ms=int(opts.get("partial_interval_ms", 800)),
                max_partial_seconds=float(opts.get("max_partial_seconds", 8.0)),
                max_new_tokens=int(opts.get("max_new_tokens", 512)),
                warmup=bool(opts.get("warmup", True)),
            )
        )

    known = ", ".join(sorted({*cfg.providers.get("asr", {}), "paraformer", "qwen3asr"}))
    raise ValueError(f"unknown ASR provider {name!r}; configured: {known}")
