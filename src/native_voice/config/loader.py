"""YAML + environment configuration loader."""
from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    ASRDefaults,
    AuthConfig,
    ConversationConfig,
    EngineConfig,
    LLMConfig,
    LoggingConfig,
    ModelsConfig,
    ProviderSpec,
    QuotaConfig,
    ServerConfig,
    ServiceConfig,
    TTSDefaults,
    VoiceProfile,
    VoiceStoreConfig,
)


def _fill(cls, data: dict[str, Any] | None):
    data = data or {}
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):  # pragma: no cover - not used yet
            value = _fill(f.type, value)
        kwargs[f.name] = value
    return cls(**{k: v for k, v in kwargs.items() if k in {f.name for f in fields(cls)}})


def _path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def _load_voices(path: Path, root: Path) -> dict[str, VoiceProfile]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, VoiceProfile] = {}
    for name, item in (raw.get("voices") or {}).items():
        item = item or {}
        out[name] = VoiceProfile(
            name=name,
            prompt_wav=_path(root, item.get("prompt_wav", "")),
            prompt_text=str(item.get("prompt_text", "")),
            description=str(item.get("description", "")),
        )
    return out


def _model_ref(root: Path, value: str) -> str:
    """Resolve a local model path, but leave hub ids such as ``Qwen/...`` alone."""
    value = str(value or "").strip()
    if not value:
        return ""
    p = Path(value)
    if p.is_absolute():
        return str(p)
    local = (root / p).resolve()
    return str(local) if local.exists() else value


def _load_providers(raw: dict[str, Any] | None, root: Path) -> dict[str, dict[str, ProviderSpec]]:
    out: dict[str, dict[str, ProviderSpec]] = {"asr": {}, "tts": {}}
    for kind in ("asr", "tts"):
        entries = ((raw or {}).get(kind) or {})
        if not isinstance(entries, dict):
            continue
        for name, item in entries.items():
            item = item or {}
            python = str(item.get("python", "") or "").strip()
            if python:
                python = str(_path(root, python))
            out[kind][str(name)] = ProviderSpec(
                name=str(name),
                python=python,
                module=str(item.get("module", "native_voice.engine.cli") or "native_voice.engine.cli"),
                model=_model_ref(root, item.get("model", "")),
                options=dict(item.get("options") or {}),
            )
    return out


def _apply_env(cfg: ServiceConfig) -> ServiceConfig:
    env = os.environ
    if env.get("NV_HOST"):
        cfg.server.host = env["NV_HOST"]
    if env.get("NV_PORT"):
        cfg.server.port = int(env["NV_PORT"])
    if env.get("NV_LOG_LEVEL"):
        cfg.logging.level = env["NV_LOG_LEVEL"]
    if env.get("NV_DB_PATH"):
        cfg.auth.db_path = _path(cfg.root, env["NV_DB_PATH"])
    if env.get("NV_DEV_KEY"):
        cfg.auth.dev_key = env["NV_DEV_KEY"].lower() in ("1", "true", "yes")
    if env.get("NV_TICKET_SECRET"):
        cfg.auth.ticket_secret = env["NV_TICKET_SECRET"]
    if env.get("NV_TTS_MODEL"):
        cfg.models.tts_model = _path(cfg.root, env["NV_TTS_MODEL"])
    if env.get("NV_TTS_PROMPT_WAV"):
        cfg.models.tts_prompt_wav = _path(cfg.root, env["NV_TTS_PROMPT_WAV"])
    if env.get("NV_ASR_DEVICE"):
        cfg.models.asr_device = env["NV_ASR_DEVICE"]
    if env.get("NV_ASR_PROVIDER"):
        cfg.engine.asr_provider = env["NV_ASR_PROVIDER"]
    if env.get("NV_TTS_PROVIDER"):
        cfg.engine.tts_provider = env["NV_TTS_PROVIDER"]
    if env.get("NV_LLM_GPU_LAYERS"):
        cfg.llm.n_gpu_layers = int(env["NV_LLM_GPU_LAYERS"])
    if env.get("NV_LLM_SERVER"):
        server = _path(cfg.root, env["NV_LLM_SERVER"])
        if server is not None:
            cfg.llm.server = server
    return cfg


def load_config(path: str | Path | None = None) -> ServiceConfig:
    cfg_path = Path(path or os.environ.get("NV_CONFIG", "configs/service.yaml")).resolve()
    root = cfg_path.parent.parent
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    cfg = ServiceConfig(
        root=root,
        server=_fill(ServerConfig, raw.get("server")),
        auth=_fill(AuthConfig, raw.get("auth")),
        quotas=_fill(QuotaConfig, raw.get("quotas")),
        models=_fill(ModelsConfig, raw.get("models")),
        tts=_fill(TTSDefaults, raw.get("tts")),
        asr=_fill(ASRDefaults, raw.get("asr")),
        engine=_fill(EngineConfig, raw.get("engine")),
        llm=_fill(LLMConfig, raw.get("llm")),
        conversation=_fill(ConversationConfig, raw.get("conversation")),
        voice_store=_fill(VoiceStoreConfig, raw.get("voice_store")),
        logging=_fill(LoggingConfig, raw.get("logging")),
        providers=_load_providers(raw.get("providers"), root),
    )
    cfg.auth.db_path = _path(root, cfg.auth.db_path)
    cfg.voice_store.dir = _path(root, cfg.voice_store.dir)
    cfg.models.root = root.resolve()
    cfg.llm.server = _path(root, cfg.llm.server)
    cfg.llm.model = _path(root, cfg.llm.model)
    cfg.conversation.speech_gate_model = _path(root, cfg.conversation.speech_gate_model)
    cfg.models.tts_model = _path(root, cfg.models.tts_model)
    cfg.models.tts_prompt_wav = _path(root, cfg.models.tts_prompt_wav)
    cfg.voices = _load_voices(cfg_path.parent / "voices.yaml", root)
    return _apply_env(cfg)
