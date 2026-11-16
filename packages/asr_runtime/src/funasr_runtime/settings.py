from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ServerConfig(_Base):
    host: str = "0.0.0.0"
    port: int = 8001

    request_timeout_s: float = 120.0
    ws_idle_timeout_s: float = 23.0
    ws_send_timeout_s: float = 10.0
    max_request_bytes: int = 1_048_576
    max_ws_message_bytes: int = 1_048_576
    max_ws_connections: int = 1_000
    # ffmpeg / file I/O pool. Model calls use their own executor.
    worker_threads: int = 16

    cors_origins: list[str] = Field(default_factory=list)
    trusted_hosts: list[str] = Field(default_factory=list)
    docs_enabled: bool = False
    # Self-contained browser test console at /ui (microphone, WAV replay and
    # offline transcription). It has no server-side state and no build step.
    ui_enabled: bool = True
    access_log: bool = True
    proxy_headers: bool = True
    forwarded_allow_ips: str = "127.0.0.1"
    keepalive_timeout_s: float = 5.0
    ssl_certfile: str = ""
    ssl_keyfile: str = ""


class TenantKeyConfig(_Base):
    api_key: str
    id: str = ""
    max_concurrent_streams: int | None = None
    requests_per_minute: int | None = None
    audio_seconds_per_minute: int | None = None


class AuthConfig(_Base):
    enabled: bool = False
    api_keys: list[str] = Field(default_factory=list)
    tenants: list[TenantKeyConfig] = Field(default_factory=list)
    metrics_public: bool = False
    metrics_token: str = ""


class MetricsConfig(_Base):
    gpu_sampling_interval_s: float = Field(default=5.0, ge=0.0)
    gpu_device: int = Field(default=0, ge=0)


class VadConfig(_Base):
    """Endpointing strategy for the realtime path.

    ``funasr`` uses the fsmn-vad streaming model and is the production default.
    ``energy`` is a dependency-free RMS endpointer used by the GGUF engine.
    ``passthrough`` treats the whole task as one utterance (mock/tests).
    """

    type: Literal["passthrough", "energy", "funasr"] = "passthrough"
    energy_threshold_db: float = -40.0
    frame_ms: int = 20
    # Session-level pre-roll so the ASR sees the first phoneme. The session
    # keeps this buffer; it is not part of the VAD model.
    pre_roll_ms: int = 200


class MockEngineConfig(_Base):
    sentences: list[str] = Field(
        default_factory=lambda: ["你好，世界。", "今天天气不错。"]
    )
    partial_interval_ms: int = 200
    # One artificial sentence every N ms so protocol tests do not depend on
    # energy thresholds.
    segment_ms: int = 1_800
    latency_ms: int = 0
    emit_words: bool = True


class FunasrEngineConfig(_Base):
    device: str = "auto"
    vad_device: str = "cpu"
    model_root: str = "./models"
    # Private deployments must opt in to network downloads explicitly.
    allow_download: bool = False
    hub: Literal["ms", "hf"] = "ms"
    executor_threads: int = Field(default=1, ge=1, le=64)
    trust_remote_code: bool = True
    remote_code: str = ""
    # Empty means "use models.<key>.vad_artifact", then the built-in
    # ``fsmn-vad`` model id when downloads are enabled.
    vad_model: str = ""
    partial_interval_ms: int = Field(default=960, ge=0)
    partial_window_s: float = Field(default=8.0, ge=0.0)
    warmup: bool = True
    # Passed to AutoModel.generate after signature filtering.
    inference_args: dict[str, Any] = Field(default_factory=dict)


class LlamaCppEngineConfig(_Base):
    binary_dir: str = "./bin"
    binaries: dict[str, str] = Field(default_factory=dict)
    # {binary}, {model}, {encoder}, {vad}, {audio} are substituted. Argument
    # lists are used directly (never shell=True).
    args: list[str] = Field(
        default_factory=lambda: ["{binary}", "-m", "{model}", "-a", "{audio}"]
    )
    timeout_s: float = 120.0
    max_processes: int = Field(default=1, ge=1)
    partial_interval_ms: int = Field(default=0, ge=0)


class EngineConfig(_Base):
    type: str = "mock"
    vad: VadConfig = Field(default_factory=VadConfig)
    mock: MockEngineConfig = Field(default_factory=MockEngineConfig)
    funasr: FunasrEngineConfig = Field(default_factory=FunasrEngineConfig)
    llamacpp: LlamaCppEngineConfig = Field(default_factory=LlamaCppEngineConfig)


class ModelConfig(_Base):
    # Empty inherits engine.type.
    engine: str = ""
    architecture: str = "fun-asr-nano"
    artifact: str = ""
    encoder_artifact: str = ""
    vad_artifact: str = ""
    language: str = "auto"
    hotwords: str = ""
    itn: bool = True
    inference_args: dict[str, Any] = Field(default_factory=dict)
    # Extra llama.cpp arguments appended after the template.
    args: list[str] = Field(default_factory=list)


class OfflineConfig(_Base):
    default_model: str = ""
    max_upload_bytes: int = 100 * 1024 * 1024
    max_duration_s: float = 3_600.0


class LimitsConfig(_Base):
    max_concurrent_streams: int = 32
    acquire_timeout_s: float = 5.0
    max_task_duration_s: float = 3_600.0
    max_audio_bytes_per_task: int = 128 * 1024 * 1024
    heartbeat_interval_ms: int = 5_000
    finish_timeout_s: float = 15.0
    max_segment_ms: int = Field(default=30_000, ge=1_000)
    min_speech_ms: int = Field(default=200, ge=20)
    shutdown_timeout_s: float = 10.0


class LoggingConfig(_Base):
    level: str = "INFO"
    format: Literal["text", "json"] = "text"
    access_log: bool = True


class RuntimeConfig(_Base):
    service_name: str = "funasr-runtime"
    version: str = "0.1.0"
    environment: str = "dev"

    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)
    offline: OfflineConfig = Field(default_factory=OfflineConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _check_config(self) -> RuntimeConfig:
        if not self.models:
            raise ValueError("at least one model must be configured")

        for name, model in self.models.items():
            if not model.engine:
                model.engine = self.engine.type
            if model.engine not in {"mock", "funasr", "llamacpp"}:
                raise ValueError(
                    f"models.{name}.engine={model.engine!r} is not supported; "
                    "use mock, funasr or llamacpp"
                )
            if model.engine == "funasr" and not model.artifact:
                raise ValueError(f"models.{name}.artifact is required")
            if model.engine == "llamacpp" and not model.artifact:
                raise ValueError(f"models.{name}.artifact is required")
            if model.engine == "llamacpp" and not model.architecture:
                raise ValueError(f"models.{name}.architecture is required")

        for alias, target in self.aliases.items():
            if target not in self.models:
                raise ValueError(
                    f"aliases.{alias}={target!r} does not name a configured model"
                )

        if not self.offline.default_model:
            self.offline.default_model = next(iter(self.models))
        elif self.offline.default_model not in self.models:
            raise ValueError(
                f"offline.default_model={self.offline.default_model!r} "
                "does not name a configured model"
            )

        if self.auth.enabled and not self.auth.api_keys and not self.auth.tenants:
            raise ValueError("auth.enabled=true requires api_keys or tenants")
        if self.limits.heartbeat_interval_ms < 100:
            raise ValueError("limits.heartbeat_interval_ms must be >= 100")
        return self

    @classmethod
    def default(cls) -> RuntimeConfig:
        return cls(
            engine=EngineConfig(type="mock"),
            models={
                "fun-asr-realtime": ModelConfig(
                    engine="mock",
                    architecture="mock",
                    artifact="",
                )
            },
            aliases={"whisper-1": "fun-asr-realtime"},
            offline=OfflineConfig(default_model="fun-asr-realtime"),
        )


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    engine_type = os.getenv("FUNASR_ENGINE_TYPE")
    if engine_type:
        raw.setdefault("engine", {})["type"] = engine_type

    device = os.getenv("FUNASR_DEVICE")
    if device:
        raw.setdefault("engine", {}).setdefault("funasr", {})["device"] = device

    models_dir = os.getenv("FUNASR_MODELS_DIR")
    if models_dir:
        raw.setdefault("engine", {}).setdefault("funasr", {})[
            "model_root"
        ] = models_dir

    api_keys = os.getenv("FUNASR_API_KEYS")
    if api_keys is not None:
        auth = raw.setdefault("auth", {})
        auth["enabled"] = bool(api_keys.strip())
        auth["api_keys"] = [k.strip() for k in api_keys.split(",") if k.strip()]

    metrics_token = os.getenv("FUNASR_METRICS_TOKEN")
    if metrics_token is not None:
        raw.setdefault("auth", {})["metrics_token"] = metrics_token

    upload = _env_int("FUNASR_OFFLINE_MAX_UPLOAD_BYTES")
    if upload is not None:
        raw.setdefault("offline", {})["max_upload_bytes"] = upload

    port = _env_int("FUNASR_PORT")
    if port is not None:
        raw.setdefault("server", {})["port"] = port

    max_streams = _env_int("FUNASR_MAX_CONCURRENT_STREAMS")
    if max_streams is not None:
        raw.setdefault("limits", {})["max_concurrent_streams"] = max_streams

    partial_interval = _env_int("FUNASR_PARTIAL_INTERVAL_MS")
    if partial_interval is not None:
        raw.setdefault("engine", {}).setdefault("funasr", {})[
            "partial_interval_ms"
        ] = partial_interval

    partial_window = _env_float("FUNASR_PARTIAL_WINDOW_S")
    if partial_window is not None:
        raw.setdefault("engine", {}).setdefault("funasr", {})[
            "partial_window_s"
        ] = partial_window

    log_level = os.getenv("FUNASR_LOG_LEVEL")
    if log_level:
        raw.setdefault("logging", {})["level"] = log_level

    log_format = os.getenv("FUNASR_LOG_FORMAT")
    if log_format:
        raw.setdefault("logging", {})["format"] = log_format

    environment = os.getenv("FUNASR_ENVIRONMENT")
    if environment:
        raw["environment"] = environment
    return raw


def load_config(path: str | Path | None = None) -> RuntimeConfig:
    config_path = Path(
        path or os.getenv("FUNASR_RUNTIME_CONFIG") or "configs/funasr.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"runtime config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"runtime config must be a mapping: {config_path}")
    return RuntimeConfig.model_validate(_apply_env_overrides(raw))
