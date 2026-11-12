from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ServerConfig(_Base):
    host: str = "0.0.0.0"
    port: int = 8000

    # End-to-end limits. The per-sentence limit protects the engine; the
    # request limit protects the gateway from a single slow client.
    request_timeout_s: float = 120.0
    ws_idle_timeout_s: float = 23.0
    ws_send_timeout_s: float = 10.0
    max_request_bytes: int = 1_048_576
    max_ws_message_bytes: int = 1_048_576
    max_ws_connections: int = 1_000

    # Thread pool for ffmpeg/resampling and file I/O. Each streaming chunk of
    # mp3/opus/resampling uses one thread; size this for the per-pod stream
    # limit, not for the model worker.
    worker_threads: int = 32

    cors_origins: list[str] = Field(default_factory=list)
    trusted_hosts: list[str] = Field(default_factory=list)
    docs_enabled: bool = False

    access_log: bool = True
    proxy_headers: bool = True
    forwarded_allow_ips: str = "127.0.0.1"
    keepalive_timeout_s: float = 5.0

    # TLS is normally terminated by the ingress; these are for single-node
    # deployments that terminate TLS in the process.
    ssl_certfile: str = ""
    ssl_keyfile: str = ""


class TenantKeyConfig(_Base):
    """Optional per-key limits.

    A tenant key may be configured here instead of in ``api_keys``. When both
    are present, the entry here wins.
    """

    api_key: str
    id: str = ""
    max_concurrent_streams: int | None = None
    requests_per_minute: int | None = None
    characters_per_minute: int | None = None


class AuthConfig(_Base):
    enabled: bool = False
    api_keys: list[str] = Field(default_factory=list)
    tenants: list[TenantKeyConfig] = Field(default_factory=list)

    # Prometheus scrapers usually cannot use the business API key. Set a
    # dedicated token, or set metrics_public=true only on an isolated network.
    metrics_public: bool = False
    metrics_token: str = ""


class MockEngineConfig(_Base):
    chunk_ms: int = 200
    realtime: bool = False
    amplitude: float = 0.12


class CosyVoiceEngineConfig(_Base):
    load_jit: bool = False
    load_trt: bool = False
    fp16: bool = False
    preload: bool = False
    max_concurrent: int = 8
    sample_rate: int = 24_000

    # Optional accelerator stack (requirements-accel.lock).
    #
    # Upstream CosyVoice3 runs the LLM on one shared CUDA stream, so threads
    # only fill CPU gaps: measured 1.17x/1.56x/1.47x aggregate at 1/2/4
    # concurrent requests on one A100. vLLM exports the LLM once
    # (<artifact>/vllm) and steps all active requests together (continuous
    # batching), which is the vendor-supported fix for that bottleneck.
    load_vllm: bool = False
    # TensorRT execution contexts for the flow (DiT) decoder estimator. >1
    # lets concurrent token2wav calls use separate contexts instead of
    # serializing on one. Only used with load_trt.
    trt_concurrent: int = Field(default=1, ge=1, le=32)

    # TF32 keeps fp32 weights while running the LLM and DiT matmuls on tensor
    # cores. On A100/H100 this is typically a large throughput win for TTS and
    # has no audible quality effect at these model sizes. Set false when a
    # strict fp32 numerics check is required.
    tf32: bool = True

    # Zero-shot voices re-extract the prompt wav (speech tokens, features and
    # speaker embedding) on every request. Registering the voice once at model
    # load with ``add_zero_shot_spk`` removes that work from the hot path and
    # is the single biggest first-audio win for call scenarios.
    cache_prompt_features: bool = True

    # The worker -> event-loop queue is bounded. A slow client must not let
    # GPU output accumulate without limit.
    queue_size: int = 64
    cancel_grace_s: float = 2.0
    warmup_text: str = "你好，欢迎使用语音合成服务。"


class EngineConfig(_Base):
    # String, not Literal: deployments can register additional engine adapters
    # (remote Triton/gRPC model services) through engine_factory.register_engine.
    type: str = "mock"
    mock: MockEngineConfig = Field(default_factory=MockEngineConfig)
    cosyvoice: CosyVoiceEngineConfig = Field(default_factory=CosyVoiceEngineConfig)


class ModelConfig(_Base):
    engine: str = "cosyvoice3"
    artifact: str = ""
    default_voice: str | None = None
    sample_rate: int = 24_000


class VoiceConfig(_Base):
    spk_id: str = ""
    mode: Literal["sft", "instruct2", "zero_shot", "auto"] = "sft"
    instruct: str = ""
    prompt_wav: str = ""
    prompt_text: str = ""
    embedding: str = ""


class LimitsConfig(_Base):
    max_text_chars: int = 20_000
    max_total_chars: int = 200_000
    max_concurrent_streams: int = 64
    acquire_timeout_s: float = 5.0
    sentence_timeout_s: float = 60.0
    first_audio_timeout_s: float = 10.0
    max_request_s: float = 120.0
    max_task_duration_s: float = 600.0
    shutdown_timeout_s: float = 10.0
    ws_queue_size: int = 32
    # Upper bound on how long finish-task(cancel) waits for the synthesis task
    # to stop. The task is also stopped cooperatively, so this only covers a
    # send that is already in flight to a slow client.
    ws_cancel_timeout_s: float = 5.0
    # Guard the gateway against unbounded memory when SSE also stores the full
    # utterance for a download URL. Streaming itself is unaffected.
    max_stream_audio_bytes: int = 64 * 1024 * 1024
    ssml_strict: bool = True


class StorageConfig(_Base):
    # String so additional stores (S3/MinIO) can be registered without a schema
    # change. The built-in factory supports local and none.
    type: str = "local"
    dir: str = "./data/audio"
    base_url: str = "http://localhost:8000"
    ttl_s: int = 86_400
    serve_files: bool = True
    signing_secret: str = ""
    cleanup_interval_s: float = 3_600.0


class LoggingConfig(_Base):
    level: str = "INFO"
    format: Literal["text", "json"] = "text"
    access_log: bool = True


class RuntimeConfig(_Base):
    service_name: str = "cosyvoice-runtime"
    version: str = "0.1.0"
    environment: str = "dev"

    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    voices: dict[str, VoiceConfig] = Field(default_factory=dict)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _check_config(self) -> RuntimeConfig:
        if self.engine.type == "cosyvoice":
            if not self.models:
                raise ValueError("engine.type=cosyvoice requires at least one model")
            for model_name, model in self.models.items():
                if not model.artifact:
                    raise ValueError(
                        f"models.{model_name}.artifact is required for the "
                        "cosyvoice engine"
                    )
        for model_name, model in self.models.items():
            if model.default_voice and model.default_voice not in self.voices:
                raise ValueError(
                    f"models.{model_name}.default_voice={model.default_voice!r} "
                    "is not defined in voices"
                )
        if self.auth.enabled and not self.auth.api_keys and not self.auth.tenants:
            raise ValueError("auth.enabled=true requires api_keys or tenants")
        if self.storage.type == "local" and self.storage.serve_files:
            if self.storage.ttl_s <= 0:
                raise ValueError(
                    "storage.ttl_s must be > 0 when local files are served; "
                    "otherwise signed URLs expire immediately"
                )
            if not self.storage.signing_secret and self.environment != "dev":
                raise ValueError(
                    "storage.signing_secret is required when local files are served "
                    "outside the dev environment"
                )
        return self

    @classmethod
    def default(cls) -> RuntimeConfig:
        return cls(
            engine=EngineConfig(type="mock"),
            models={
                "my-tts-prod": ModelConfig(
                    engine="mock",
                    artifact="",
                    default_voice="voice-a",
                )
            },
            voices={
                "voice-a": VoiceConfig(
                    spk_id="voice-a",
                    mode="sft",
                    instruct="你是一个温柔、自然的助手。",
                )
            },
            storage=StorageConfig(type="none"),
        )


def _apply_env_overrides(raw: dict) -> dict:
    """Small set of env overrides for container deployments.

    Secrets should come from env/secret mounts, not from the YAML file.
    """
    engine_type = os.getenv("COSYVOICE_ENGINE_TYPE")
    if engine_type:
        raw.setdefault("engine", {})["type"] = engine_type

    api_keys = os.getenv("COSYVOICE_API_KEYS")
    if api_keys is not None:
        auth = raw.setdefault("auth", {})
        auth["enabled"] = bool(api_keys.strip())
        auth["api_keys"] = [k.strip() for k in api_keys.split(",") if k.strip()]

    metrics_token = os.getenv("COSYVOICE_METRICS_TOKEN")
    if metrics_token is not None:
        raw.setdefault("auth", {})["metrics_token"] = metrics_token

    storage_dir = os.getenv("COSYVOICE_STORAGE_DIR")
    if storage_dir:
        raw.setdefault("storage", {})["dir"] = storage_dir

    base_url = os.getenv("COSYVOICE_STORAGE_BASE_URL")
    if base_url:
        raw.setdefault("storage", {})["base_url"] = base_url

    signing_secret = os.getenv("COSYVOICE_STORAGE_SIGNING_SECRET")
    if signing_secret is not None:
        raw.setdefault("storage", {})["signing_secret"] = signing_secret

    log_level = os.getenv("COSYVOICE_LOG_LEVEL")
    if log_level:
        raw.setdefault("logging", {})["level"] = log_level

    log_format = os.getenv("COSYVOICE_LOG_FORMAT")
    if log_format:
        raw.setdefault("logging", {})["format"] = log_format

    environment = os.getenv("COSYVOICE_ENVIRONMENT")
    if environment:
        raw["environment"] = environment
    return raw


def load_config(path: str | Path | None = None) -> RuntimeConfig:
    config_path = Path(
        path
        or os.getenv("COSYVOICE_RUNTIME_CONFIG")
        or "configs/runtime.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"runtime config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"runtime config must be a mapping: {config_path}")
    return RuntimeConfig.model_validate(_apply_env_overrides(raw))
