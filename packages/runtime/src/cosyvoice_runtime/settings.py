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

    # Self-contained browser test console at /ui. It is a static page with no
    # server-side state and no build step, so it is safe to leave enabled on a
    # GPU acceptance host; set false if the runtime should expose only the API.
    # The page calls the same public API the business uses (the API key, when
    # auth is enabled, is typed into the page and never stored server-side).
    ui_enabled: bool = True

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


class MetricsConfig(_Base):
    # The runtime exports request/stream/RTF metrics itself. GPU utilization
    # is the only signal it cannot observe from Python without a driver
    # binding, so a low-rate nvidia-smi sampler publishes it for dashboards and
    # replica autoscaling. 0 disables sampling (and the subprocess entirely).
    gpu_sampling_interval_s: float = Field(default=5.0, ge=0.0)
    gpu_device: int = Field(default=0, ge=0)


class MockEngineConfig(_Base):
    chunk_ms: int = 200
    realtime: bool = False
    amplitude: float = 0.12


class VllmEngineConfig(_Base):
    """Engine sizing for the vLLM LLM path.

    ``cosyvoice.cli.model.CosyVoiceModel.load_vllm`` hardcodes
    ``gpu_memory_utilization=0.2`` and leaves the model's full 32768-token
    context in place. With the flow / HiFi-GAN / tokenizer models already
    resident, 0.2 of a 32GB card is not enough for weights plus the
    torch.compile activation peak: the engine aborts during KV-cache init with
    "available KV cache memory (0.1 GiB)". These knobs let a deployment size
    the LLM to its card instead. They are applied by
    :func:`cosyvoice_runtime.engines.cosyvoice_compat.install_vllm_engine_overrides`.

    ``gpu_memory_utilization`` is a fraction of *total* GPU memory (vLLM
    semantics), not of the memory that is free after the TTS modules load.
    Leave at least ~25% of the card free when the flow decoder and HiFi-GAN
    run in the same process.
    """

    gpu_memory_utilization: float = Field(default=0.5, gt=0.0, le=1.0)
    # Upstream segments text into <=80-token chunks before the LLM, so a
    # request rarely exceeds a few thousand tokens (prompt + generated audio).
    max_model_len: int = Field(default=8192, ge=256)
    max_num_seqs: int = Field(default=32, ge=1, le=1024)
    max_num_batched_tokens: int = Field(default=8192, ge=256)

    def overrides(self) -> dict[str, Any]:
        """EngineArgs fields to force, in vLLM's own spelling."""
        return {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
        }


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
    # Sizing for the vLLM engine. See VllmEngineConfig: upstream hardcodes
    # gpu_memory_utilization=0.2, which cannot start on a 32GB card once the
    # TTS modules own several GB.
    vllm: VllmEngineConfig = Field(default_factory=VllmEngineConfig)
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


class CloningConfig(_Base):
    """Zero-shot voice cloning from a reference recording.

    Each cloned voice is one directory under ``dir`` containing a normalized
    ``prompt.wav`` and ``meta.json``. The runtime loads them at startup and
    registers them as regular zero-shot voices, so no model reload is needed
    when a new voice is created.
    """

    enabled: bool = True
    dir: str = "./data/voices"
    # The upload is spooled by Starlette and then normalized; cap both the
    # request body (middleware) and the bytes we read into memory.
    max_upload_bytes: int = 16 * 1024 * 1024
    min_duration_s: float = 1.0
    max_duration_s: float = 30.0
    # Target rate for ffmpeg normalization. 24k keeps enough detail for the
    # speaker encoder, which resamples to 16k internally.
    prompt_sample_rate: int = 24_000
    # Transcript length is bounded so one upload cannot bloat the metadata file.
    max_transcript_chars: int = 2_000


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
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    cloning: CloningConfig = Field(default_factory=CloningConfig)
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


def _apply_env_overrides(raw: dict) -> dict:
    """Small set of env overrides for container deployments.

    Secrets should come from env/secret mounts, not from the YAML file.
    The engine settings also let a container image or Helm chart override
    COSYVOICE_MAX_CONCURRENT / COSYVOICE_TRT_CONCURRENT per environment
    without generating a YAML variant per deployment.
    """
    engine_type = os.getenv("COSYVOICE_ENGINE_TYPE")
    if engine_type:
        raw.setdefault("engine", {})["type"] = engine_type

    cosyvoice = raw.setdefault("engine", {}).setdefault("cosyvoice", {})
    for env_name, key in (
        ("COSYVOICE_MAX_CONCURRENT", "max_concurrent"),
        ("COSYVOICE_TRT_CONCURRENT", "trt_concurrent"),
    ):
        value = _env_int(env_name)
        if value is not None:
            cosyvoice[key] = value
    for env_name, key in (
        ("COSYVOICE_LOAD_VLLM", "load_vllm"),
        ("COSYVOICE_LOAD_TRT", "load_trt"),
    ):
        enabled = _env_bool(env_name)
        if enabled is not None:
            cosyvoice[key] = enabled

    # vLLM sizing is per-card (32GB vs 48GB vs 80GB) and per-image, so it must
    # be overridable from the pod spec without a config variant per host.
    vllm = cosyvoice.get("vllm")
    if not isinstance(vllm, dict):
        vllm = {}
        cosyvoice["vllm"] = vllm
    for env_name, key, cast in (
        ("COSYVOICE_VLLM_GPU_MEMORY_UTILIZATION", "gpu_memory_utilization", _env_float),
        ("COSYVOICE_VLLM_MAX_MODEL_LEN", "max_model_len", _env_int),
        ("COSYVOICE_VLLM_MAX_NUM_SEQS", "max_num_seqs", _env_int),
        ("COSYVOICE_VLLM_MAX_NUM_BATCHED_TOKENS", "max_num_batched_tokens", _env_int),
    ):
        value = cast(env_name)
        if value is not None:
            vllm[key] = value

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

    cloning_dir = os.getenv("COSYVOICE_CLONING_DIR")
    if cloning_dir:
        raw.setdefault("cloning", {})["dir"] = cloning_dir
    cloning_enabled = _env_bool("COSYVOICE_CLONING_ENABLED")
    if cloning_enabled is not None:
        raw.setdefault("cloning", {})["enabled"] = cloning_enabled

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
