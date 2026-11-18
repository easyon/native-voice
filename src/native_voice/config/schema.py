"""Typed service configuration.

Every field can be overridden from ``configs/service.yaml``; a small set of
deployment-critical fields also accepts NV_* environment variables in
``loader.py``.  Model/data paths are always resolved to absolute paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8090
    cors_origins: list[str] = field(default_factory=list)
    max_message_bytes: int = 65536
    ws_idle_timeout_s: float = 60.0
    graceful_shutdown_s: float = 10.0


@dataclass
class AuthConfig:
    db_path: Path = Path(".data/native_voice.db")
    dev_key: bool = False
    ticket_ttl_s: int = 60
    ticket_secret: str = ""  # empty = random per process (single replica)


@dataclass
class QuotaConfig:
    max_concurrent_tasks: int = 4
    max_concurrent_per_key: int = 2
    rpm_per_key: int = 120
    tts_chars_per_day: int = 200_000
    asr_seconds_per_day: int = 14_400


@dataclass
class ModelsConfig:
    root: Path = Path(".")
    asr_stream_model: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
    asr_device: str = "cuda:0"
    asr_chunk_ms: int = 600
    tts_model: Path = Path("models/Fun-CosyVoice3-0.5B")
    tts_prompt_wav: Path = Path("third_party/CosyVoice-main/asset/zero_shot_prompt.wav")
    tts_prompt_text: str = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    tts_hop: int = 15
    tts_max_cache_len: int = 2048
    use_cuda_graph: bool = True
    warmup: bool = True


@dataclass
class TTSDefaults:
    default_format: str = "pcm"
    default_sample_rate: int = 24000
    supported_formats: list[str] = field(default_factory=lambda: ["pcm", "wav", "mp3", "opus"])
    supported_sample_rates: list[int] = field(
        default_factory=lambda: [8000, 16000, 22050, 24000, 44100, 48000]
    )
    voice_aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class ASRDefaults:
    max_sentence_silence_ms: int = 1300
    language_hints: list[str] = field(default_factory=lambda: ["zh"])
    final_pass: str = "streaming"


@dataclass
class LLMConfig:
    enabled: bool = True
    mode: str = "spawn"  # spawn | external
    server: Path = Path("bin/llama/llama-server.exe")
    model: Path = Path("models/Qwen3.5-2B-Q8_0.gguf")
    port: int = 8081
    threads: int = 8
    batch_threads: int = 16
    ctx: int = 16384          # total llama-server context, split across slots
    slots: int = 2            # -> 8192 tokens per concurrent request
    n_gpu_layers: int = 0     # layers offloaded to GPU (99 = all)
    kv_cache_type: str = "f16"  # f16 | q8_0 | q4_0 (quantized KV needs flash-attn)
    batch_size: int = 2048
    ubatch_size: int = 512
    max_tokens: int = 160     # LLM reply length per turn
    prompt_char_budget: int = 6000  # system + history character budget
    temperature: float = 0.7
    top_p: float = 0.9
    history_turns: int = 8
    warm: bool = True
    # Function calling. Names refer to built-in tools in conversation/tools.py;
    # an empty list disables tools entirely.
    tools: list[str] = field(
        default_factory=lambda: ["get_current_time", "web_search", "web_fetch"]
    )
    tool_choice: str = "auto"  # auto | required | none
    max_tool_rounds: int = 3  # tool executions per user turn
    tool_filler: str = "我查一下。"  # spoken before a tool runs; "" disables
    system_prompt: str = (
        "你是一个温暖、自然的中文语音助手，正在和用户进行实时语音通话。\n"
        "要求：\n"
        "1. 口语化、简短，通常一到三句话；除非用户明确要求，不要长篇大论。\n"
        "2. 不要使用 Markdown、列表、表情符号、括号动作或特殊符号。\n"
        "3. 默认用中文回答；用户使用其他语言时跟随用户。\n"
        "4. 不要提到你无法播放声音，也不要暴露你是文本模型。\n"
        "5. 如果没听清或不确定，礼貌地请用户再说一遍。\n"
    )


@dataclass
class ConversationConfig:
    enabled: bool = True
    endpoint_silence_ms: int = 300
    noise_override_ms: int = 500
    merge_window_s: float = 1.6
    barge_db: float = -30.0
    barge_sustain_ms: int = 400
    # How long the user may keep talking without an ASR partial before the
    # assistant is interrupted anyway (covers slow/starved streaming ASR).
    barge_content_wait_ms: int = 900
    min_speech_ms: int = 240
    max_utterance_s: float = 20.0
    preroll_ms: int = 450
    voice: str = "default"
    input_rate: int = 16000
    output_rate: int = 24000
    speech_gate: bool = True
    speech_gate_model: Path = Path("assets/silero_vad.onnx")
    speech_start_threshold: float = 0.5
    speech_silence_threshold: float = 0.3
    # Short acknowledgements while the user is still speaking. Off by default;
    # enable per deployment after checking the speaker/AEC setup.
    backchannel_enabled: bool = False
    backchannel_min_speech_s: float = 1.5
    backchannel_interval_s: float = 6.0
    backchannel_max_per_utterance: int = 2
    backchannel_volume: float = 0.45
    backchannel_phrases: list[str] = field(
        default_factory=lambda: ["嗯", "嗯嗯", "我在听"]
    )


@dataclass
class ProviderSpec:
    """One selectable engine implementation (ASR or TTS).

    ``python`` may point at a different virtualenv because model runtimes often
    pin conflicting ``transformers``/``torch`` versions. Only the provider
    selected in ``engine.asr_provider`` / ``engine.tts_provider`` is started, so
    the environments can coexist on disk without competing for the GPU.
    """

    name: str = ""
    python: str = ""            # interpreter; empty -> sys.executable
    module: str = "native_voice.engine.cli"
    model: str = ""             # local path or hub id, provider-specific
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineConfig:
    # external: engines are already running (Docker/systemd)
    # spawn: the gateway starts local engine subprocesses (dev / single box)
    mode: str = "external"
    asr_provider: str = "paraformer"
    tts_provider: str = "cosyvoice3"
    asr_http: str = "http://127.0.0.1:8101"
    tts_http: str = "http://127.0.0.1:8102"
    asr_ws: str = "ws://127.0.0.1:8101/asr"
    tts_ws: str = "ws://127.0.0.1:8102/tts"
    asr_port: int = 8101
    tts_port: int = 8102
    token: str = ""


@dataclass
class VoiceStoreConfig:
    dir: Path = Path(".data/voices")
    max_upload_mb: int = 20
    min_seconds: float = 1.5
    max_seconds: float = 30.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    json_logs: bool = True


@dataclass
class VoiceProfile:
    name: str
    prompt_wav: Path
    prompt_text: str
    description: str = ""


@dataclass
class ServiceConfig:
    root: Path
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    quotas: QuotaConfig = field(default_factory=QuotaConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    tts: TTSDefaults = field(default_factory=TTSDefaults)
    asr: ASRDefaults = field(default_factory=ASRDefaults)
    engine: EngineConfig = field(default_factory=EngineConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    voice_store: VoiceStoreConfig = field(default_factory=VoiceStoreConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    providers: dict[str, dict[str, ProviderSpec]] = field(default_factory=dict)
    voices: dict[str, VoiceProfile] = field(default_factory=dict)
