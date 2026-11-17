"""Runtime configuration for the voice call."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    root: Path = ROOT

    # --- audio I/O ---
    mic_device: int | None = None       # None -> sounddevice default input
    speaker_device: int | None = None   # None -> sounddevice default output
    input_rate: int = 16000
    output_rate: int = 24000
    output_gain: float = 1.0
    no_play: bool = False

    # --- VAD / turn taking ---
    frame_ms: int = 20
    endpoint_silence_ms: int = 500      # silence before the user's turn is considered over
    merge_window_s: float = 1.6         # merge a quick follow-up when no AI audio was heard yet
    min_speech_ms: int = 240            # ignore shorter blips
    max_utterance_s: float = 20.0
    preroll_ms: int = 450               # audio kept before speech onset (also used on barge-in)
    silence_db_abs: float = -46.0       # absolute speech floor while idle
    barge_db: float = -30.0             # absolute floor while the agent is talking
    barge_sustain_ms: int = 160         # speech duration needed to trigger barge-in
    echo_tail_s: float = 0.45           # keep the barge threshold for a moment after playback
    vad_warmup_frames: int = 40         # 800 ms calibration (skips stream-open settling)

    # --- ASR ---
    # streaming: online Paraformer only (default, lowest latency, no punctuation)
    # hybrid: streaming partials + SenseVoice final (more accurate punctuation/ITN)
    # sensevoice: offline SenseVoice after VAD endpoint (previous behaviour)
    asr_mode: str = "streaming"
    asr_model: str = "iic/SenseVoiceSmall"
    asr_stream_model: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
    asr_stream_chunk_ms: int = 600        # partial cadence only; endpoint flush is immediate
    asr_device: str = "cuda:0"
    asr_language: str = "zh"
    show_partials: bool = True

    # --- LLM (llama.cpp server, CPU) ---
    llm_model: Path = ROOT / "models" / "Qwen3-1.7B-Q8_0.gguf"
    llama_server: Path = ROOT / "bin" / "llama" / "llama-server.exe"
    llama_port: int = 8081
    llama_threads: int = 8
    llama_batch_threads: int = 16
    llama_ctx: int = 4096
    max_tokens: int = 160
    temperature: float = 0.7
    top_p: float = 0.9
    history_turns: int = 8
    warm_llm: bool = True           # prefill the system prompt into llama-server's cache
    system_prompt: str = (
        "你是一个温暖、自然的中文语音助手，正在和用户进行实时语音通话。\n"
        "要求：\n"
        "1. 口语化、简短，通常一到三句话；除非用户明确要求，否则不要长篇大论。\n"
        "2. 不要使用 Markdown、列表、表情符号、括号动作或特殊符号。\n"
        "3. 默认用中文回答；用户使用其他语言时跟随用户。\n"
        "4. 不要提到你无法播放声音，也不要暴露你是文本模型。\n"
        "5. 如果没听清或不确定，礼貌地请用户再说一遍。\n"
    )

    # --- TTS (CosyVoice3 on GPU) ---
    tts_model: Path = ROOT / "models" / "Fun-CosyVoice3-0.5B"
    prompt_wav: Path = ROOT / "third_party" / "CosyVoice-main" / "asset" / "zero_shot_prompt.wav"
    prompt_text: str = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    tts_hop: int = 15                   # speech tokens per first streaming chunk
    tts_max_cache_len: int = 2048
    use_cuda_graph: bool = True
    warmup_tts: bool = True

    # --- behaviour ---
    text_mode: bool = False
    wav_input: Path | None = None
    verbose: bool = False
    save_audio: bool = False
    log_dir: Path = ROOT / "audio" / "calls"
