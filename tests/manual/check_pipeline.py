"""Offline end-to-end check: wav -> VAD -> ASR -> LLM -> TTS -> ASR."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "desktop"
if str(DESKTOP) not in sys.path:
    sys.path.insert(0, str(DESKTOP))

from nv.config import Config  # noqa: E402
from nv.conversation import SentenceSplitter  # noqa: E402
from nv.vad import EndpointVad  # noqa: E402

OUT = ROOT / "audio" / "selftest"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    cfg = Config()
    cfg.verbose = True

    print("=== 1. VAD ===", flush=True)
    wav, sr = sf.read(str(ROOT / "audio" / "asr_test.wav"), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    print(f"wav {len(wav) / sr:.2f}s @ {sr}Hz", flush=True)
    if sr != cfg.input_rate:
        from scipy.signal import resample_poly

        wav = resample_poly(wav, cfg.input_rate, sr)
    pad = np.zeros(int(0.5 * cfg.input_rate), dtype=np.float32)
    tail = np.zeros(int(1.0 * cfg.input_rate), dtype=np.float32)
    stream = np.concatenate([pad, wav, tail])

    vad = EndpointVad(cfg)
    events = []
    step = int(cfg.input_rate * cfg.frame_ms / 1000)
    for i in range(0, len(stream) - step + 1, step):
        events.extend(vad.feed(stream[i:i + step]))
    print("events:", [(e.kind, round(e.duration, 2), round(e.db, 1)) for e in events], flush=True)
    utter = next((e.audio for e in events if e.kind == "end"), None)
    if utter is None:
        print("VAD 未检测到一句话", flush=True)
        return

    print("\n=== 2. ASR ===", flush=True)
    from nv.asr import ASR

    asr = ASR(cfg)
    t0 = time.time()
    asr.load()
    print(f"load {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    text = asr.transcribe(utter)
    print(f"ASR {time.time() - t0:.2f}s -> {text!r}", flush=True)

    print("\n=== 3. LLM ===", flush=True)
    from nv.llm import LlamaClient

    llm = LlamaClient(cfg)
    llm.start()
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": text or "你好"},
    ]
    splitter = SentenceSplitter()
    sentences: list[str] = []
    t0 = time.time()
    first = None
    first_sent = None
    reply = ""
    for delta in llm.chat_stream(messages):
        if first is None:
            first = time.time() - t0
        reply += delta
        for s in splitter.feed(delta):
            if first_sent is None:
                first_sent = time.time() - t0
            sentences.append(s)
    sentences.extend(splitter.flush())
    print(f"reply: {reply!r}")
    print(f"first_delta={first:.2f}s first_sentence={first_sent:.2f}s total={time.time() - t0:.2f}s", flush=True)

    print("\n=== 4. TTS + round-trip ASR ===", flush=True)
    from nv.tts import CosyVoiceTTS

    tts = CosyVoiceTTS(cfg)
    test_sents = [s for s in sentences if s][:2] or ["你好，我是你的本地语音助手。"]
    for idx, sent in enumerate(test_sents):
        t0 = time.time()
        first_chunk = None
        audio = []
        for chunk in tts.synthesize_stream(sent):
            if first_chunk is None:
                first_chunk = time.time() - t0
            audio.append(chunk)
        full = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
        dur = len(full) / cfg.output_rate
        path = OUT / f"tts_{idx}.wav"
        sf.write(str(path), full, cfg.output_rate)
        print(
            f"[{sent}] first={first_chunk:.2f}s total={time.time() - t0:.2f}s audio={dur:.2f}s "
            f"rtf={(time.time() - t0) / max(dur, 1e-3):.2f}",
            flush=True,
        )
        from scipy.signal import resample_poly

        wav16 = resample_poly(full, 2, 3)
        print(f"  回读: {asr.transcribe(wav16)!r}", flush=True)

    import torch

    print(f"\nVRAM alloc={torch.cuda.memory_allocated() / 1e9:.2f}GB peak={torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
    llm.close()
    tts.close()


if __name__ == "__main__":
    main()
