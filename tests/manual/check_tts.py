import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main", "third_party", "Matcha-TTS"))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = os.path.join(ROOT, "models", "Fun-CosyVoice3-0.5B")
PROMPT = os.path.join(ROOT, "third_party", "CosyVoice-main", "asset", "zero_shot_prompt.wav")

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
t0 = time.time()
m = AutoModel(model_dir=MODEL_DIR)
print(f"[load] {time.time()-t0:.1f}s sample_rate={m.sample_rate}", flush=True)

TEXT = "你好，我是你的本地语音助手，现在正在用 CPU 合成声音。"
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
for stream in (False, True):
    t0 = time.time()
    first = None
    total_audio = 0.0
    out = []
    for i, j in enumerate(m.inference_zero_shot(TEXT, PROMPT_TEXT, PROMPT, stream=stream)):
        if first is None:
            first = time.time() - t0
        total_audio += j["tts_speech"].shape[1] / m.sample_rate
        out.append(j["tts_speech"])
    elapsed = time.time() - t0
    speech = torch.cat(out, dim=1)
    torchaudio.save(os.path.join(ROOT, "audio", f"test_stream_{stream}.wav"), speech, m.sample_rate)
    print(f"[stream={stream}] first_chunk={first:.2f}s total={elapsed:.2f}s audio={total_audio:.2f}s RTF={elapsed/max(total_audio,0.01):.2f}", flush=True)
