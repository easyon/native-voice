"""GPU benchmark: CosyVoice3 streaming TTS + FunASR VAD/ASR on the local 8GB GPU.

Run:
  C:\\Users\\xcc\\AppData\\Local\\miniconda3\\envs\\cosyvoice\\python.exe app\\bench_gpu.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main", "third_party", "Matcha-TTS"))

import torch

torch.set_num_threads(8)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
print("gpu", torch.cuda.get_device_name(0), flush=True)

MODEL_DIR = os.path.join(ROOT, "models", "Fun-CosyVoice3-0.5B")
PROMPT = os.path.join(ROOT, "third_party", "CosyVoice-main", "asset", "zero_shot_prompt.wav")
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"


def vram():
    return torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9


print("\n=== CosyVoice3 ===", flush=True)
from cosyvoice.cli.cosyvoice import AutoModel

t0 = time.time()
model = AutoModel(model_dir=MODEL_DIR, fp16=True)
print(f"load {time.time() - t0:.1f}s sr={model.sample_rate} vram alloc/max={vram()}", flush=True)


def speak(tag, text, sentences=1):
    t0 = time.time()
    first = None
    audio = 0.0
    n = 0
    for j in model.inference_zero_shot(text, PROMPT_TEXT, PROMPT, stream=True):
        if first is None:
            first = time.time() - t0
        n += 1
        audio += j["tts_speech"].shape[1] / model.sample_rate
    dt = time.time() - t0
    print(
        f"[{tag}] chars={len(text)} first_chunk={first:.2f}s total={dt:.2f}s "
        f"audio={audio:.2f}s RTF={dt / audio:.2f} chunks={n} vram={vram()}",
        flush=True,
    )
    return first, dt, audio


speak("short", "好的，我明白了。")
speak("medium", "好的，我明白了。你想了解哪方面的内容呢？")
speak("long", "这是一个用于测试实时语音对话的长句子，主要目的是观察流式合成的首块延迟以及整体合成速度。")

# A quick simulation of sentence-pipelined synthesis (one call per sentence).
t0 = time.time()
first = None
audio = 0.0
for s in ["你好，我是你的本地语音助手。", "我可以陪你聊天，也可以回答你的问题。", "有什么想说的，直接告诉我就可以。"]:
    for j in model.inference_zero_shot(s, PROMPT_TEXT, PROMPT, stream=True):
        if first is None:
            first = time.time() - t0
        audio += j["tts_speech"].shape[1] / model.sample_rate
dt = time.time() - t0
print(f"[3-sentence] first={first:.2f}s total={dt:.2f}s audio={audio:.2f}s RTF={dt / audio:.2f}", flush=True)


def release(model):
    del model
    torch.cuda.empty_cache()


release(model)
model = None
print(f"after tts free vram={vram()}", flush=True)

print("\n=== FunASR ===", flush=True)
from funasr import AutoModel as FunASR

t0 = time.time()
vad = FunASR(
    model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    disable_update=True,
    device="cuda:0",
    disable_pbar=True,
)
asr = FunASR(
    model="iic/SenseVoiceSmall",
    disable_update=True,
    device="cuda:0",
    disable_pbar=True,
)
print(f"load {time.time() - t0:.1f}s vram={vram()}", flush=True)

wav = os.path.join(ROOT, "audio", "asr_test.wav")
for i in range(3):
    t0 = time.time()
    res = asr.generate(input=wav, language="zh", use_itn=True, batch_size_s=60)
    print(f"[asr {i}] {time.time() - t0:.2f}s -> {res[0]['text']}", flush=True)
