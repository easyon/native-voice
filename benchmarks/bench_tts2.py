import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main", "third_party", "Matcha-TTS"))
import torch

torch.set_num_threads(16)
from cosyvoice.cli.cosyvoice import AutoModel

m = AutoModel(model_dir=os.path.join(ROOT, "models", "Fun-CosyVoice3-0.5B"))
PROMPT = os.path.join(ROOT, "third_party", "CosyVoice-main", "asset", "zero_shot_prompt.wav")
PT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

def run(tag, text, stream):
    t0 = time.time(); first = None; audio = 0.0
    for j in m.inference_zero_shot(text, PT, PROMPT, stream=stream):
        if first is None: first = time.time() - t0
        audio += j["tts_speech"].shape[1] / m.sample_rate
    dt = time.time() - t0
    print(f"[{tag}] first={first:.2f}s total={dt:.2f}s audio={audio:.2f}s RTF={dt/audio:.2f}", flush=True)

run("short-stream", "好的，我明白了。", True)
run("med-stream", "好的，我明白了。你想了解哪方面的内容呢？", True)

# int8 dynamic quantization of the LLM backbone
try:
    q = torch.quantization.quantize_dynamic(m.model.llm.llm, {torch.nn.Linear}, dtype=torch.qint8)
    m.model.llm.llm = q
    run("q8-llm-stream", "好的，我明白了。你想了解哪方面的内容呢？", True)
except Exception as e:
    print("quant failed:", repr(e), flush=True)
