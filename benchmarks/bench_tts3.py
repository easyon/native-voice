"""TTS speed variants: cached speaker, fp16 weights, fewer flow steps, no CFG."""
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main", "third_party", "Matcha-TTS"))

import torch

torch.set_num_threads(8)
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = os.path.join(ROOT, "models", "Fun-CosyVoice3-0.5B")
PROMPT = os.path.join(ROOT, "third_party", "CosyVoice-main", "asset", "zero_shot_prompt.wav")
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

print("loading...", flush=True)
tts = AutoModel(model_dir=MODEL_DIR, fp16=True)
tts.add_zero_shot_spk(PROMPT_TEXT, PROMPT, "voice")
print("ready", flush=True)

OUT = os.path.join(ROOT, "audio", "variants")
os.makedirs(OUT, exist_ok=True)
TEXTS = [
    "好的，我明白了。",
    "好的，我明白了。你想了解哪方面的内容呢？",
    "今天天气不错，我们可以聊点轻松的话题。",
]


def run(tag, text, save=True):
    t0 = time.time()
    first = None
    audio = 0.0
    chunks = []
    for j in tts.inference_zero_shot(text, "", "", zero_shot_spk_id="voice", stream=True):
        if first is None:
            first = time.time() - t0
        chunks.append(j["tts_speech"])
        audio += j["tts_speech"].shape[1] / tts.sample_rate
    dt = time.time() - t0
    if save and chunks:
        wav = torch.cat(chunks, dim=1)
        torchaudio.save(os.path.join(OUT, f"{tag}.wav"), wav, tts.sample_rate)
    print(f"[{tag}] first={first:.2f} total={dt:.2f} audio={audio:.2f} rtf={dt / audio:.2f}", flush=True)
    return first, dt, audio


def bench(tag):
    for text in TEXTS:
        run(f"{tag}-{TEXTS.index(text)}", text)


# warmup / baseline (cached speaker)
run("warmup", TEXTS[0])
bench("base")

# fp16 weights
for name in ("llm", "flow", "hift"):
    getattr(tts.model, name).half()
bench("half")

# 5 flow steps
from cosyvoice.flow.flow_matching import CausalConditionalCFM

_orig_forward = CausalConditionalCFM.forward
N_STEPS = 5


def forward_fewer(self, mu, mask, n_timesteps, **kw):
    return _orig_forward(self, mu, mask, N_STEPS, **kw)


CausalConditionalCFM.forward = forward_fewer
bench("half5")

# no classifier-free guidance (batch 1 instead of 2)
dec = tts.model.flow.decoder


def solve_euler_nocfg(self, x, t_span, mu, mask, spks, cond, streaming=False):
    t = t_span[0].unsqueeze(dim=0)
    dt = t_span[1] - t_span[0]
    sol = []
    for step in range(1, len(t_span)):
        dphi_dt = self.forward_estimator(x, mask, mu, t, spks, cond, streaming)
        x = x + dt * dphi_dt
        t = t + dt
        sol.append(x)
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    return sol[-1].float()


dec.solve_euler = types.MethodType(solve_euler_nocfg, dec)
bench("half5nocfg")
