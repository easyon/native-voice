"""Test vectorized nucleus sampling to fix CosyVoice3 speech-token throughput."""
import functools
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main", "third_party", "Matcha-TTS"))

import torch

torch.set_num_threads(8)
from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = os.path.join(ROOT, "models", "Fun-CosyVoice3-0.5B")
PROMPT = os.path.join(ROOT, "third_party", "CosyVoice-main", "asset", "zero_shot_prompt.wav")
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

tts = AutoModel(model_dir=MODEL_DIR, fp16=True)
tts.add_zero_shot_spk(PROMPT_TEXT, PROMPT, "voice")
tts.model.llm.half()
tts.model.flow.half()
tts.model.hift.half()
llm = tts.model.llm
TEXTS = [
    "好的，我明白了。",
    "好的，我明白了。你想了解哪方面的内容呢？",
    "今天天气不错，我们可以聊点轻松的话题。",
]


def run(tag, text):
    t0 = time.time()
    first = None
    audio = 0.0
    for j in tts.inference_zero_shot(text, "", "", zero_shot_spk_id="voice", stream=True):
        if first is None:
            first = time.time() - t0
        audio += j["tts_speech"].shape[1] / tts.sample_rate
    dt = time.time() - t0
    print(f"[{tag}] first={first:.2f} total={dt:.2f} audio={audio:.2f} rtf={dt / audio:.2f}", flush=True)


run("warmup", TEXTS[0])
for i, t in enumerate(TEXTS):
    run(f"orig-{i}", t)


def fast_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    probs = weighted_scores.softmax(dim=0)
    topv, topi = probs.topk(min(top_k, probs.numel()))
    cum = topv.cumsum(dim=0)
    keep = cum - topv < top_p
    n = int(keep.sum().item())
    idx = topi[:n]
    p = topv[:n]
    top_ids = int(idx[torch.multinomial(p / p.sum(), 1)].item())
    if decoded_tokens:
        recent = torch.tensor(decoded_tokens[-win_size:], device=probs.device)
        if int((recent == top_ids).sum().item()) >= win_size * tau_r:
            weighted_scores[top_ids] = -float("inf")
            top_ids = int(weighted_scores.softmax(dim=0).multinomial(1).item())
    return top_ids


kwargs = getattr(llm.sampling, "keywords", {})
print("sampling kwargs", kwargs, flush=True)
llm.sampling = functools.partial(fast_sampling, **kwargs)
for i, t in enumerate(TEXTS):
    run(f"fast-{i}", t)

# Also try without repetition penalty / plain top-k for reference.
def plain_topk(weighted_scores, decoded_tokens, sampling, top_k=25, **kw):
    topv, topi = weighted_scores.softmax(dim=0).topk(top_k)
    return int(topi[torch.multinomial(topv / topv.sum(), 1)].item())


llm.sampling = functools.partial(plain_topk, top_k=25)
for i, t in enumerate(TEXTS):
    run(f"plain-{i}", t)
