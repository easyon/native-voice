"""Prototype: CUDA-graph the Qwen2 decode step with StaticCache."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main", "third_party", "Matcha-TTS"))

import torch
from transformers import StaticCache

torch.set_num_threads(8)
from cosyvoice.cli.cosyvoice import AutoModel

tts = AutoModel(model_dir=os.path.join(ROOT, "models", "Fun-CosyVoice3-0.5B"), fp16=True)
enc = tts.model.llm.llm
enc.half()
hf = enc.model.to("cuda").eval()
cfg = hf.config

P, MAXLEN, N = 120, 1024, 40
prompt_ids = torch.randint(0, 1000, (1, P), device="cuda")
with torch.inference_mode():
    prompt_emb = hf.model.embed_tokens(prompt_ids)
    tok = torch.randint(0, 1000, (1, 1), device="cuda")
    tok_emb = hf.model.embed_tokens(tok)


def make_cache():
    return StaticCache(config=cfg, max_batch_size=1, max_cache_len=MAXLEN, device="cuda", dtype=torch.float16)


def prefill(cache):
    mask = torch.zeros(1, MAXLEN, dtype=torch.long, device="cuda")
    mask[:, :P] = 1
    pos = torch.arange(P, device="cuda")
    out = hf(inputs_embeds=prompt_emb, attention_mask=mask, past_key_values=cache, cache_position=pos, use_cache=True, return_dict=True)
    return out.logits[:, -1].clone()


def decode(cache, mask, pos, emb):
    return hf(inputs_embeds=emb, attention_mask=mask, past_key_values=cache, cache_position=pos, use_cache=True, return_dict=True)


with torch.inference_mode():
    # reference eager step
    ref_cache = make_cache()
    prefill(ref_cache)
    m = torch.zeros(1, MAXLEN, dtype=torch.long, device="cuda")
    m[:, : P + 1] = 1
    ref = decode(ref_cache, m, torch.tensor([P], device="cuda"), tok_emb).logits[:, -1].clone()

    # graph
    cache = make_cache()
    prefill(cache)
    static_emb = torch.zeros_like(tok_emb)
    static_mask = torch.zeros(1, MAXLEN, dtype=torch.long, device="cuda")
    static_mask[:, : P + 1] = 1
    static_pos = torch.tensor([P], device="cuda", dtype=torch.long)
    static_emb.copy_(tok_emb)

    for _ in range(3):
        decode(cache, static_mask, static_pos, static_emb)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = decode(cache, static_mask, static_pos, static_emb)
    torch.cuda.synchronize()

    static_emb.copy_(tok_emb)
    static_pos.fill_(P)
    static_mask.zero_()
    static_mask[:, : P + 1] = 1
    g.replay()
    torch.cuda.synchronize()
    got = static_out.logits[:, -1].clone()
    print("max|ref-graph| =", (ref - got).abs().max().item(), "argmax", ref.argmax().item(), got.argmax().item(), flush=True)

    # benchmark replay
    n = 50
    # advance positions within max len
    t0 = time.time()
    for i in range(n):
        pos = P + 1 + i
        static_pos.fill_(pos)
        static_mask[:, pos - 1] = 1
        g.replay()
    torch.cuda.synchronize()
    wall = time.time() - t0
    print(f"graph replay {n}: {wall:.3f}s -> {n / wall:.1f} tok/s ({wall / n * 1000:.2f} ms/step)", flush=True)

    # per-step with sampling-equivalent sync
    t0 = time.time()
    for i in range(n):
        pos = P + 1 + i
        static_pos.fill_(pos)
        g.replay()
        _ = static_out.logits[:, -1].argmax().item()
    dt = time.time() - t0
    print(f"graph replay+sync {n}: {dt:.3f}s -> {n / dt:.1f} tok/s", flush=True)
