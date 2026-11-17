"""Decode with Qwen2Model (no 152k text lm_head) + CUDA graph."""
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
lm = tts.model.llm
lm.half()
core = lm.llm.model.model.to("cuda").eval()  # Qwen2Model
head = lm.llm_decoder
cfg = core.config

P, MAXLEN, N = 120, 1024, 50
with torch.inference_mode():
    prompt_emb = core.embed_tokens(torch.randint(0, 1000, (1, P), device="cuda"))
    tok_emb = core.embed_tokens(torch.randint(0, 1000, (1, 1), device="cuda"))


def make_cache():
    return StaticCache(config=cfg, max_batch_size=1, max_cache_len=MAXLEN, device="cuda", dtype=torch.float16)


def step(cache, mask, pos, emb):
    o = core(inputs_embeds=emb, attention_mask=mask, past_key_values=cache, cache_position=pos, use_cache=True, return_dict=True)
    return head(o.last_hidden_state[:, -1])


with torch.inference_mode():
    cache = make_cache()
    mask = torch.zeros(1, MAXLEN, dtype=torch.long, device="cuda")
    mask[:, :P] = 1
    core(inputs_embeds=prompt_emb, attention_mask=mask, past_key_values=cache, cache_position=torch.arange(P, device="cuda"), use_cache=True, return_dict=True)
    for _ in range(5):
        pos = torch.tensor([P], device="cuda")
        mask[:, : P + 1] = 1
        logits = step(cache, mask, pos, tok_emb)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(N):
        pos = torch.tensor([P + 1 + i], device="cuda")
        mask[:, P + i] = 1
        logits = step(cache, mask, pos, tok_emb)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"core eager: {N} steps {dt:.3f}s -> {N / dt:.1f} tok/s", flush=True)

    # graph
    cache = make_cache()
    mask = torch.zeros(1, MAXLEN, dtype=torch.long, device="cuda")
    mask[:, :P] = 1
    core(inputs_embeds=prompt_emb, attention_mask=mask, past_key_values=cache, cache_position=torch.arange(P, device="cuda"), use_cache=True, return_dict=True)
    s_emb = tok_emb.clone()
    s_mask = torch.zeros(1, MAXLEN, dtype=torch.long, device="cuda")
    s_mask[:, :P] = 1
    s_pos = torch.tensor([P], device="cuda", dtype=torch.long)
    for _ in range(3):
        step(cache, s_mask, s_pos, s_emb)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        s_logits = step(cache, s_mask, s_pos, s_emb)
    torch.cuda.synchronize()
    for i in range(5):
        s_pos.fill_(P + i)
        s_mask[:, P + i] = 1
        g.replay()
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(N):
        s_pos.fill_(P + 5 + i)
        s_mask[:, P + 5 + i] = 1
        g.replay()
        _ = s_logits.argmax().item()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"core graph+sync: {N} steps {dt:.3f}s -> {N / dt:.1f} tok/s", flush=True)
    print("logits", s_logits.shape, s_logits.dtype, flush=True)
