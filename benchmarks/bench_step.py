"""Microbenchmark Qwen2 decode step: attention mask handling / cache growth."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main", "third_party", "Matcha-TTS"))

import torch

torch.set_num_threads(8)
from cosyvoice.cli.cosyvoice import AutoModel

tts = AutoModel(model_dir=os.path.join(ROOT, "models", "Fun-CosyVoice3-0.5B"), fp16=True)
enc = tts.model.llm.llm
enc.half()
hf = enc.model.to("cuda").eval()
print("attn impl:", hf.config._attn_implementation, "dtype:", next(hf.parameters()).dtype, flush=True)


def bench(label, proper_mask, n=40, prompt_len=120):
    ids = torch.randint(0, 1000, (1, prompt_len), device="cuda")
    x = hf.model.embed_tokens(ids)
    mask = torch.ones((1, prompt_len), device="cuda", dtype=torch.long)
    with torch.inference_mode():
        out = hf(inputs_embeds=x, attention_mask=mask, use_cache=True, return_dict=True)
        cache = out.past_key_values
        # warmup
        for _ in range(3):
            m = torch.ones((1, cache.get_seq_length() + 1), device="cuda", dtype=torch.long) if proper_mask else torch.ones((1, 1), device="cuda", dtype=torch.long)
            emb = hf.model.embed_tokens(torch.zeros((1, 1), dtype=torch.long, device="cuda"))
            out = hf(inputs_embeds=emb, attention_mask=m, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            m = torch.ones((1, cache.get_seq_length() + 1), device="cuda", dtype=torch.long) if proper_mask else torch.ones((1, 1), device="cuda", dtype=torch.long)
            emb = hf.model.embed_tokens(torch.zeros((1, 1), dtype=torch.long, device="cuda"))
            out = hf(inputs_embeds=emb, attention_mask=m, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
        torch.cuda.synchronize()
        dt = time.time() - t0
    print(f"[{label}] {n} steps in {dt:.2f}s -> {n / dt:.1f} tok/s ({dt / n * 1000:.1f} ms/step), cache={cache.get_seq_length()}", flush=True)


bench("mask-len1", False)
bench("mask-full", True)
bench("mask-len1-b", False)
bench("mask-full-b", True)

# Time just the sampling path
w = torch.randn(151936, device="cuda").log_softmax(dim=0)
