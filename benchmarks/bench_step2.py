"""Is decode CPU-dispatch-bound? Compare wall time vs CUDA event time + matmul health."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CosyVoice-main", "third_party", "Matcha-TTS"))

import torch

torch.set_num_threads(8)
from cosyvoice.cli.cosyvoice import AutoModel

print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
print("tf32", torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32", torch.backends.cudnn.allow_tf32, flush=True)

# matmul health
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(3):
    a @ b
torch.cuda.synchronize()
t0 = time.time()
n = 20
for _ in range(n):
    a @ b
torch.cuda.synchronize()
dt = time.time() - t0
print(f"fp16 matmul 4096^3 x{n}: {dt:.2f}s -> {n * 2 * 4096 ** 3 / dt / 1e12:.1f} TFLOPS", flush=True)

tts = AutoModel(model_dir=os.path.join(ROOT, "models", "Fun-CosyVoice3-0.5B"), fp16=True)
enc = tts.model.llm.llm
enc.half()
hf = enc.model.to("cuda").eval()
print("param device", next(hf.parameters()).device, next(hf.parameters()).dtype, flush=True)

prompt_len = 120
ids = torch.randint(0, 1000, (1, prompt_len), device="cuda")
x = hf.model.embed_tokens(ids)
mask = torch.ones((1, prompt_len), device="cuda", dtype=torch.long)
with torch.inference_mode():
    out = hf(inputs_embeds=x, attention_mask=mask, use_cache=True, return_dict=True)
    cache = out.past_key_values
    last = out.logits[:, -1:]
    emb = hf.model.embed_tokens(torch.zeros((1, 1), dtype=torch.long, device="cuda"))

    def step(cache, emb):
        m = torch.ones((1, cache.get_seq_length() + 1), device="cuda", dtype=torch.long)
        o = hf(inputs_embeds=emb, attention_mask=m, past_key_values=cache, use_cache=True, return_dict=True)
        return o.past_key_values, o.logits[:, -1:]

    for _ in range(5):
        cache, last = step(cache, emb)
    torch.cuda.synchronize()

    n = 40
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    t0 = time.time()
    for _ in range(n):
        cache, last = step(cache, emb)
    wall = time.time() - t0
    end.record()
    torch.cuda.synchronize()
    gpu = start.elapsed_time(end) / 1000
    print(f"decode {n} steps: wall={wall:.3f}s gpu={gpu:.3f}s -> wall {n / wall:.1f} tok/s, gpu {n / gpu:.1f} tok/s, cache={cache.get_seq_length()}", flush=True)
