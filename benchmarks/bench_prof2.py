"""torch.profiler on Qwen2 decode steps."""
import os
import sys

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

prompt_len = 120
ids = torch.randint(0, 1000, (1, prompt_len), device="cuda")
x = hf.model.embed_tokens(ids)
mask = torch.ones((1, prompt_len), device="cuda", dtype=torch.long)
with torch.inference_mode():
    out = hf(inputs_embeds=x, attention_mask=mask, use_cache=True, return_dict=True)
    cache = out.past_key_values
    emb = hf.model.embed_tokens(torch.zeros((1, 1), dtype=torch.long, device="cuda"))

    def step(cache):
        m = torch.ones((1, cache.get_seq_length() + 1), device="cuda", dtype=torch.long)
        o = hf(inputs_embeds=emb, attention_mask=m, past_key_values=o_cache[0], use_cache=True, return_dict=True)
        o_cache[0] = o.past_key_values
        return o

    o_cache = [cache]
    for _ in range(3):
        step(cache)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(5):
            o = step(cache)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
