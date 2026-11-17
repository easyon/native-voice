"""Stage-level profile of CosyVoice3 streaming: LLM tokens / flow / hift."""
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

stats = {"llm": [], "flow": [], "hift": []}


def wrap(name, obj, attr, is_generator):
    orig = getattr(obj, attr)

    if is_generator:
        def gen(*a, **k):
            t0 = time.time()
            n = 0
            for x in orig(*a, **k):
                n += 1
                yield x
            torch.cuda.synchronize()
            stats[name].append((time.time() - t0, n))
        setattr(obj, attr, gen)
    else:
        def fn(*a, **k):
            torch.cuda.synchronize()
            t0 = time.time()
            r = orig(*a, **k)
            torch.cuda.synchronize()
            stats[name].append((time.time() - t0, 0))
            return r
        setattr(obj, attr, fn)


wrap("llm", tts.model.llm, "inference", True)
wrap("flow", tts.model.flow, "inference", False)
wrap("hift", tts.model.hift, "inference", False)


def speak(text):
    for k in stats:
        stats[k].clear()
    t0 = time.time()
    first = None
    audio = 0.0
    chunk_times = []
    for j in tts.inference_zero_shot(text, "", "", zero_shot_spk_id="voice", stream=True):
        now = time.time() - t0
        if first is None:
            first = now
        chunk_times.append(now)
        audio += j["tts_speech"].shape[1] / tts.sample_rate
    print(f"text: {text}")
    print(f"  first={first:.2f}s total={time.time() - t0:.2f}s audio={audio:.2f}s chunks={chunk_times}")
    print(f"  llm: {[(round(t, 2), n) for t, n in stats['llm']]}")
    print(f"  flow: {[round(t, 2) for t, _ in stats['flow']]}")
    print(f"  hift: {[round(t, 2) for t, _ in stats['hift']]}", flush=True)


speak("你好。")
speak("好的，我明白了。你想了解哪方面的内容呢？")
speak("今天天气不错，我们可以聊点轻松的话题。")
