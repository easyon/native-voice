"""Streaming CosyVoice3 TTS on the GPU, tuned for low first-chunk latency.

Optimizations applied on top of the stock CosyVoice3 inference path:
  * zero-shot speaker embedding is computed once and cached;
  * weights run in fp16 (autocast + half parameters);
  * vectorised nucleus sampling (the stock version syncs the GPU ~25x per token);
  * token_hop_len is reset per sentence (upstream mutates it to 100 and leaks it);
  * the Qwen2 speech-token decoder is CUDA-graphed with a StaticCache (optional).
"""
from __future__ import annotations

import functools
import sys
import threading
import types
from pathlib import Path

from .config import Config

ROOT = Path(__file__).resolve().parents[2]
_COSY = ROOT / "third_party" / "CosyVoice-main"
for p in (str(_COSY), str(_COSY / "third_party" / "Matcha-TTS")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _fast_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    probs = weighted_scores.softmax(dim=0)
    topv, topi = probs.topk(min(top_k, probs.numel()))
    cum = topv.cumsum(dim=0)
    keep = cum - topv < top_p
    n = int(keep.sum().item())
    if n <= 0:
        n = 1
    idx = topi[:n]
    p = topv[:n]
    top_ids = int(idx[_TORCH.multinomial(p / p.sum(), 1)].item())
    if decoded_tokens:
        recent = _TORCH.tensor(decoded_tokens[-win_size:], device=probs.device)
        if int((recent == top_ids).sum().item()) >= win_size * tau_r:
            weighted_scores[top_ids] = -float("inf")
            top_ids = int(_TORCH.multinomial(weighted_scores.softmax(dim=0), 1).item())
    return top_ids


_TORCH = None


class CosyVoiceTTS:
    def __init__(self, cfg: Config):
        global _TORCH
        import torch

        _TORCH = torch
        self.cfg = cfg
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.spk_id = "native_voice"
        self._hop = cfg.tts_hop
        self._graph_ok = False
        self._orig_wrapper = None

        print(f"[TTS] 加载 CosyVoice3 ({self.device}) ...", flush=True)
        # We call inference with text_frontend=False, so skip wetext/cosyvoice's
        # network-backed normalizer entirely.
        sys.modules.setdefault("wetext", None)
        import cosyvoice.cli.cosyvoice as cvcli

        cvcli.tqdm = lambda it, *a, **k: it
        from cosyvoice.cli.cosyvoice import AutoModel

        self.tts = AutoModel(model_dir=str(cfg.tts_model), fp16=True)
        self.tts.add_zero_shot_spk(cfg.prompt_text, str(cfg.prompt_wav), self.spk_id)
        for name in ("llm", "flow", "hift"):
            getattr(self.tts.model, name).half()
        self._patch_sampling()
        if cfg.use_cuda_graph:
            try:
                self._setup_graph()
            except Exception as exc:
                self._graph_ok = False
                print(f"[TTS] CUDA graph 不可用，使用常规解码: {exc!r}", flush=True)
        if cfg.warmup_tts:
            self._warmup()
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[TTS] 就绪，显存 {alloc:.2f}GB (峰值 {peak:.2f}GB)", flush=True)
        else:
            print("[TTS] 就绪 (CPU)", flush=True)

    # ------------------------------------------------------------- tuning
    def _patch_sampling(self):
        lm = self.tts.model.llm
        kwargs = getattr(lm.sampling, "keywords", {}) or {}
        lm.sampling = functools.partial(_fast_sampling, **kwargs)

    def _setup_graph(self):
        torch = self.torch
        from transformers import StaticCache

        lm = self.tts.model.llm
        core = lm.llm.model.model          # Qwen2Model (no 152k text lm_head)
        self._head = lm.llm_decoder
        device = next(core.parameters()).device
        dtype = next(core.parameters()).dtype
        max_len = int(self.cfg.tts_max_cache_len)

        self._g_cache = StaticCache(
            config=core.config,
            max_batch_size=1,
            max_cache_len=max_len,
            device=device,
            dtype=dtype,
        )
        self._g_mask = torch.zeros(1, max_len, dtype=torch.long, device=device)
        self._g_pos = torch.zeros(1, dtype=torch.long, device=device)
        self._g_emb = torch.zeros(1, 1, core.config.hidden_size, dtype=dtype, device=device)
        self._core = core

        p0 = 8
        dummy = torch.zeros(1, p0, core.config.hidden_size, dtype=dtype, device=device)
        self._g_mask.zero_()
        self._g_mask[:, :p0] = 1
        with torch.inference_mode():
            core(
                inputs_embeds=dummy,
                attention_mask=self._g_mask,
                past_key_values=self._g_cache,
                cache_position=torch.arange(p0, device=device),
                use_cache=True,
                return_dict=True,
            )
            self._g_pos.fill_(p0)
            self._g_mask[:, p0] = 1
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    out = core(
                        inputs_embeds=self._g_emb,
                        attention_mask=self._g_mask,
                        past_key_values=self._g_cache,
                        cache_position=self._g_pos,
                        use_cache=True,
                        return_dict=True,
                    )
                    self._head(out.last_hidden_state[:, -1])
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            self._g_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._g_graph):
                out = core(
                    inputs_embeds=self._g_emb,
                    attention_mask=self._g_mask,
                    past_key_values=self._g_cache,
                    cache_position=self._g_pos,
                    use_cache=True,
                    return_dict=True,
                )
                self._g_logits = self._head(out.last_hidden_state[:, -1])
        torch.cuda.synchronize()

        self._orig_wrapper = lm.inference_wrapper
        self._lm = lm
        self._g_n = 0
        self._max_len = max_len

        tts_self = self

        def wrapper(lm_self, lm_input, sampling, min_len, max_len_, uuid):
            try:
                yield from tts_self._graph_generate(lm_self, lm_input, sampling, min_len, max_len_)
            except Exception as exc:
                if tts_self.cfg.verbose:
                    print(f"[TTS] graph 解码失败，回退: {exc!r}", flush=True)
                yield from tts_self._orig_wrapper(lm_input, sampling, min_len, max_len_, uuid)

        lm.inference_wrapper = types.MethodType(wrapper, lm)
        self._graph_ok = True
        print("[TTS] 语音 token 解码已启用 CUDA graph", flush=True)

    def _graph_generate(self, lm, lm_input, sampling, min_len, max_len):
        torch = self.torch
        p = lm_input.shape[1]
        if p + max_len + 2 >= self._max_len:
            raise RuntimeError("prompt too long for static cache")

        lm_input = lm_input.to(next(self._core.parameters()).dtype)
        self._g_mask.zero_()
        self._g_mask[:, :p] = 1
        with torch.inference_mode():
            out = self._core(
                inputs_embeds=lm_input,
                attention_mask=self._g_mask,
                past_key_values=self._g_cache,
                cache_position=torch.arange(p, device=self._g_mask.device),
                use_cache=True,
                return_dict=True,
            )
            logits = self._head(out.last_hidden_state[:, -1])
            self._g_n = p

            sampled: list[int] = []
            for i in range(max_len):
                top_ids = lm.sampling_ids(
                    logits.squeeze(0), sampled, sampling, ignore_eos=i < min_len
                )
                if top_ids in lm.stop_token_ids:
                    break
                yield top_ids
                sampled.append(top_ids)
                emb = lm.speech_embedding.weight[top_ids].reshape(1, 1, -1)
                self._g_emb.copy_(emb.detach())
                self._g_pos.fill_(self._g_n)
                self._g_mask[:, self._g_n] = 1
                self._g_graph.replay()
                logits = self._g_logits
                self._g_n += 1

    # --------------------------------------------------------------- warmup
    def _warmup(self):
        print("[TTS] 预热 ...", flush=True)
        try:
            for _ in self.synthesize_stream("你好。"):
                pass
        except Exception as exc:
            print(f"[TTS] 预热失败: {exc!r}", flush=True)

    # ------------------------------------------------------------- synthesis
    def synthesize_stream(self, text: str, cancel: threading.Event | None = None):
        """Yield mono float32 24 kHz chunks. On cancel, drains silently."""
        text = (text or "").strip()
        if not text:
            return
        with self.lock:
            model = self.tts.model
            model.token_hop_len = self._hop
            try:
                for out in self.tts.inference_zero_shot(
                    text,
                    "",
                    "",
                    zero_shot_spk_id=self.spk_id,
                    stream=True,
                    text_frontend=False,
                ):
                    if cancel is not None and cancel.is_set():
                        continue
                    chunk = out["tts_speech"]
                    yield chunk.squeeze(0).float().cpu().numpy()
            finally:
                model.token_hop_len = self._hop

    def close(self):
        try:
            del self.tts
        except Exception:
            pass
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
