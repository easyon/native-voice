"""CUDA-graph decoder for the CosyVoice speech-token LLM (Qwen2 0.5B).

Upstream decodes one token at a time through the full Qwen2ForCausalLM
(including the 152k text vocabulary head) with fresh Python-side cache
objects, reaching ~55 tok/s on an RTX 4070 Laptop. This module keeps the
decode step in a StaticCache and replays a captured graph (~110 tok/s).
"""
from __future__ import annotations

import logging
import types

log = logging.getLogger(__name__)


class GraphDecoder:
    def __init__(self, lm, max_cache_len: int = 2048, verbose: bool = False):
        self.lm = lm
        self.max_len = int(max_cache_len)
        self.verbose = verbose
        self._n = 0
        self._ok = False
        self._setup()

    def _setup(self) -> None:
        import torch
        from transformers import StaticCache

        core = self.lm.llm.model.model           # Qwen2Model, no text lm_head
        head = self.lm.llm_decoder
        device = next(core.parameters()).device
        dtype = next(core.parameters()).dtype
        self._torch = torch
        self._core = core
        self._head = head
        self._cache = StaticCache(
            config=core.config,
            max_batch_size=1,
            max_cache_len=self.max_len,
            device=device,
            dtype=dtype,
        )
        self._mask = torch.zeros(1, self.max_len, dtype=torch.long, device=device)
        self._pos = torch.zeros(1, dtype=torch.long, device=device)
        self._emb = torch.zeros(1, 1, core.config.hidden_size, dtype=dtype, device=device)

        p0 = 8
        dummy = torch.zeros(1, p0, core.config.hidden_size, dtype=dtype, device=device)
        self._mask.zero_()
        self._mask[:, :p0] = 1
        with torch.inference_mode():
            core(
                inputs_embeds=dummy,
                attention_mask=self._mask,
                past_key_values=self._cache,
                cache_position=torch.arange(p0, device=device),
                use_cache=True,
                return_dict=True,
            )
            self._pos.fill_(p0)
            self._mask[:, p0] = 1
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    out = core(
                        inputs_embeds=self._emb,
                        attention_mask=self._mask,
                        past_key_values=self._cache,
                        cache_position=self._pos,
                        use_cache=True,
                        return_dict=True,
                    )
                    head(out.last_hidden_state[:, -1])
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                out = core(
                    inputs_embeds=self._emb,
                    attention_mask=self._mask,
                    past_key_values=self._cache,
                    cache_position=self._pos,
                    use_cache=True,
                    return_dict=True,
                )
                self._logits = head(out.last_hidden_state[:, -1])
        torch.cuda.synchronize()

        self._orig = self.lm.inference_wrapper
        decoder = self

        def wrapper(lm_self, lm_input, sampling, min_len, max_len, uuid):
            try:
                yield from decoder.generate(lm_self, lm_input, sampling, min_len, max_len)
            except Exception as exc:  # pragma: no cover - fallback path
                if decoder.verbose:
                    log.warning("graph decode failed, falling back: %r", exc)
                yield from decoder._orig(lm_input, sampling, min_len, max_len, uuid)

        self.lm.inference_wrapper = types.MethodType(wrapper, self.lm)
        self._ok = True
        log.info("TTS speech-token decoder uses CUDA graph")

    @property
    def ok(self) -> bool:
        return self._ok

    def generate(self, lm, lm_input, sampling, min_len, max_len):
        torch = self._torch
        p = lm_input.shape[1]
        if p + max_len + 2 >= self.max_len:
            raise RuntimeError("prompt too long for static cache")
        lm_input = lm_input.to(next(self._core.parameters()).dtype)
        self._mask.zero_()
        self._mask[:, :p] = 1
        with torch.inference_mode():
            out = self._core(
                inputs_embeds=lm_input,
                attention_mask=self._mask,
                past_key_values=self._cache,
                cache_position=torch.arange(p, device=self._mask.device),
                use_cache=True,
                return_dict=True,
            )
            logits = self._head(out.last_hidden_state[:, -1])
            self._n = p
            sampled: list[int] = []
            for i in range(max_len):
                top_ids = lm.sampling_ids(logits.squeeze(0), sampled, sampling, ignore_eos=i < min_len)
                if top_ids in lm.stop_token_ids:
                    break
                yield top_ids
                sampled.append(top_ids)
                emb = lm.speech_embedding.weight[top_ids].reshape(1, 1, -1)
                self._emb.copy_(emb.detach())
                self._pos.fill_(self._n)
                self._mask[:, self._n] = 1
                self._graph.replay()
                logits = self._logits
                self._n += 1
