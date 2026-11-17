"""Vectorised nucleus sampling.

Upstream CosyVoice samples token-by-token with a Python loop over sorted
probabilities, synchronising the GPU ~25 times per token. This is the same
distribution with one sync per token.
"""
from __future__ import annotations

import functools


def fast_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    import torch

    probs = weighted_scores.softmax(dim=0)
    topv, topi = probs.topk(min(top_k, probs.numel()))
    cum = topv.cumsum(dim=0)
    keep = cum - topv < top_p
    n = max(1, int(keep.sum().item()))
    idx, p = topi[:n], topv[:n]
    top_ids = int(idx[torch.multinomial(p / p.sum(), 1)].item())
    if decoded_tokens:
        recent = torch.tensor(decoded_tokens[-win_size:], device=probs.device)
        if int((recent == top_ids).sum().item()) >= win_size * tau_r:
            weighted_scores[top_ids] = -float("inf")
            top_ids = int(torch.multinomial(weighted_scores.softmax(dim=0), 1).item())
    return top_ids


def patch_sampling(lm) -> None:
    kwargs = getattr(lm.sampling, "keywords", {}) or {}
    lm.sampling = functools.partial(fast_sampling, **kwargs)
