"""Recognise short acknowledgements that must not barge in.

While the assistant is talking, a user's ``嗯``/``对``/``好的`` is usually just a
backchannel, not a request to stop.  ASR may transcribe it as a normal partial,
so the conversation engine has to filter it before cancelling the turn.
"""
from __future__ import annotations

import re

# Punctuation and fillers are removed from the whole string.  Longer tokens are
# matched first so "嗯嗯" is consumed before "嗯".
_PUNCT = re.compile(r"[\s　，。！？、,.!?;；:：~～…—\-]+")
_FILLERS = {
    "嗯哼",
    "嗯嗯",
    "嗯",
    "恩",
    "呃",
    "额",
    "啊",
    "哦",
    "噢",
    "喔",
    "唉",
    "呀",
    "哈",
    "嘿",
    "对对",
    "对",
    "是的",
    "是",
    "好的",
    "好吧",
    "好",
    "行的",
    "行",
    "可以",
    "我在听",
    "你说",
    "您说",
    "继续说",
    "然后呢",
    "接着",
    "收到",
    "明白",
    "了解",
    "okay",
    "ok",
    "yeah",
    "yep",
    "yup",
    "mhm",
    "mm-hmm",
    "uh-huh",
    "hmm",
    "mm",
    "isee",
}


def _normalize(text: str) -> str:
    return _PUNCT.sub("", (text or "").lower())


def is_backchannel_text(text: str) -> bool:
    """True when the transcript contains only filler/acknowledgement words."""
    residual = _normalize(text)
    if not residual:
        return True
    tokens = sorted(_FILLERS, key=len, reverse=True)
    changed = True
    while changed:
        changed = False
        for token in tokens:
            if token and token in residual:
                residual = residual.replace(token, "")
                changed = True
    return not residual
