"""Incremental sentence buffering for streaming TTS text."""
from __future__ import annotations

import re

_SENT_END = "。！？!?；;\n"
_SOFT = "，,、：:"
_CLEAN = re.compile(r"[*#`]|[\U0001F300-\U0001FAFF☀-➿️‍]+")


class TTSSentenceBuffer:
    def __init__(self, soft_limit: int = 18, hard_limit: int = 48):
        self.buf = ""
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", _CLEAN.sub("", text)).strip()

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while True:
            cut = -1
            for i, ch in enumerate(self.buf):
                if ch in _SENT_END:
                    cut = i + 1
                    while cut < len(self.buf) and self.buf[cut] in "”’」』）)\"'":
                        cut += 1
                    break
            if cut < 0 and len(self.buf) >= self.soft_limit:
                soft = [i for i, ch in enumerate(self.buf) if ch in _SOFT]
                if soft:
                    cut = soft[-1] + 1
            if cut < 0 and len(self.buf) >= self.hard_limit:
                cut = len(self.buf)
            if cut < 0:
                break
            piece = self._clean(self.buf[:cut])
            self.buf = self.buf[cut:]
            if piece and re.search(r"[\w一-鿿]", piece):
                out.append(piece)
        return out

    def flush(self) -> list[str]:
        piece = self._clean(self.buf)
        self.buf = ""
        return [piece] if piece and re.search(r"[\w一-鿿]", piece) else []
