"""Strip Qwen3 think blocks that may be split across SSE deltas."""
from __future__ import annotations


def _suffix_prefix_len(text: str, token: str) -> int:
    for n in range(min(len(text), len(token) - 1), 0, -1):
        if text.endswith(token[:n]):
            return n
    return 0


class ThinkFilter:
    def __init__(self):
        self.pending = ""
        self.inside = False

    def feed(self, text: str) -> str:
        self.pending += text
        out: list[str] = []
        while True:
            if self.inside:
                end_tag = " response"
                idx = self.pending.find(end_tag)
                if idx >= 0:
                    self.pending = self.pending[idx + len(end_tag):]
                    self.inside = False
                    continue
                keep = _suffix_prefix_len(self.pending, end_tag)
                self.pending = self.pending[-keep:] if keep else ""
                break
            idx = self.pending.find(" thinking")
            if idx >= 0:
                out.append(self.pending[:idx])
                self.pending = self.pending[idx + 7:]
                self.inside = True
                continue
            keep = _suffix_prefix_len(self.pending, " thinking")
            if keep:
                out.append(self.pending[:-keep])
                self.pending = self.pending[-keep:]
            else:
                out.append(self.pending)
                self.pending = ""
            break
        return "".join(out)

    def flush(self) -> str:
        text, self.pending = self.pending, ""
        return "" if self.inside else text
