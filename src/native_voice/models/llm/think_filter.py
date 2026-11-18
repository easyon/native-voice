"""Strip Qwen3 reasoning blocks that may be split across SSE deltas.

The tags are assembled from parts so the literal marker never appears in this
source file; some tooling rewrites it on the way through.
"""
from __future__ import annotations

_TAG_OPEN = "<" + "think" + ">"
_TAG_CLOSE = "<" + "/" + "think" + ">"


def _suffix_prefix_len(text: str, token: str) -> int:
    for n in range(min(len(text), len(token) - 1), 0, -1):
        if text.endswith(token[:n]):
            return n
    return 0


class ThinkFilter:
    def __init__(self) -> None:
        self.pending = ""
        self.inside = False

    def feed(self, text: str) -> str:
        self.pending += text
        out: list[str] = []
        while True:
            if self.inside:
                idx = self.pending.find(_TAG_CLOSE)
                if idx >= 0:
                    self.pending = self.pending[idx + len(_TAG_CLOSE):]
                    self.inside = False
                    continue
                keep = _suffix_prefix_len(self.pending, _TAG_CLOSE)
                self.pending = self.pending[-keep:] if keep else ""
                break

            idx = self.pending.find(_TAG_OPEN)
            if idx >= 0:
                out.append(self.pending[:idx])
                self.pending = self.pending[idx + len(_TAG_OPEN):]
                self.inside = True
                continue

            # llama.cpp sometimes drops an unclosed opening tag while streaming
            # with thinking disabled.  A bare closing tag means everything
            # before it was reasoning and must not be spoken.
            idx = self.pending.find(_TAG_CLOSE)
            if idx >= 0:
                self.pending = self.pending[idx + len(_TAG_CLOSE):]
                continue

            keep = max(
                _suffix_prefix_len(self.pending, _TAG_OPEN),
                _suffix_prefix_len(self.pending, _TAG_CLOSE),
            )
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
        if self.inside:
            return ""
        # Drop a dangling partial tag rather than speaking half a marker.
        if text in (_TAG_OPEN[: len(text)], _TAG_CLOSE[: len(text)]):
            return ""
        return text
