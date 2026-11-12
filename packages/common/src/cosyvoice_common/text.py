from __future__ import annotations

import re
from typing import Any, Protocol

from .errors import invalid_parameter, unsupported_parameter


class TextOptions(Protocol):
    """The text-front-end options shared by runtime and offline pipelines.

    ``runtime.schemas.SpeechInput`` satisfies this protocol structurally, so
    ``common`` does not depend on the runtime package.
    """

    enable_markdown_filter: bool
    enable_ssml: bool
    hot_fix: Any


_SENTENCE_END = re.compile(r"[。！？!?；;\n]")
_LONG_BREAK = re.compile(r"<break\s+time=\"(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s)\"\s*/?>", re.I)
_TAG = re.compile(r"</?([a-zA-Z][\w:-]*)(?:\s[^>]*)?>")
_ALLOWED_SSML = {"speak", "break", "prosody", "emphasis", "say-as", "lang"}


def strip_markdown(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.M)
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.M)
    text = text.replace("**", "").replace("__", "")
    text = text.replace("*", "").replace("_", "")
    return text


def apply_hot_fix(text: str, hot_fix: object | None) -> str:
    if hot_fix is None:
        return text
    replace = getattr(hot_fix, "replace", None) or []
    for group in replace:
        if not isinstance(group, dict):
            continue
        for old, new in group.items():
            if old:
                text = text.replace(str(old), str(new))
    return text


def sanitize_ssml(text: str, strict: bool = True) -> str:
    if strict:
        for match in _TAG.finditer(text):
            if match.group(1).lower() not in _ALLOWED_SSML:
                raise unsupported_parameter(
                    f"unsupported SSML tag: <{match.group(1)}>"
                )

    def _break(match: re.Match[str]) -> str:
        value = float(match.group("value"))
        unit = match.group("unit").lower()
        pause_ms = value * 1000 if unit == "s" else value
        return "。" if pause_ms >= 250 else "，"

    text = _LONG_BREAK.sub(_break, text)
    text = _TAG.sub("", text)
    return text


def preprocess_text(text: str, options: TextOptions, *, ssml_strict: bool = True) -> str:
    if options.enable_markdown_filter:
        text = strip_markdown(text)
    text = apply_hot_fix(text, options.hot_fix)
    if options.enable_ssml:
        text = sanitize_ssml(text, strict=ssml_strict)
    text = text.strip()
    if not text:
        raise invalid_parameter("text must not be empty")
    return text


def count_characters(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def split_sentences(text: str, max_chars: int = 200) -> list[str]:
    """Split text after sentence-ending punctuation, then split long pieces."""
    # re.split loses delimiters, so scan and keep each delimiter.
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        start = end
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    if not parts:
        parts = [text.strip()] if text.strip() else []

    result: list[str] = []
    for part in parts:
        while len(part) > max_chars:
            cut = max(
                part.rfind("，", 0, max_chars),
                part.rfind(",", 0, max_chars),
                part.rfind("、", 0, max_chars),
                part.rfind(" ", 0, max_chars),
            )
            if cut <= 0:
                cut = max_chars
            head = part[: cut + 1].strip()
            if head:
                result.append(head)
            part = part[cut + 1 :].strip()
        if part:
            result.append(part)
    return result


class SentenceBuffer:
    """Incrementally extracts complete sentences from continue-task text."""

    def __init__(self, max_chars: int = 200) -> None:
        self._buffer = ""
        self._max_chars = max_chars

    @property
    def pending(self) -> str:
        return self._buffer

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        out: list[str] = []
        start = 0
        for match in _SENTENCE_END.finditer(self._buffer):
            end = match.end()
            piece = self._buffer[start:end].strip()
            if piece:
                out.append(piece)
            start = end
        self._buffer = self._buffer[start:]

        while len(self._buffer) > self._max_chars:
            cut = max(
                self._buffer.rfind("，", 0, self._max_chars),
                self._buffer.rfind(",", 0, self._max_chars),
                self._buffer.rfind("、", 0, self._max_chars),
                self._buffer.rfind(" ", 0, self._max_chars),
            )
            if cut <= 0:
                cut = self._max_chars
            piece = self._buffer[: cut + 1].strip()
            if piece:
                out.append(piece)
            self._buffer = self._buffer[cut + 1 :]
        return out

    def flush(self) -> list[str]:
        piece = self._buffer.strip()
        self._buffer = ""
        return [piece] if piece else []


def approx_words(text: str, duration_ms: int) -> list[dict[str, int | str]]:
    """Approximate character timestamps; replace with forced alignment later."""
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return []
    per = max(1, duration_ms // len(chars))
    words: list[dict[str, int | str]] = []
    for i, ch in enumerate(chars):
        begin = i * per
        words.append(
            {
                "text": ch,
                "begin_index": i,
                "end_index": i + 1,
                "begin_time": begin,
                "end_time": min(duration_ms, begin + per),
            }
        )
    return words
