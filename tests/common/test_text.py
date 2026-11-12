from __future__ import annotations

import pytest
from cosyvoice_common.audio import validate_format, validate_sample_rate
from cosyvoice_common.errors import APIError
from cosyvoice_common.text import (
    SentenceBuffer,
    apply_hot_fix,
    preprocess_text,
    split_sentences,
)


class Options:
    enable_markdown_filter = True
    enable_ssml = True
    hot_fix = None


def test_split_sentences_keeps_punctuation() -> None:
    assert split_sentences("你好。今天好吗？好的！") == [
        "你好。",
        "今天好吗？",
        "好的！",
    ]


def test_split_sentences_splits_long_text() -> None:
    text = "甲" * 50 + "，" + "乙" * 50
    parts = split_sentences(text, max_chars=40)
    assert all(len(part) <= 41 for part in parts)
    assert "".join(parts) == text


def test_sentence_buffer_waits_for_delimiter() -> None:
    buffer = SentenceBuffer()
    assert buffer.feed("你好") == []
    assert buffer.feed("，今天好吗？") == ["你好，今天好吗？"]
    assert buffer.pending == ""
    assert buffer.feed("还没结束") == []
    assert buffer.flush() == ["还没结束"]


def test_preprocess_markdown_and_hotfix() -> None:
    class WithHotfix(Options):
        hot_fix = type(
            "HotFix",
            (),
            {"replace": [{"Cosy Voice": "CosyVoice"}]},
        )()

    text = preprocess_text("**Cosy Voice** 很好用。", WithHotfix())
    assert text == "CosyVoice 很好用。"


def test_preprocess_ssml_break_and_tag_validation() -> None:
    options = Options()
    options.enable_markdown_filter = False
    assert preprocess_text('你好<break time="300ms"/>世界', options) == "你好。世界"

    with pytest.raises(APIError) as exc:
        preprocess_text('<script>alert(1)</script>', options)
    assert exc.value.code == "UnsupportedParameter"


def test_audio_validation_errors_are_api_errors() -> None:
    with pytest.raises(APIError) as exc:
        validate_format("aac")
    assert exc.value.code == "UnsupportedFormat"

    with pytest.raises(APIError) as exc:
        validate_sample_rate(12345)
    assert exc.value.code == "UnsupportedParameter"


def test_apply_hot_fix_ignores_malformed_groups() -> None:
    class HotFix:
        replace = [{"旧": "新"}, "not-a-dict", {"": "ignored"}]

    assert apply_hot_fix("旧文本", HotFix()) == "新文本"
