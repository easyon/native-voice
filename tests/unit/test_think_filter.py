from __future__ import annotations

from native_voice.models.llm.think_filter import ThinkFilter

# Assemble the markers so this file never contains the literal tag.
OPEN = "<" + "think" + ">"
CLOSE = "<" + "/" + "think" + ">"


def test_strips_reasoning_split_across_deltas():
    filt = ThinkFilter()
    assert filt.feed(OPEN + "让我想") == ""
    assert filt.feed("一想" + CLOSE + "现在") == "现在"
    assert filt.feed("是三点。") == "是三点。"
    assert filt.flush() == ""


def test_strips_stray_closing_tag():
    # llama.cpp can drop the opening tag while streaming; the reasoning text
    # before the bare closing tag must not reach TTS.
    filt = ThinkFilter()
    assert filt.feed("用户问的是时间。" + CLOSE + "现在是三点。") == "现在是三点。"


def test_plain_text_passes_through():
    filt = ThinkFilter()
    assert filt.feed("你好，") == "你好，"
    assert filt.feed("有什么可以帮你？") == "有什么可以帮你？"
    assert filt.flush() == ""


def test_partial_tag_is_held_until_complete():
    filt = ThinkFilter()
    assert filt.feed("<thi") == ""
    assert filt.feed("nk>内部") == ""
    assert filt.feed(CLOSE) == ""
    assert filt.feed("答案") == "答案"


def test_unclosed_reasoning_is_dropped_on_flush():
    filt = ThinkFilter()
    assert filt.feed(OPEN + "还在想") == ""
    assert filt.flush() == ""
