from __future__ import annotations

import time

import pytest

from native_voice.backends.base import Backends
from native_voice.backends.fake import FakeASR, FakeTTS
from native_voice.conversation.ack import is_backchannel_text
from native_voice.conversation.engine import ConversationEngine, Turn
from native_voice.conversation.vad import VadEvent


class CollectingWriter:
    def __init__(self):
        self.items = []

    async def send_json(self, obj):
        self.items.append(obj)

    async def send_bytes(self, data):
        self.items.append(data)


def make_engine(cfg):
    writer = CollectingWriter()
    engine = ConversationEngine(cfg, Backends(FakeTTS(), FakeASR()), None, writer)
    engine.vad.active = True
    return engine, writer


def make_turn(t_tts: float = 0.0) -> Turn:
    now = time.monotonic()
    turn = Turn("turn-1", "上一轮问题", 0.0, now, now)
    turn.t_tts = t_tts
    return turn


@pytest.mark.parametrize(
    "text",
    ["嗯", "嗯嗯。", "啊？", "对啊", "好的", "OK", "嗯哼，对对", "我在听", "   "],
)
def test_acknowledgements_are_not_content(text):
    assert is_backchannel_text(text)


@pytest.mark.parametrize(
    "text",
    ["等一下", "嗯，我想问一下", "不对", "你好", "好的，谢谢", "stop", "继续说，为什么"],
)
def test_real_content_is_not_an_acknowledgement(text):
    assert not is_backchannel_text(text)


@pytest.mark.asyncio
async def test_filler_partial_does_not_interrupt(cfg):
    engine, writer = make_engine(cfg)
    engine.turn = make_turn(t_tts=time.monotonic())

    await engine._maybe_barge_in("嗯嗯，对。")

    assert engine.turn is not None
    assert not any(i.get("type") == "assistant.interrupted" for i in writer.items)


@pytest.mark.asyncio
async def test_content_partial_interrupts_immediately(cfg):
    engine, writer = make_engine(cfg)
    engine.turn = make_turn(t_tts=time.monotonic())

    await engine._maybe_barge_in("等等，我想问")

    assert engine.turn is None
    interrupted = [i for i in writer.items if i.get("type") == "assistant.interrupted"]
    assert interrupted and interrupted[0]["reason"] == "barge_in"


@pytest.mark.asyncio
async def test_quick_followup_with_content_merges_before_audio(cfg):
    engine, writer = make_engine(cfg)
    engine.turn = make_turn(t_tts=0.0)

    await engine._maybe_barge_in("还有一点")

    assert engine.turn is None
    assert engine.merge_pending
    assert engine.merge_text == "上一轮问题"
    assert any(i.get("type") == "assistant.cancelled" for i in writer.items)


@pytest.mark.asyncio
async def test_filler_final_is_ignored_while_assistant_speaks(cfg, monkeypatch):
    engine, writer = make_engine(cfg)
    engine.turn = make_turn(t_tts=time.monotonic())
    engine._user_over_turn = True
    started = []

    async def finalize():
        return "嗯，好的"

    async def start_turn(*args):
        started.append(args)

    monkeypatch.setattr(engine.asr, "finalize", finalize)
    monkeypatch.setattr(engine, "_start_turn", start_turn)

    await engine._on_end(VadEvent("end", duration=1.0))

    assert engine.turn is not None
    assert started == []
    assert not any(i.get("type") == "asr.final" for i in writer.items)


@pytest.mark.asyncio
async def test_meaningful_final_interrupts_if_no_partial(cfg, monkeypatch):
    engine, writer = make_engine(cfg)
    engine.turn = make_turn(t_tts=time.monotonic())
    engine._user_over_turn = True
    started = []

    async def finalize():
        return "等一下，我想问个问题"

    async def start_turn(*args):
        started.append(args)

    monkeypatch.setattr(engine.asr, "finalize", finalize)
    monkeypatch.setattr(engine, "_start_turn", start_turn)

    await engine._on_end(VadEvent("end", duration=1.0))

    assert engine.turn is None
    assert started
    assert any(i.get("type") == "assistant.interrupted" for i in writer.items)


@pytest.mark.asyncio
async def test_partial_is_emitted_while_assistant_turn_active(cfg):
    engine, writer = make_engine(cfg)
    engine.turn = make_turn(t_tts=time.monotonic())
    engine.begin_ms = 1200

    await engine._emit_partial("你好")

    partials = [i for i in writer.items if i.get("type") == "asr.partial"]
    assert partials and partials[0]["text"] == "你好"


@pytest.mark.asyncio
async def test_sustained_speech_interrupts_without_partial(cfg):
    engine, writer = make_engine(cfg)
    engine.turn = make_turn(t_tts=time.monotonic())
    engine.vad.active = True
    engine.stream_ms = 5000
    engine._speech_started_ms = (
        engine.stream_ms - cfg.conversation.barge_content_wait_ms - 1
    )

    await engine._maybe_barge_in("")

    assert engine.turn is None
    assert any(i.get("type") == "assistant.interrupted" for i in writer.items)


@pytest.mark.asyncio
async def test_short_speech_without_partial_does_not_interrupt(cfg):
    engine, writer = make_engine(cfg)
    engine.turn = make_turn(t_tts=time.monotonic())
    engine.vad.active = True
    engine.stream_ms = 5000
    engine._speech_started_ms = engine.stream_ms - 300

    await engine._maybe_barge_in("")

    assert engine.turn is not None
    assert not any(i.get("type") == "assistant.interrupted" for i in writer.items)
