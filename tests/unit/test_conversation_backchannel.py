from __future__ import annotations

import numpy as np
import pytest

from native_voice.audio.pcm import pcm16_to_float32
from native_voice.backends.base import Backends
from native_voice.backends.fake import FakeASR, FakeTTS
from native_voice.conversation.engine import ConversationEngine


class CollectingWriter:
    def __init__(self):
        self.items = []

    async def send_json(self, obj):
        self.items.append(obj)

    async def send_bytes(self, data):
        self.items.append(data)


@pytest.mark.asyncio
async def test_backchannel_emits_quiet_pcm_while_user_speaks(cfg):
    cfg.conversation.backchannel_enabled = True
    cfg.conversation.backchannel_min_speech_s = 0.1
    cfg.conversation.backchannel_interval_s = 0.0
    cfg.conversation.backchannel_volume = 0.4
    writer = CollectingWriter()
    engine = ConversationEngine(
        cfg, Backends(FakeTTS(), FakeASR()), None, writer
    )
    engine.vad.active = True
    engine.stream_ms = 1000.0
    engine._bc_started_ms = 900.0

    engine._maybe_backchannel()
    assert engine._bc_task is not None
    await engine._bc_task

    kinds = [item.get("type") for item in writer.items if isinstance(item, dict)]
    assert kinds == ["assistant.backchannel.start", "assistant.backchannel.done"]
    start = writer.items[0]
    assert start["volume"] == pytest.approx(0.4)
    pcm = next(item for item in writer.items if isinstance(item, bytes))
    audio = pcm16_to_float32(pcm)
    assert audio.size > 2400
    assert np.max(np.abs(audio)) > 0.01


@pytest.mark.asyncio
async def test_backchannel_waits_for_minimum_speech(cfg):
    cfg.conversation.backchannel_enabled = True
    cfg.conversation.backchannel_min_speech_s = 2.0
    writer = CollectingWriter()
    engine = ConversationEngine(
        cfg, Backends(FakeTTS(), FakeASR()), None, writer
    )
    engine.vad.active = True
    engine.stream_ms = 500.0
    engine._bc_started_ms = 1.0

    engine._maybe_backchannel()

    assert engine._bc_task is None
    assert writer.items == []
