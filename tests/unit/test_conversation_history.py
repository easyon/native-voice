from __future__ import annotations

from native_voice.backends.base import Backends
from native_voice.backends.fake import FakeASR, FakeTTS
from native_voice.conversation.engine import ConversationEngine, Turn


class DummyWriter:
    async def send_json(self, obj):
        return None

    async def send_bytes(self, data):
        return None


def test_partial_assistant_saved_when_barged_in(cfg):
    engine = ConversationEngine(cfg, Backends(FakeTTS(), FakeASR()), None, DummyWriter())
    engine.history.append({"role": "user", "content": "我想听故事"})
    turn = Turn(turn_id="t1", user_text="我想听故事", t_ref=0, t_endpoint=0, t_text=0)
    turn.text = "好的，我来给你讲一个故事。"
    engine._save_assistant_history(turn)
    assert engine.history[-1] == {"role": "assistant", "content": "好的，我来给你讲一个故事。"}

    # A merged fragment was never actually answered; it must not enter history.
    merged = Turn(turn_id="t2", user_text="还有", t_ref=0, t_endpoint=0, t_text=0)
    merged.text = "不应该保存"
    merged.merged = True
    engine._save_assistant_history(merged)
    assert all(m["content"] != "不应该保存" for m in engine.history)
