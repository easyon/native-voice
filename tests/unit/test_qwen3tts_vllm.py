from __future__ import annotations

import json

import numpy as np
import pytest
import websockets.sync.client

from native_voice.models.tts.qwen3tts_vllm import Qwen3TTSVLLM, Qwen3TTSVLLMConfig


def test_container_path_maps_host_root(tmp_path):
    ref = tmp_path / "voices" / "ref.wav"
    ref.parent.mkdir()
    ref.write_bytes(b"RIFF")
    model = Qwen3TTSVLLM(
        Qwen3TTSVLLMConfig(root=tmp_path, container_root="/workspace")
    )
    assert model._container_path(ref) == "/workspace/voices/ref.wav"
    assert model._container_path(ref.relative_to(tmp_path)) == "/workspace/voices/ref.wav"


def test_session_config_uses_base_task_and_reference(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    model = Qwen3TTSVLLM(
        Qwen3TTSVLLMConfig(root=tmp_path, container_root="/workspace")
    )
    model._add_voice("default", "参考文本", ref, "默认音色")
    cfg = model._session_config("default")
    assert cfg["task_type"] == "Base"
    assert cfg["stream_audio"] is True
    assert cfg["response_format"] == "pcm"
    assert cfg["ref_audio"] == "file:///workspace/ref.wav"
    assert cfg["ref_text"] == "参考文本"
    assert cfg["x_vector_only_mode"] is False


def test_cosyvoice3_session_omits_qwen_only_fields(tmp_path):
    from native_voice.models.tts.cosyvoice3_vllm import CosyVoice3VLLM, CosyVoice3VLLMConfig

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    model = CosyVoice3VLLM(
        CosyVoice3VLLMConfig(root=tmp_path, container_root="/workspace")
    )
    model._add_voice("default", "参考文本", ref, "默认音色")
    cfg = model._session_config("default")

    assert "task_type" not in cfg
    assert "language" not in cfg
    assert "x_vector_only_mode" not in cfg
    assert cfg["stream_audio"] is True
    assert cfg["response_format"] == "pcm"
    assert cfg["ref_audio"] == "file:///workspace/ref.wav"
    assert cfg["ref_text"] == "参考文本"


def test_pcm16_bytes_are_converted_to_float32():
    model = Qwen3TTSVLLM(Qwen3TTSVLLMConfig())
    pcm = np.array([0, 16384, -32768], dtype="<i2").tobytes()
    chunks = list(model._pcm16(pcm))
    assert len(chunks) == 1
    assert chunks[0].dtype == np.float32
    assert chunks[0][0] == 0.0
    assert chunks[0][1] == 0.5
    assert chunks[0][2] == -1.0


def test_load_warms_up_before_becoming_ready(tmp_path, monkeypatch):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    calls = []
    model = Qwen3TTSVLLM(
        Qwen3TTSVLLMConfig(
            root=tmp_path,
            container_root="/workspace",
            prompt_wav=ref,
            prompt_text="参考文本",
            warmup=True,
            warmup_text="预热",
        )
    )
    monkeypatch.setattr(model, "_wait_for_server", lambda: None)
    monkeypatch.setattr(
        model,
        "iter_synthesize",
        lambda text, voice, cancel=None: calls.append((text, voice)) or iter(()),
    )

    model.load()

    assert model.ready
    assert calls == [("预热", "default")]


def test_reuses_one_websocket_across_clauses(tmp_path, monkeypatch):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    pcm = np.array([100, -100], dtype="<i2").tobytes()
    session = [
        pcm,
        json.dumps({"type": "audio.done", "error": False}),
        json.dumps({"type": "session.done"}),
    ]
    frames = session * 2
    connects = []

    class FakeWS:
        close_code = None

        def send(self, data):
            return None

        def recv(self, timeout=None):
            return frames.pop(0)

        def close(self):
            return None

    monkeypatch.setattr(
        websockets.sync.client,
        "connect",
        lambda *a, **kw: connects.append(1) or FakeWS(),
    )

    model = Qwen3TTSVLLM(
        Qwen3TTSVLLMConfig(root=tmp_path, container_root="/workspace")
    )
    model._add_voice("default", "参考文本", ref)

    assert len(list(model.iter_synthesize("第一句。", "default"))) == 1
    assert len(list(model.iter_synthesize("第二句。", "default"))) == 1
    assert len(connects) == 1


def test_reconnects_after_idle_or_closed_socket(tmp_path, monkeypatch):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    pcm = np.array([100, -100], dtype="<i2").tobytes()
    session = [
        pcm,
        json.dumps({"type": "audio.done", "error": False}),
        json.dumps({"type": "session.done"}),
    ]
    frames = list(session)
    connects = []

    class FakeWS:
        close_code = None

        def send(self, data):
            return None

        def recv(self, timeout=None):
            return frames.pop(0)

        def close(self):
            return None

    monkeypatch.setattr(
        websockets.sync.client,
        "connect",
        lambda *a, **kw: connects.append(1) or FakeWS(),
    )
    model = Qwen3TTSVLLM(
        Qwen3TTSVLLMConfig(root=tmp_path, container_root="/workspace")
    )
    model._add_voice("default", "参考文本", ref)

    assert len(list(model.iter_synthesize("第一句。", "default"))) == 1
    frames.extend(session)
    model._ws_last_used -= model.cfg.idle_reconnect_s + 1
    assert len(list(model.iter_synthesize("第二句。", "default"))) == 1
    assert len(connects) == 2

    frames.extend(session)
    model._ws.close_code = 1000
    assert len(list(model.iter_synthesize("第三句。", "default"))) == 1
    assert len(connects) == 3


def test_streaming_round_trip_parses_binary_and_json(tmp_path, monkeypatch):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    pcm = np.array([100, -100], dtype="<i2").tobytes()
    frames = [
        json.dumps({"type": "audio.start", "sentence_text": "你好"}),
        pcm,
        json.dumps({"type": "audio.done", "error": False}),
        json.dumps({"type": "session.done"}),
    ]

    class FakeWS:
        def __init__(self):
            self.sent = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def send(self, data):
            self.sent.append(json.loads(data))

        def recv(self, timeout=None):
            if not frames:
                raise TimeoutError
            return frames.pop(0)

    fake = FakeWS()
    import websockets.sync.client

    monkeypatch.setattr(websockets.sync.client, "connect", lambda *a, **kw: fake)

    model = Qwen3TTSVLLM(
        Qwen3TTSVLLMConfig(root=tmp_path, container_root="/workspace")
    )
    model._add_voice("default", "参考文本", ref)
    chunks = list(model.iter_synthesize("你好。", "default"))

    assert len(chunks) == 1
    assert chunks[0].tolist() == [pytest.approx(100 / 32768), pytest.approx(-100 / 32768)]
    assert fake.sent[0]["type"] == "session.config"
    assert fake.sent[0]["task_type"] == "Base"
    assert fake.sent[1] == {"type": "input.text", "text": "你好。"}
    assert fake.sent[2] == {"type": "input.done"}
