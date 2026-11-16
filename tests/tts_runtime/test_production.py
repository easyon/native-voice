from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
from cosyvoice_common.errors import APIError
from fastapi.testclient import TestClient
from tts_runtime.core.limits import TenantLimiter
from tts_runtime.core.tenant import Tenant
from tts_runtime.engines.cosyvoice import CosyVoiceEngine
from tts_runtime.main import build_parser, create_app
from tts_runtime.schemas import SynthesisRequest
from tts_runtime.settings import (
    CosyVoiceEngineConfig,
    ModelConfig,
    RuntimeConfig,
    TenantKeyConfig,
    VoiceConfig,
)

PATH = "/api/v1/services/audio/tts/SpeechSynthesizer"
WS_PATH = "/api-ws/v1/inference"
TASK_ID = "2bf83b9a-baeb-4fda-8d9a-000000000002"


def payload(text: str = "你好。", fmt: str = "wav") -> dict:
    return {
        "model": "my-tts-prod",
        "input": {
            "text": text,
            "voice": "voice-a",
            "format": fmt,
            "sample_rate": 24000,
        },
    }


@pytest.fixture
def config() -> RuntimeConfig:
    config = RuntimeConfig.default()
    config.storage.type = "none"
    return config


def test_metrics_requires_dedicated_token(config: RuntimeConfig) -> None:
    config.auth.enabled = True
    config.auth.api_keys = ["business-key"]
    config.auth.metrics_token = "metrics-key"
    with TestClient(create_app(config)) as client:
        assert client.get("/metrics").status_code == 401
        assert (
            client.get(
                "/metrics",
                headers={"Authorization": "Bearer business-key"},
            ).status_code
            == 401
        )
        response = client.get(
            "/metrics",
            headers={"Authorization": "Bearer metrics-key"},
        )
        assert response.status_code == 200
        assert "cosyvoice_build_info" in response.text


def test_request_body_limit(config: RuntimeConfig) -> None:
    config.server.max_request_bytes = 128
    with TestClient(create_app(config)) as client:
        response = client.post(PATH, json=payload("长" * 500))
        assert response.status_code == 413
        assert response.json()["code"] == "RequestTooLarge"


def test_local_audio_url_is_signed(config: RuntimeConfig, tmp_path: Path) -> None:
    config.storage.type = "local"
    config.storage.dir = str(tmp_path)
    config.storage.base_url = "http://testserver"
    config.storage.signing_secret = "test-secret"
    config.storage.serve_files = True
    with TestClient(create_app(config)) as client:
        response = client.post(PATH, json=payload("你好。"))
        assert response.status_code == 200
        url = response.json()["output"]["audio"]["url"]
        parsed = urlparse(url)
        assert parsed.path.startswith("/files/audio_")
        assert "sig=" in parsed.query

        audio = client.get(parsed.path + "?" + parsed.query)
        assert audio.status_code == 200
        assert audio.content.startswith(b"RIFF")

        assert client.get(parsed.path).status_code == 403
        assert client.get(parsed.path + "?exp=1&sig=bad").status_code == 403


def test_metrics_exposes_latency_histogram(config: RuntimeConfig) -> None:
    with TestClient(create_app(config)) as client:
        assert client.post(PATH, json=payload("你好。")).status_code == 200
        assert (
            client.post(
                PATH,
                json=payload("你好。"),
                headers={"X-DashScope-SSE": "enable"},
            ).status_code
            == 200
        )
        text = client.get("/metrics").text
        assert "cosyvoice_request_duration_seconds_bucket" in text
        assert "cosyvoice_first_audio_seconds_bucket" in text


def test_tenant_limiter_enforces_concurrency() -> None:
    async def scenario() -> None:
        limits = RuntimeConfig.default().limits
        limits.acquire_timeout_s = 0.1
        limiter = TenantLimiter(limits)
        tenant = Tenant("tenant-a", max_concurrent_streams=1)
        first = await limiter.acquire(tenant)
        try:
            with pytest.raises(APIError) as exc:
                await limiter.acquire(tenant)
            assert exc.value.code == "Throttling"
        finally:
            await first.release()
        second = await limiter.acquire(tenant)
        await second.release()

    asyncio.run(scenario())


def test_ws_message_size_limit(config: RuntimeConfig) -> None:
    config.server.max_ws_message_bytes = 512
    with TestClient(create_app(config)) as client:
        with client.websocket_connect(WS_PATH) as websocket:
            websocket.send_json(
                {
                    "header": {
                        "action": "run-task",
                        "task_id": TASK_ID,
                        "streaming": "duplex",
                    },
                    "payload": {
                        "task_group": "audio",
                        "task": "tts",
                        "function": "SpeechSynthesizer",
                        "model": "my-tts-prod",
                        "parameters": {"voice": "voice-a", "format": "pcm"},
                        "input": {},
                    },
                }
            )
            assert websocket.receive_json()["header"]["event"] == "task-started"
            websocket.send_json(
                {
                    "header": {
                        "action": "continue-task",
                        "task_id": TASK_ID,
                    },
                    "payload": {"input": {"text": "长" * 500}},
                }
            )
            event = websocket.receive_json()
            while event["header"]["event"] == "result-generated":
                event = websocket.receive_json()
            assert event["header"]["event"] == "task-failed"
            assert event["header"]["error_code"] == "InvalidParameter"


def test_cosyvoice_engine_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeModel:
        sample_rate = 24000

        def inference_sft(self, text, spk_id, *, instruct_text="", stream=True, speed=1.0):
            yield {"tts_speech": np.zeros(2400, dtype=np.float32)}

    fake_cv = types.ModuleType("cosyvoice.cli.cosyvoice")
    fake_cv.CosyVoice3 = lambda artifact, **kwargs: FakeModel()
    fake_cli = types.ModuleType("cosyvoice.cli")
    fake_cli.cosyvoice = fake_cv
    fake_pkg = types.ModuleType("cosyvoice")
    fake_pkg.cli = fake_cli
    monkeypatch.setitem(sys.modules, "cosyvoice", fake_pkg)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", fake_cv)

    models = {
        "m": ModelConfig(engine="cosyvoice3", artifact=str(tmp_path)),
    }
    engine = CosyVoiceEngine(
        CosyVoiceEngineConfig(preload=False, queue_size=4, cancel_grace_s=0.1),
        models,
    )
    request = SynthesisRequest(
        text="你好。",
        model="m",
        voice="v",
        model_config=models["m"],
        voice_config=VoiceConfig(spk_id="v"),
        format="pcm",
        sample_rate=24000,
        volume=50,
        rate=1.0,
        pitch=1.0,
        bit_rate=None,
        instruction=None,
        seed=0,
        language_hints=[],
        word_timestamp_enabled=False,
        request_id="test-request",
        streaming=True,
    )

    async def collect() -> bytes:
        result = bytearray()
        async for chunk in engine.synthesize(request):
            result.extend(chunk.pcm)
        return bytes(result)

    audio = asyncio.run(collect())
    assert len(audio) == 4800
    assert audio[:2] == b"\x00\x00"


def test_cosyvoice_engine_preload_uses_configured_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """preload must load each artifact with its own engine class.

    A deployment can mix CosyVoice2 and CosyVoice3 artifacts; loading both
    through a hard-coded class name would silently build the wrong model.
    """

    class FakeModel:
        sample_rate = 24000

    loaded: list[tuple[str, str]] = []

    def factory(name: str):
        def build(artifact: str, **kwargs: object) -> FakeModel:
            loaded.append((name, artifact))
            return FakeModel()

        return build

    fake_cv = types.ModuleType("cosyvoice.cli.cosyvoice")
    fake_cv.CosyVoice3 = factory("cosyvoice3")
    fake_cv.CosyVoice2 = factory("cosyvoice2")
    fake_cli = types.ModuleType("cosyvoice.cli")
    fake_cli.cosyvoice = fake_cv
    fake_pkg = types.ModuleType("cosyvoice")
    fake_pkg.cli = fake_cli
    monkeypatch.setitem(sys.modules, "cosyvoice", fake_pkg)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", fake_cv)

    models = {
        "v3": ModelConfig(engine="cosyvoice3", artifact=str(tmp_path)),
        "v2": ModelConfig(engine="cosyvoice2", artifact=str(tmp_path)),
    }
    engine = CosyVoiceEngine(CosyVoiceEngineConfig(preload=True), models)

    async def load() -> bool:
        await engine.startup()
        return await engine.ready()

    assert asyncio.run(load()) is True
    assert sorted(loaded) == [
        ("cosyvoice2", str(tmp_path)),
        ("cosyvoice3", str(tmp_path)),
    ]


def _install_fake_cosyvoice(
    monkeypatch: pytest.MonkeyPatch, model_factory: object
) -> None:
    fake_cv = types.ModuleType("cosyvoice.cli.cosyvoice")
    fake_cv.CosyVoice3 = model_factory
    fake_cv.CosyVoice2 = model_factory
    fake_cli = types.ModuleType("cosyvoice.cli")
    fake_cli.cosyvoice = fake_cv
    fake_pkg = types.ModuleType("cosyvoice")
    fake_pkg.cli = fake_cli
    monkeypatch.setitem(sys.modules, "cosyvoice", fake_pkg)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", fake_cv)


def _zero_shot_request(
    model_cfg: ModelConfig, voice_cfg: VoiceConfig
) -> SynthesisRequest:
    return SynthesisRequest(
        text="你好。",
        model="m",
        voice="v",
        model_config=model_cfg,
        voice_config=voice_cfg,
        format="pcm",
        sample_rate=24000,
        volume=50,
        rate=1.0,
        pitch=1.0,
        bit_rate=None,
        instruction=None,
        seed=0,
        language_hints=[],
        word_timestamp_enabled=False,
        request_id="test-request",
        streaming=True,
    )


@pytest.mark.parametrize("cache", [True, False])
def test_cosyvoice_engine_prompt_feature_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cache: bool
) -> None:
    """Zero-shot voices must not re-extract prompt wav features per request.

    With the cache on, the prompt is registered once at load and every request
    passes its id; with it off, upstream's per-request path is untouched.
    """
    added: list[tuple[str, str, str]] = []
    inferred: list[str] = []

    class FakeFrontend:
        def text_normalize(self, text: str, split: bool = False) -> str:
            return text

    class FakeModel:
        sample_rate = 24000
        frontend = FakeFrontend()

        def add_zero_shot_spk(
            self, prompt_text: str, prompt_wav: str, zero_shot_spk_id: str
        ) -> bool:
            added.append((prompt_text, prompt_wav, zero_shot_spk_id))
            return True

        def inference_zero_shot(
            self,
            tts_text: str,
            prompt_text: str,
            prompt_wav: str,
            zero_shot_spk_id: str = "",
            stream: bool = True,
            speed: float = 1.0,
        ):
            inferred.append(zero_shot_spk_id)
            yield {"tts_speech": np.zeros(2400, dtype=np.float32)}

    _install_fake_cosyvoice(monkeypatch, lambda artifact, **kwargs: FakeModel())

    prompt_wav = tmp_path / "prompt.wav"
    prompt_wav.write_bytes(b"RIFF")
    model_cfg = ModelConfig(engine="cosyvoice3", artifact=str(tmp_path))
    voice_cfg = VoiceConfig(
        mode="zero_shot",
        prompt_wav=str(prompt_wav),
        prompt_text="You are a helpful assistant.<|endofprompt|>你好。",
    )
    engine = CosyVoiceEngine(
        CosyVoiceEngineConfig(
            preload=True,
            cache_prompt_features=cache,
            queue_size=4,
            cancel_grace_s=0.1,
        ),
        {"m": model_cfg},
        {"v": voice_cfg},
    )

    async def run() -> None:
        await engine.startup()
        request = _zero_shot_request(model_cfg, voice_cfg)
        for _ in range(2):
            async for _chunk in engine.synthesize(request):
                pass

    asyncio.run(run())

    if cache:
        assert len(added) == 1
        assert added[0][1] == str(prompt_wav)
        spk_id = added[0][2]
        assert spk_id
        assert inferred == [spk_id, spk_id]
    else:
        assert added == []
        assert inferred == ["", ""]


def test_cli_accepts_config_and_overrides() -> None:
    args = build_parser().parse_args(
        ["--config", "custom.yaml", "--host", "127.0.0.1", "--port", "9001"]
    )
    assert args.config == "custom.yaml"
    assert args.host == "127.0.0.1"
    assert args.port == 9001
    assert build_parser().parse_args([]).config is None


def test_signed_url_does_not_leak_path() -> None:
    # Keep this as a cheap contract test for the filename allow-list.
    from tts_runtime.api.files import _FILENAME

    assert _FILENAME.match("audio_abc.wav")
    assert not _FILENAME.match("../audio_abc.wav")
    assert not _FILENAME.match("audio_abc.sh")


def test_tenant_key_config_can_be_used(config: RuntimeConfig) -> None:
    config.auth.enabled = True
    config.auth.tenants = [
        TenantKeyConfig(api_key="tenant-key", id="tenant-a", max_concurrent_streams=2)
    ]
    with TestClient(create_app(config)) as client:
        response = client.post(
            PATH,
            json=payload("你好。"),
            headers={"Authorization": "Bearer tenant-key"},
        )
        assert response.status_code == 200


def test_streaming_wav_has_a_single_header(config: RuntimeConfig) -> None:
    with TestClient(create_app(config)) as client:
        with client.websocket_connect(WS_PATH) as websocket:
            websocket.send_json(
                {
                    "header": {
                        "action": "run-task",
                        "task_id": TASK_ID,
                        "streaming": "duplex",
                    },
                    "payload": {
                        "task_group": "audio",
                        "task": "tts",
                        "function": "SpeechSynthesizer",
                        "model": "my-tts-prod",
                        "parameters": {"voice": "voice-a", "format": "wav"},
                        "input": {},
                    },
                }
            )
            assert websocket.receive_json()["header"]["event"] == "task-started"
            websocket.send_json(
                {
                    "header": {"action": "continue-task", "task_id": TASK_ID},
                    "payload": {"input": {"text": "你好。今天好吗？"}},
                }
            )
            websocket.send_json(
                {
                    "header": {"action": "finish-task", "task_id": TASK_ID},
                    "payload": {"input": {}},
                }
            )
            audio = bytearray()
            while True:
                message = websocket.receive()
                data = message.get("bytes")
                if data is not None:
                    audio.extend(data)
                    continue
                text = message.get("text")
                if text is None:
                    continue
                event = json.loads(text)
                if event["header"]["event"] == "task-finished":
                    break
                assert event["header"]["event"] != "task-failed"
        assert audio[:4] == b"RIFF"
        assert audio.count(b"RIFF") == 1


def test_streaming_opus_is_rejected(config: RuntimeConfig) -> None:
    with TestClient(create_app(config)) as client:
        response = client.post(
            PATH,
            json=payload("你好。", "opus"),
            headers={"X-DashScope-SSE": "enable"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "UnsupportedParameter"

        with client.websocket_connect(WS_PATH) as websocket:
            websocket.send_json(
                {
                    "header": {
                        "action": "run-task",
                        "task_id": TASK_ID,
                        "streaming": "duplex",
                    },
                    "payload": {
                        "task_group": "audio",
                        "task": "tts",
                        "function": "SpeechSynthesizer",
                        "model": "my-tts-prod",
                        "parameters": {"voice": "voice-a", "format": "opus"},
                        "input": {},
                    },
                }
            )
            event = websocket.receive_json()
            assert event["header"]["event"] == "task-failed"
            assert event["header"]["error_code"] == "UnsupportedParameter"
