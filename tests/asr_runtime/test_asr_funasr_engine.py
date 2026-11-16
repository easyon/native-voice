from __future__ import annotations

import contextlib
import sys
import types

import pytest
from cosyvoice_common.errors import APIError
from funasr_runtime.engines.funasr import FunasrEngine
from funasr_runtime.schemas import StreamParams
from funasr_runtime.settings import (
    EngineConfig,
    FunasrEngineConfig,
    ModelConfig,
    VadConfig,
)


class FakeAutoModel:
    instances: list[FakeAutoModel] = []
    text = "你好世界。"
    raise_on_generate = False

    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        instance.kwargs = kwargs
        instance.calls = []
        cls.instances.append(instance)
        return instance

    def generate(
        self,
        input,
        cache=None,
        batch_size=1,
        language=None,
        hotwords=None,
        hotword=None,
        use_itn=True,
        itn=None,
        sentence_timestamp=False,
    ):
        self.calls.append(
            {
                "input": input,
                "cache": cache,
                "batch_size": batch_size,
                "language": language,
                "hotwords": hotwords,
                "hotword": hotword,
                "use_itn": use_itn,
                "itn": itn,
                "sentence_timestamp": sentence_timestamp,
            }
        )
        if self.raise_on_generate:
            raise RuntimeError("boom")
        samples = len(input[0]) if input else 0
        if samples < 1_600:
            return [{"text": ""}]
        return [
            {
                "text": self.text,
                "timestamp": [[0, 100]] * len(self.text),
            }
        ]


class FakeDynamicVAD:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.current_speech_start: int | None = None
        self._fed = 0

    def feed(self, audio, is_final=False):
        self._fed += len(audio)
        if self.current_speech_start is None:
            self.current_speech_start = 0
        if self._fed >= 16_000 or is_final:
            start = self.current_speech_start
            self.current_speech_start = None
            return [[start, int(self._fed / 16)]]
        return []

    def finalize(self):
        if self.current_speech_start is None:
            return []
        start = self.current_speech_start
        self.current_speech_start = None
        return [[start, int(self._fed / 16)]]

    def process(self, audio):
        return [[0, int(len(audio) / 16)]]


@pytest.fixture
def fake_funasr(monkeypatch: pytest.MonkeyPatch):
    FakeAutoModel.instances = []
    FakeAutoModel.text = "你好世界。"
    module = types.ModuleType("funasr")
    module.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "funasr", module)

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

    torch_module = types.ModuleType("torch")
    torch_module.from_numpy = lambda values: FakeTensor(values)
    torch_module.zeros = lambda size, **kwargs: FakeTensor([0.0] * int(size))
    torch_module.inference_mode = lambda: contextlib.nullcontext()
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    models_module = types.ModuleType("funasr.models")
    vad_package = types.ModuleType("funasr.models.fsmn_vad_streaming")
    dynamic_module = types.ModuleType("funasr.models.fsmn_vad_streaming.dynamic_vad")
    dynamic_module.DynamicStreamingVAD = FakeDynamicVAD
    monkeypatch.setitem(sys.modules, "funasr.models", models_module)
    monkeypatch.setitem(sys.modules, "funasr.models.fsmn_vad_streaming", vad_package)
    monkeypatch.setitem(
        sys.modules, "funasr.models.fsmn_vad_streaming.dynamic_vad", dynamic_module
    )
    yield FakeAutoModel


def _config(tmp_path, architecture: str = "paraformer") -> tuple[EngineConfig, ModelConfig, dict]:
    artifact = tmp_path / "model"
    artifact.mkdir(exist_ok=True)
    vad = tmp_path / "vad"
    vad.mkdir(exist_ok=True)
    config = EngineConfig(
        type="funasr",
        vad=VadConfig(type="energy"),
        funasr=FunasrEngineConfig(
            device="cpu",
            model_root=str(tmp_path),
            allow_download=False,
            warmup=False,
            partial_interval_ms=100,
        ),
    )
    model = ModelConfig(
        engine="funasr",
        architecture=architecture,
        artifact=str(artifact),
        vad_artifact=str(vad),
    )
    return config, model, {"m": model}


async def test_loads_model_with_deployment_options(fake_funasr, tmp_path) -> None:
    config, model, models = _config(tmp_path)
    engine = FunasrEngine(config, models)
    await engine.startup()
    try:
        assert engine._loaded["m"].model.kwargs["device"] == "cpu"
        assert engine._loaded["m"].model.kwargs["disable_update"] is True
        assert engine._loaded["m"].model.kwargs["hub"] == "ms"
        assert not engine._loaded["m"].model.kwargs.get("vad_model")
        assert await engine.ready()
    finally:
        await engine.shutdown()


async def test_stream_partial_and_final(fake_funasr, tmp_path) -> None:
    config, model, models = _config(tmp_path)
    engine = FunasrEngine(config, models)
    await engine.startup()
    try:
        params = StreamParams(
            partial_interval_ms=100,
            partial_window_s=8.0,
            extra={"begin_time_ms": 500},
        )
        stream = engine.create_stream(model, params)
        partials = await stream.feed(b"\x00\x00" * 3_200)
        assert partials
        assert partials[0].text == "你好世界。"
        assert partials[0].begin_time == 500
        final = await stream.finish()
        assert final is not None
        assert final.begin_time == 500
        assert final.end_time == 700
        assert len(final.words) == len("你好世界。")
        assert final.words[0].begin_time == 500
    finally:
        await engine.shutdown()


async def test_architecture_kwargs(fake_funasr, tmp_path) -> None:
    config, model, models = _config(tmp_path, architecture="paraformer")
    engine = FunasrEngine(config, models)
    await engine.startup()
    try:
        params = StreamParams(
            language="zh",
            hotwords="阿里巴巴,通义实验室",
            itn=False,
            timestamps=True,
            partial_interval_ms=100,
        )
        stream = engine.create_stream(model, params)
        await stream.feed(b"\x00\x00" * 3_200)
        await stream.finish()
        call = fake_funasr.instances[0].calls[-1]
        assert call["hotword"] == "阿里巴巴 通义实验室"
        assert call["use_itn"] is False
        assert call["sentence_timestamp"] is True
        assert call["language"] is None
    finally:
        await engine.shutdown()


async def test_sensevoice_language_and_itn(fake_funasr, tmp_path) -> None:
    config, model, models = _config(tmp_path, architecture="sensevoice")
    engine = FunasrEngine(config, models)
    await engine.startup()
    try:
        params = StreamParams(language="en", itn=True, partial_interval_ms=100)
        stream = engine.create_stream(model, params)
        await stream.feed(b"\x00\x00" * 3_200)
        await stream.finish()
        call = fake_funasr.instances[0].calls[-1]
        assert call["language"] == "en"
        assert call["use_itn"] is True
    finally:
        await engine.shutdown()


async def test_nano_hotwords_and_language(fake_funasr, tmp_path) -> None:
    config, model, models = _config(tmp_path, architecture="fun-asr-nano")
    model.inference_args = {"language": "中文"}
    engine = FunasrEngine(config, models)
    await engine.startup()
    try:
        params = StreamParams(language="zh", itn=True, partial_interval_ms=100)
        stream = engine.create_stream(model, params)
        await stream.feed(b"\x00\x00" * 3_200)
        await stream.finish()
        call = fake_funasr.instances[0].calls[-1]
        assert call["language"] == "中文"
        assert call["itn"] is True
    finally:
        await engine.shutdown()


async def test_text_tags_are_stripped(fake_funasr, tmp_path) -> None:
    fake_funasr.text = "<|zh|><|NEUTRAL|><|Speech|>你好世界。"
    config, model, models = _config(tmp_path, architecture="sensevoice")
    engine = FunasrEngine(config, models)
    await engine.startup()
    try:
        stream = engine.create_stream(model, StreamParams(partial_interval_ms=100))
        await stream.feed(b"\x00\x00" * 3_200)
        final = await stream.finish()
        assert final is not None
        assert final.text == "你好世界。"
    finally:
        await engine.shutdown()


async def test_missing_artifact_fails_startup(fake_funasr, tmp_path) -> None:
    config, model, models = _config(tmp_path)
    model.artifact = str(tmp_path / "missing")
    engine = FunasrEngine(config, models)
    with pytest.raises(APIError, match="not found"):
        await engine.startup()
    await engine.shutdown()


async def test_funasr_vad_stream(fake_funasr, tmp_path) -> None:
    config, model, models = _config(tmp_path)
    config.vad = VadConfig(type="funasr")
    engine = FunasrEngine(config, models)
    await engine.startup()
    try:
        vad = engine.create_vad_stream(model, StreamParams())
        events = await vad.feed(b"\x00\x00" * 16_000)
        assert any(event.__class__.__name__ == "SpeechEnd" for event in events)
    finally:
        await engine.shutdown()


async def test_transcribe_returns_segments(fake_funasr, tmp_path) -> None:
    config, model, models = _config(tmp_path)
    engine = FunasrEngine(config, models)
    await engine.startup()
    try:
        transcript = await engine.transcribe(
            model,
            b"\x00\x00" * 3_200,
            StreamParams(),
        )
        assert transcript.text == "你好世界。"
        assert transcript.duration_ms == 200
        assert transcript.segments
    finally:
        await engine.shutdown()
