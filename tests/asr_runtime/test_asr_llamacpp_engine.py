from __future__ import annotations

import subprocess

import pytest
from cosyvoice_common.errors import APIError
from funasr_runtime.engines.llamacpp import LlamaCppEngine
from funasr_runtime.schemas import StreamParams
from funasr_runtime.settings import (
    EngineConfig,
    LlamaCppEngineConfig,
    ModelConfig,
    VadConfig,
)


def _setup(tmp_path, runner):
    model_path = tmp_path / "paraformer-q8.gguf"
    model_path.write_bytes(b"gguf")
    vad_path = tmp_path / "fsmn-vad.gguf"
    vad_path.write_bytes(b"vad")
    config = EngineConfig(
        type="llamacpp",
        vad=VadConfig(type="energy"),
        llamacpp=LlamaCppEngineConfig(
            binary_dir=str(tmp_path / "bin"),
            args=["{binary}", "-m", "{model}", "-a", "{audio}"],
            timeout_s=5,
        ),
    )
    model = ModelConfig(
        engine="llamacpp",
        architecture="paraformer",
        artifact=str(model_path),
        vad_artifact=str(vad_path),
    )
    engine = LlamaCppEngine(config, {"m": model}, runner=runner)
    return engine, model


async def test_transcribe_uses_binary_and_vad(tmp_path) -> None:
    seen = {}

    def runner(command, timeout):
        seen["command"] = command
        seen["timeout"] = timeout
        seen["audio_exists"] = __import__("os").path.exists(
            command[command.index("-a") + 1]
        )
        return subprocess.CompletedProcess(command, 0, stdout="你好世界", stderr="")

    engine, model = _setup(tmp_path, runner)
    await engine.startup()
    try:
        transcript = await engine.transcribe(
            model, b"\x00\x00" * 3_200, StreamParams(timestamps=False)
        )
    finally:
        await engine.shutdown()
    assert transcript.text == "你好世界"
    assert transcript.duration_ms == 200
    assert "--vad" in seen["command"]
    assert str(model.artifact) in seen["command"]
    assert seen["audio_exists"] is True
    assert seen["timeout"] == 5


async def test_transcribe_parses_srt_for_verbose_json(tmp_path) -> None:
    srt = (
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        "你好\n\n"
        "2\n"
        "00:00:00,500 --> 00:00:01,000\n"
        "世界\n"
    )

    def runner(command, timeout):
        return subprocess.CompletedProcess(command, 0, stdout=srt, stderr="")

    engine, model = _setup(tmp_path, runner)
    await engine.startup()
    try:
        transcript = await engine.transcribe(
            model, b"\x00\x00" * 8_000, StreamParams(timestamps=True)
        )
    finally:
        await engine.shutdown()
    assert transcript.text == "你好世界"
    assert [segment.text for segment in transcript.segments] == ["你好", "世界"]
    assert transcript.segments[0].begin_time == 0
    assert transcript.segments[1].begin_time == 500


async def test_stream_finish_does_not_apply_vad_again(tmp_path) -> None:
    commands = []

    def runner(command, timeout):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="你好", stderr="")

    engine, model = _setup(tmp_path, runner)
    await engine.startup()
    try:
        stream = engine.create_stream(model, StreamParams(partial_interval_ms=0))
        assert await stream.feed(b"\x00\x00" * 3_200) == []
        final = await stream.finish()
    finally:
        await engine.shutdown()
    assert final is not None
    assert final.text == "你好"
    assert "--vad" not in commands[-1]


async def test_nonzero_exit_is_internal_error(tmp_path) -> None:
    def runner(command, timeout):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="bad model")

    engine, model = _setup(tmp_path, runner)
    await engine.startup()
    try:
        with pytest.raises(APIError, match="bad model"):
            await engine.transcribe(model, b"\x00\x00" * 3_200, StreamParams())
    finally:
        await engine.shutdown()


async def test_timeout_maps_to_timeout_error(tmp_path) -> None:
    def runner(command, timeout):
        raise subprocess.TimeoutExpired(command, timeout)

    engine, model = _setup(tmp_path, runner)
    await engine.startup()
    try:
        with pytest.raises(APIError) as excinfo:
            await engine.transcribe(model, b"\x00\x00" * 3_200, StreamParams())
        assert excinfo.value.code == "Timeout"
    finally:
        await engine.shutdown()


async def test_missing_binary_fails_startup(tmp_path) -> None:
    engine, _ = _setup(tmp_path, runner=None)  # type: ignore[arg-type]
    with pytest.raises(APIError, match="binary not found"):
        await engine.startup()
    await engine.shutdown()


def test_nano_command_includes_encoder(tmp_path) -> None:
    engine, model = _setup(tmp_path, runner=lambda command, timeout: None)
    model.architecture = "fun-asr-nano"
    encoder = tmp_path / "encoder.gguf"
    encoder.write_bytes(b"encoder")
    model.encoder_artifact = str(encoder)
    command = engine._build_command(model, "audio.wav", True, False)
    assert command[0].endswith("llama-funasr-cli")
    assert command[1:3] == ["--enc", str(encoder)]
    assert "--vad" in command
