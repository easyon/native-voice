"""Conformance regression against the official Alibaba DashScope Python SDK.

The point of these tests is the promise we make to business code: switching
from Bailian to this runtime should be a base-URL change only.  The runtime is
started on a random loopback port with the mock engine, then the real SDK
talks to it over HTTP, HTTP/SSE and WebSocket.

SDK 2.0.4 maps the public entry points as follows:

* ``dashscope.audio.http_tts.HttpSpeechSynthesizer`` -> HTTP and HTTP/SSE
  (the class sends ``X-DashScope-SSE: enable`` when ``stream=True``).
* ``dashscope.audio.tts.SpeechSynthesizer`` -> legacy WebSocket (v1 message
  flow); despite living under ``audio.tts`` it is *not* the HTTP endpoint.
* ``dashscope.audio.tts_v2.SpeechSynthesizer`` -> WebSocket (v2 flow).

The tests deliberately clear proxy environment variables: this host may export
``HTTP_PROXY``/``ALL_PROXY``, and both ``requests`` and ``aiohttp`` honour them
even for ``127.0.0.1`` unless ``NO_PROXY`` is set.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from tts_runtime.main import create_app
from tts_runtime.settings import RuntimeConfig

dashscope = pytest.importorskip(
    "dashscope",
    reason="official dashscope SDK is not installed in this environment",
)

MODEL = "my-tts-prod"
VOICE = "voice-a"
TEXT = "你好。今天天气不错。"

_PROXY_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture(scope="module")
def runtime_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Run the runtime with the mock engine on a random loopback port."""
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    config = RuntimeConfig.default()
    config.server.host = "127.0.0.1"
    config.server.port = port
    config.logging.access_log = False
    # A real Bailian response carries a download URL, so use local storage to
    # exercise the same shape instead of the base64-only storage=none path.
    config.storage.type = "local"
    config.storage.dir = str(tmp_path_factory.mktemp("dashscope-audio"))
    config.storage.base_url = base_url
    config.storage.signing_secret = "dashscope-conformance-secret"

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(config),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="on",
        )
    )
    thread = threading.Thread(
        target=server.run,
        name="cosyvoice-dashscope-conformance",
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 30.0
    ready = False
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("runtime server thread exited during startup")
        try:
            response = httpx.get(
                f"{base_url}/readyz",
                timeout=1.0,
                trust_env=False,
            )
        except httpx.HTTPError:
            response = None
        if response is not None and response.status_code == 200:
            ready = True
            break
        time.sleep(0.05)
    if not ready:
        server.should_exit = True
        thread.join(timeout=10)
        raise RuntimeError(f"runtime did not become ready at {base_url}/readyz")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=20)
        if thread.is_alive():
            raise RuntimeError("runtime server thread did not stop")


@pytest.fixture
def endpoints(
    runtime_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, str]:
    """Point the SDK's global endpoints at the local runtime."""
    http_base = f"{runtime_server}/api/v1"
    ws_url = f"ws://{runtime_server.removeprefix('http://')}/api-ws/v1/inference"
    monkeypatch.setattr(dashscope, "base_http_api_url", http_base)
    monkeypatch.setattr(dashscope, "base_websocket_api_url", ws_url)
    monkeypatch.setattr(dashscope, "api_key", "conformance-key", raising=False)
    return http_base, ws_url


def _assert_wav(data: bytes) -> None:
    assert data, "SDK returned no audio"
    assert data.startswith(b"RIFF"), f"not a WAV payload: {data[:16]!r}"


def test_http_non_streaming(endpoints: tuple[str, str]) -> None:
    from dashscope.audio.http_tts import HttpSpeechSynthesizer

    result = HttpSpeechSynthesizer.call(
        model=MODEL,
        text=TEXT,
        voice=VOICE,
        audio_format="wav",
        sample_rate=24000,
        url=endpoints[0],  # explicit URL; base_http_api_url also works
    )

    assert result.status_code == 200, result.message
    assert result.request_id
    assert result.audio_url, "runtime returned no download URL"
    output = result.response.output
    assert output["finish_reason"] == "stop"
    # Bailian puts usage next to output, and the SDK exposes it as .usage.
    assert result.usage["characters"] > 0

    audio = httpx.get(result.audio_url, timeout=10.0, trust_env=False)
    assert audio.status_code == 200
    _assert_wav(audio.content)


def test_http_sse_streaming(endpoints: tuple[str, str]) -> None:
    from dashscope.audio.http_tts import HttpSpeechSynthesizer

    parts = list(
        HttpSpeechSynthesizer.call(
            model=MODEL,
            text=TEXT,
            voice=VOICE,
            audio_format="wav",
            sample_rate=24000,
            stream=True,
            url=endpoints[0],
        )
    )
    assert parts, "SSE path produced no results"

    audio_chunks = [part.audio_data for part in parts if part.audio_data]
    assert audio_chunks, "SSE path produced no audio"
    _assert_wav(audio_chunks[0])
    _assert_wav(b"".join(audio_chunks))

    final = parts[-1]
    assert final.response.output["finish_reason"] == "stop"
    assert final.audio_url


def test_websocket_v2(endpoints: tuple[str, str]) -> None:
    from dashscope.audio.tts_v2 import AudioFormat
    from dashscope.audio.tts_v2 import SpeechSynthesizer as SpeechSynthesizerV2

    synthesizer = SpeechSynthesizerV2(
        model=MODEL,
        voice=VOICE,
        format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        url=endpoints[1],
    )
    audio = synthesizer.call(TEXT, timeout_millis=30_000)

    assert audio, "websocket v2 returned no audio"
    assert len(audio) % 2 == 0  # 16-bit mono PCM frames
    assert synthesizer.last_response["header"]["event"] == "task-finished"


def test_websocket_v1_legacy(endpoints: tuple[str, str]) -> None:
    """The API the task names as ``dashscope.audio.tts.SpeechSynthesizer``.

    In SDK 2.0.4 this class speaks the v1 WebSocket flow, not HTTP.
    """
    from dashscope.audio.tts import SpeechSynthesizer

    result = SpeechSynthesizer.call(
        model=MODEL,
        text=TEXT,
        voice=VOICE,
        format="pcm",
        sample_rate=24000,
        base_address=endpoints[1],
        request_timeout=30,
    )

    audio = result.get_audio_data()
    assert audio, "websocket v1 returned no audio"
    assert len(audio) % 2 == 0
    # The v1 parser only records a final response when task-finished has no
    # payload; the runtime therefore omits usage for streaming="out" tasks.
    response = result.get_response()
    assert response is None or response.status_code == 200
