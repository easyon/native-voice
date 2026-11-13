#!/usr/bin/env python3
"""Gateway load generator for the Bailian-compatible runtime.

This is a protocol/backpressure test, not a model benchmark. Run it against the
mock engine first, then against the real GPU service. It measures end-to-end
latency from request start to first audio and to task completion.

Examples:
  python scripts/loadtest.py --mode ws --concurrency 100 --tasks 500
  python scripts/loadtest.py --mode sse --concurrency 100 --tasks 500
  python scripts/loadtest.py --mode http --concurrency 100 --tasks 500 \
      --base-url http://127.0.0.1:8000 --api-key test-key

  # Worker pool: hit each worker directly (round-robin across concurrent
  # clients), then compare with the nginx endpoint.
  python scripts/loadtest.py --mode sse --concurrency 4 --tasks 40 \
      --base-urls http://127.0.0.1:18000,http://127.0.0.1:18001

The report includes `aggregate_rtf`: synthesized audio seconds per wall
second. Compare it across pool sizes — that, not request count, is the
throughput number that matters.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import websockets

HTTP_PATH = "/api/v1/services/audio/tts/SpeechSynthesizer"
WS_PATH = "/api-ws/v1/inference"
SAMPLE_RATE = 24_000
BYTES_PER_SAMPLE = 2  # pcm_s16le, mono


@dataclass
class Sample:
    ttff_ms: float
    total_ms: float
    # Raw (decoded) audio bytes. JSON payloads carry base64, WS binary frames
    # are raw; both are normalised here so throughput is comparable.
    audio_bytes: int = 0
    audio_s: float = 0.0
    url: str = ""
    error: str | None = None


@dataclass
class Results:
    samples: list[Sample] = field(default_factory=list)

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)


def _b64_bytes(data: str) -> int:
    """Decoded size of a base64 string without decoding it."""
    if not data:
        return 0
    padding = data[-2:].count("=")
    return max(0, len(data) * 3 // 4 - padding)


def _audio_seconds(fmt: str, raw_bytes: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Approximate duration of raw PCM/WAV audio (16-bit mono)."""
    if fmt == "pcm":
        return raw_bytes / (sample_rate * BYTES_PER_SAMPLE)
    if fmt == "wav":
        # 44-byte canonical header; the runtime writes a plain PCM wav.
        return max(0, raw_bytes - 44) / (sample_rate * BYTES_PER_SAMPLE)
    return 0.0


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[index]


def _target_ws_url(base_url: str) -> str:
    if base_url.startswith("https://"):
        return "wss://" + base_url.removeprefix("https://") + WS_PATH
    if base_url.startswith("http://"):
        return "ws://" + base_url.removeprefix("http://") + WS_PATH
    return base_url.rstrip("/") + WS_PATH


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _http_payload(model: str, voice: str, text: str, fmt: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": {
            "text": text,
            "voice": voice,
            "format": fmt,
            "sample_rate": 24000,
        },
    }


def _run_task(task_id: str, model: str, voice: str, fmt: str) -> dict[str, Any]:
    return {
        "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
        "payload": {
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "model": model,
            "parameters": {"voice": voice, "format": fmt, "sample_rate": 24000},
            "input": {},
        },
    }


def _continue_task(task_id: str, text: str) -> dict[str, Any]:
    return {
        "header": {"action": "continue-task", "task_id": task_id, "streaming": "duplex"},
        "payload": {"input": {"text": text}},
    }


def _finish_task(task_id: str) -> dict[str, Any]:
    return {
        "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
        "payload": {"input": {}},
    }


async def one_http(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str | None,
    model: str,
    voice: str,
    text: str,
    fmt: str,
) -> Sample:
    started = time.perf_counter()
    try:
        response = await client.post(
            base_url.rstrip("/") + HTTP_PATH,
            json=_http_payload(model, voice, text, fmt),
            headers=_headers(api_key),
        )
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            return Sample(elapsed, elapsed, error=f"HTTP {response.status_code}")
        body = response.json()
        audio = body.get("output", {}).get("audio", {})
        raw = _b64_bytes(audio.get("data") or "")
        return Sample(
            elapsed,
            elapsed,
            audio_bytes=raw,
            audio_s=_audio_seconds(fmt, raw),
        )
    except Exception as exc:  # noqa: BLE001 - load generator must keep going
        elapsed = (time.perf_counter() - started) * 1000
        return Sample(elapsed, elapsed, error=f"{type(exc).__name__}: {exc}")


async def one_sse(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str | None,
    model: str,
    voice: str,
    text: str,
    fmt: str,
) -> Sample:
    started = time.perf_counter()
    first: float | None = None
    audio_bytes = 0
    try:
        headers = {**_headers(api_key), "X-DashScope-SSE": "enable"}
        async with client.stream(
            "POST",
            base_url.rstrip("/") + HTTP_PATH,
            json=_http_payload(model, voice, text, fmt),
            headers=headers,
        ) as response:
            if response.status_code != 200:
                await response.aread()
                elapsed = (time.perf_counter() - started) * 1000
                return Sample(elapsed, elapsed, error=f"HTTP {response.status_code}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:].strip())
                output = payload.get("output") or {}
                if output.get("finish_reason") == "stop":
                    break
                audio = output.get("audio") or {}
                if output.get("type") == "sentence-synthesis" and audio.get("data"):
                    if first is None:
                        first = time.perf_counter()
                    audio_bytes += _b64_bytes(audio["data"])
        elapsed = (time.perf_counter() - started) * 1000
        ttff = (first - started) * 1000 if first is not None else elapsed
        return Sample(
            ttff,
            elapsed,
            audio_bytes=audio_bytes,
            audio_s=_audio_seconds(fmt, audio_bytes),
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000
        return Sample(elapsed, elapsed, error=f"{type(exc).__name__}: {exc}")


async def _ws_task(
    websocket: Any,
    model: str,
    voice: str,
    text: str,
    fmt: str,
    started: float,
) -> Sample:
    task_id = str(uuid.uuid4())
    first: float | None = None
    audio_bytes = 0
    await websocket.send(json.dumps(_run_task(task_id, model, voice, fmt)))
    while True:
        message = await websocket.recv()
        if isinstance(message, bytes):
            continue
        event = json.loads(message)
        if event["header"]["event"] == "task-started":
            break
        if event["header"]["event"] == "task-failed":
            elapsed = (time.perf_counter() - started) * 1000
            return Sample(elapsed, elapsed, error=event["header"].get("error_code"))

    await websocket.send(json.dumps(_continue_task(task_id, text)))
    await websocket.send(json.dumps(_finish_task(task_id)))
    while True:
        message = await websocket.recv()
        if isinstance(message, bytes):
            if first is None:
                first = time.perf_counter()
            audio_bytes += len(message)
            continue
        event = json.loads(message)
        name = event["header"]["event"]
        if name == "result-generated":
            audio = (event.get("payload") or {}).get("output", {}).get("audio") or {}
            if audio.get("data"):
                if first is None:
                    first = time.perf_counter()
                audio_bytes += _b64_bytes(audio["data"])
            continue
        if name == "task-failed":
            elapsed = (time.perf_counter() - started) * 1000
            return Sample(elapsed, elapsed, error=event["header"].get("error_code"))
        if name == "task-finished":
            elapsed = (time.perf_counter() - started) * 1000
            ttff = (first - started) * 1000 if first is not None else elapsed
            return Sample(
                ttff,
                elapsed,
                audio_bytes=audio_bytes,
                audio_s=_audio_seconds(fmt, audio_bytes),
            )


async def one_ws_connection(
    ws_url: str,
    api_key: str | None,
    model: str,
    voice: str,
    text: str,
    fmt: str,
    tasks: int,
) -> list[Sample]:
    samples: list[Sample] = []
    for _ in range(max(1, tasks)):
        started = time.perf_counter()
        try:
            headers = _headers(api_key)
            try:
                connect = websockets.connect(
                    ws_url,
                    additional_headers=headers,
                    max_size=None,
                    ping_interval=20,
                    open_timeout=10,
                )
            except TypeError:  # websockets < 13
                connect = websockets.connect(
                    ws_url,
                    extra_headers=headers,
                    max_size=None,
                    ping_interval=20,
                    open_timeout=10,
                )
            async with connect as websocket:
                samples.append(
                    await _ws_task(websocket, model, voice, text, fmt, started)
                )
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - started) * 1000
            samples.append(Sample(elapsed, elapsed, error=f"{type(exc).__name__}: {exc}"))
    return samples


async def run(args: argparse.Namespace) -> Results:
    results = Results()
    limits = httpx.Limits(
        max_connections=max(100, args.concurrency * 2),
        max_keepalive_connections=max(100, args.concurrency),
    )
    timeout = httpx.Timeout(args.timeout_s)
    client = httpx.AsyncClient(timeout=timeout, limits=limits)
    urls: list[str] = args.urls

    async def worker(index: int) -> None:
        if args.ramp_s:
            await asyncio.sleep(args.ramp_s * index / max(1, args.concurrency))
        # Round-robin over the supplied endpoints. For a process pool this maps
        # concurrent clients onto workers without needing shared state.
        url = urls[index % len(urls)]
        per_worker = max(1, args.tasks // args.concurrency)
        if args.mode == "ws":
            samples = await one_ws_connection(
                _target_ws_url(url),
                args.api_key,
                args.model,
                args.voice,
                args.text,
                args.format,
                per_worker,
            )
        else:
            samples = []
            for _ in range(per_worker):
                if args.mode == "sse":
                    sample = await one_sse(
                        client,
                        url,
                        args.api_key,
                        args.model,
                        args.voice,
                        args.text,
                        args.format,
                    )
                else:
                    sample = await one_http(
                        client,
                        url,
                        args.api_key,
                        args.model,
                        args.voice,
                        args.text,
                        args.format,
                    )
                samples.append(sample)
        for sample in samples:
            sample.url = url
            results.add(sample)

    try:
        await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
    finally:
        await client.aclose()
    return results


def report(results: Results, elapsed_s: float, urls: list[str] | None = None) -> int:
    ok = [s for s in results.samples if s.error is None]
    failed = [s for s in results.samples if s.error is not None]
    print(f"requests: {len(results.samples)}  ok: {len(ok)}  failed: {len(failed)}")
    print(f"wall: {elapsed_s:.2f}s  throughput: {len(ok) / max(elapsed_s, 1e-9):.2f}/s")
    if ok:
        ttff = [s.ttff_ms for s in ok]
        total = [s.total_ms for s in ok]
        audio = [s.audio_bytes for s in ok]
        print(
            "ttff_ms  "
            f"p50={_pct(ttff, 50):.1f} p95={_pct(ttff, 95):.1f} "
            f"p99={_pct(ttff, 99):.1f} max={max(ttff):.1f}"
        )
        print(
            "total_ms "
            f"p50={_pct(total, 50):.1f} p95={_pct(total, 95):.1f} "
            f"p99={_pct(total, 99):.1f} max={max(total):.1f}"
        )
        print(
            "audio_bytes "
            f"mean={statistics.fmean(audio):.0f} "
            f"p50={_pct(audio, 50):.0f} p95={_pct(audio, 95):.0f}"
        )
        # Aggregate real-time factor: synthesized audio seconds per wall
        # second. This is the number that should grow with pool size.
        audio_s = sum(s.audio_s for s in ok)
        print(
            f"audio: {audio_s:.2f}s synthesized  "
            f"aggregate_rtf={audio_s / max(elapsed_s, 1e-9):.2f}x"
        )
    if ok and urls and len(urls) > 1:
        print("per-url:")
        print(f"  {'url':<28} {'ok':>5} {'ttff_p50':>9} {'audio_s':>8}")
        for url in urls:
            subset = [s for s in ok if s.url == url]
            if not subset:
                continue
            subset_ttff = [s.ttff_ms for s in subset]
            subset_audio = sum(s.audio_s for s in subset)
            print(
                f"  {url:<28} {len(subset):>5} {_pct(subset_ttff, 50):>9.1f} "
                f"{subset_audio:>8.2f}"
            )
    if failed:
        counts: dict[str, int] = {}
        for sample in failed:
            counts[sample.error or "unknown"] = counts.get(sample.error or "unknown", 0) + 1
        print("failures:")
        for message, count in sorted(counts.items(), key=lambda item: -item[1]):
            print(f"  {count:6d}  {message}")
        return 1
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--base-urls",
        default=None,
        help=(
            "comma-separated endpoints, assigned round-robin to concurrent "
            "workers; for a process pool pass each worker and the public "
            "endpoint to compare them"
        ),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--mode", choices=("ws", "sse", "http"), default="ws")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--ramp-s", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--model", default="my-tts-prod")
    parser.add_argument("--voice", default="voice-a")
    parser.add_argument("--format", default="pcm")
    parser.add_argument(
        "--text",
        default="你好，这里是实时语音合成压测。今天想了解点什么？",
    )
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.tasks < 1:
        parser.error("concurrency and tasks must be >= 1")
    raw_urls = args.base_urls if args.base_urls else args.base_url
    urls = [url.strip().rstrip("/") for url in raw_urls.split(",") if url.strip()]
    if not urls:
        parser.error("--base-url / --base-urls produced no endpoint")
    args.urls = urls
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    results = asyncio.run(run(args))
    return report(results, time.perf_counter() - started, args.urls)


if __name__ == "__main__":
    raise SystemExit(main())
