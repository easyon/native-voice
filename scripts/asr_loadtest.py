#!/usr/bin/env python3
"""Concurrent WebSocket load test for the FunASR realtime endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
import wave
from pathlib import Path

try:  # websockets >= 13
    from websockets.asyncio.client import connect

    _HEADER_KW = "additional_headers"
except ImportError:  # websockets 12
    from websockets.client import connect  # type: ignore[no-redef]

    _HEADER_KW = "extra_headers"


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != 16_000
        ):
            raise SystemExit("load test expects 16 kHz mono 16-bit WAV")
        return reader.readframes(reader.getnframes())


async def one_task(url: str, model: str, pcm: bytes, api_key: str) -> dict:
    started = time.monotonic()
    task_id = uuid.uuid4().hex
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    first_partial = None
    sentences = 0
    error = ""
    async with connect(url, **{_HEADER_KW: headers}, open_timeout=10) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "header": {
                        "action": "run-task",
                        "task_id": task_id,
                        "streaming": "duplex",
                    },
                    "payload": {
                        "task_group": "audio",
                        "task": "asr",
                        "function": "recognition",
                        "model": model,
                        "parameters": {
                            "format": "pcm",
                            "sample_rate": 16_000,
                        },
                        "input": {},
                    },
                }
            )
        )
        event = json.loads(await websocket.recv())
        if event["header"]["event"] != "task-started":
            error = event["header"].get("error_code", "task-not-started")
            return {"ok": False, "error": error, "elapsed": 0.0}

        step = 3_200
        for offset in range(0, len(pcm), step):
            await websocket.send(pcm[offset : offset + step])
        await websocket.send(
            json.dumps(
                {
                    "header": {
                        "action": "finish-task",
                        "task_id": task_id,
                        "streaming": "duplex",
                    },
                    "payload": {"input": {}},
                }
            )
        )
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                continue
            event = json.loads(message)
            name = event["header"]["event"]
            if name == "result-generated":
                sentence = event["payload"]["output"]["sentence"]
                if sentence["sentence_begin"] and first_partial is None:
                    first_partial = time.monotonic() - started
                if sentence["sentence_end"]:
                    sentences += 1
            elif name == "task-finished":
                break
            elif name == "task-failed":
                error = event["header"].get("error_code", "task-failed")
                break
    return {
        "ok": not error,
        "error": error,
        "elapsed": time.monotonic() - started,
        "first_partial": first_partial,
        "sentences": sentences,
    }


async def run(args) -> int:
    pcm = read_pcm(args.audio)
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async def worker():
        async with semaphore:
            results.append(
                await one_task(args.url, args.model, pcm, args.api_key)
            )

    started = time.monotonic()
    await asyncio.gather(*(worker() for _ in range(args.tasks)))
    elapsed = time.monotonic() - started

    failures = [result for result in results if not result["ok"]]
    latencies = sorted(result["elapsed"] for result in results if result["ok"])
    partials = sorted(
        result["first_partial"]
        for result in results
        if result.get("first_partial") is not None
    )
    print(f"tasks={len(results)} concurrency={args.concurrency} wall={elapsed:.2f}s")
    print(f"ok={len(results) - len(failures)} failed={len(failures)}")
    if latencies:
        print(
            f"latency p50={_pct(latencies, 50):.3f}s "
            f"p95={_pct(latencies, 95):.3f}s max={latencies[-1]:.3f}s"
        )
    if partials:
        print(
            f"first_partial p50={_pct(partials, 50):.3f}s "
            f"p95={_pct(partials, 95):.3f}s"
        )
    if failures:
        counts: dict[str, int] = {}
        for result in failures:
            counts[result["error"]] = counts.get(result["error"], 0) + 1
        print(f"errors={counts}")
        return 1
    return 0


def _pct(values: list[float], percentile: float) -> float:
    index = min(len(values) - 1, int(round((percentile / 100) * (len(values) - 1))))
    return values[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="ws://host:port/api-ws/v1/inference")
    parser.add_argument("model", help="configured model key")
    parser.add_argument("audio", type=Path, help="16 kHz mono 16-bit WAV")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=16)
    parser.add_argument("--api-key", default="")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
