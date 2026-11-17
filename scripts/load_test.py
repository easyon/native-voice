"""Concurrent protocol load test (ASR/TTS) against a running gateway.

    python scripts/load_test.py --key "$NV_KEY" tts -n 4
    python scripts/load_test.py --key "$NV_KEY" mixed -n 4 --wav audio/asr_test.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid

import numpy as np
import websockets


def headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


async def _recv(ws):
    msg = await ws.recv()
    return msg if isinstance(msg, bytes) else json.loads(msg)


async def tts_once(url: str, key: str, text: str) -> dict:
    task_id = str(uuid.uuid4())
    t0 = time.monotonic()
    first = None
    async with websockets.connect(url, extra_headers=headers(key)) as ws:
        await ws.send(
            json.dumps(
                {
                    "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
                    "payload": {
                        "task_group": "audio",
                        "task": "tts",
                        "function": "SpeechSynthesizer",
                        "model": "local",
                        "input": {},
                        "parameters": {
                            "text_type": "PlainText",
                            "voice": "default",
                            "format": "pcm",
                            "sample_rate": 24000,
                        },
                    },
                },
                ensure_ascii=False,
            )
        )
        await _recv(ws)
        await ws.send(
            json.dumps(
                {
                    "header": {"action": "continue-task", "task_id": task_id, "streaming": "duplex"},
                    "payload": {"input": {"text": text}},
                },
                ensure_ascii=False,
            )
        )
        await ws.send(
            json.dumps(
                {
                    "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
                    "payload": {"input": {}},
                }
            )
        )
        while True:
            msg = await _recv(ws)
            if isinstance(msg, bytes):
                if first is None:
                    first = time.monotonic() - t0
                continue
            if msg.get("header", {}).get("event") == "task-finished":
                break
    return {"kind": "tts", "first": first or 0.0, "total": time.monotonic() - t0}


async def asr_once(url: str, key: str, pcm: bytes) -> dict:
    task_id = str(uuid.uuid4())
    t0 = time.monotonic()
    first = None
    async with websockets.connect(url, extra_headers=headers(key)) as ws:
        await ws.send(
            json.dumps(
                {
                    "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
                    "payload": {
                        "task_group": "audio",
                        "task": "asr",
                        "function": "recognition",
                        "model": "local",
                        "input": {},
                        "parameters": {"format": "pcm", "sample_rate": 16000},
                    },
                }
            )
        )
        await _recv(ws)
        step = 6400
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i + step])
        await ws.send(
            json.dumps(
                {
                    "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
                    "payload": {"input": {}},
                }
            )
        )
        while True:
            msg = await _recv(ws)
            if isinstance(msg, bytes):
                if first is None:
                    first = time.monotonic() - t0
                continue
            if msg.get("header", {}).get("event") == "result-generated" and first is None:
                first = time.monotonic() - t0
            if msg.get("header", {}).get("event") == "task-finished":
                break
    return {"kind": "asr", "first": first or 0.0, "total": time.monotonic() - t0}


def pcm_from_wav(path: str) -> bytes:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), 16000)
        data = resample_poly(data, 16000 // g, int(sr) // g)
    return (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()


async def main_async(args) -> None:
    pcm = pcm_from_wav(args.wav) if args.wav else b""
    tasks = []
    for i in range(args.n):
        if args.mode == "tts" or (args.mode == "mixed" and i % 2 == 0):
            tasks.append(tts_once(args.url, args.key, f"你好，我是第{i + 1}位用户，正在测试并发语音合成。"))
        else:
            tasks.append(asr_once(args.url, args.key, pcm))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = [r for r in results if isinstance(r, dict)]
    for r in results:
        if isinstance(r, Exception):
            print("ERROR", repr(r))
    for kind in ("tts", "asr"):
        vals = [r for r in ok if r["kind"] == kind]
        if vals:
            firsts = [v["first"] for v in vals]
            totals = [v["total"] for v in vals]
            print(
                f"{kind}: n={len(vals)} first p50={statistics.median(firsts):.2f}s "
                f"max={max(firsts):.2f}s total p50={statistics.median(totals):.2f}s"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tts", "asr", "mixed"])
    parser.add_argument("-n", type=int, default=4)
    parser.add_argument("--url", default="ws://127.0.0.1:8090/api-ws/v1/inference")
    parser.add_argument("--key", required=True)
    parser.add_argument("--wav", default="audio/asr_test.wav")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
