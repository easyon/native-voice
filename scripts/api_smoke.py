"""End-to-end protocol smoke test against a running Bailian-compatible runtime.

Checks the three access paths a business caller can use -- plain HTTP, SSE and
the duplex WebSocket -- and reports latency / audio volume for each. Run it on
the GPU host so the numbers reflect the real model, not a mock engine.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import sys
import time
import uuid

import httpx
import websockets

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "cosyvoice3-0.5b"
VOICE = sys.argv[3] if len(sys.argv) > 3 else "assistant"
TEXT = "你好，我是你的语音助手，很高兴为你服务。今天想聊点什么？"

HTTP_PATH = "/api/v1/services/audio/tts/SpeechSynthesizer"
WS_PATH = "/api-ws/v1/inference"


def payload(fmt: str) -> dict:
    return {
        "model": MODEL,
        "input": {
            "text": TEXT,
            "voice": VOICE,
            "format": fmt,
            "sample_rate": 24000,
        },
    }


async def http_plain() -> dict:
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        started = time.perf_counter()
        response = await client.post(BASE + HTTP_PATH, json=payload("pcm"))
        elapsed = time.perf_counter() - started
    body = response.json()
    audio_out = body.get("output", {}).get("audio", {}) or {}
    audio = base64.b64decode(audio_out.get("data") or "")
    if not audio and audio_out.get("url"):
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            audio = (await client.get(audio_out["url"])).content
    return {
        "ok": response.status_code == 200 and len(audio) > 0,
        "status": response.status_code,
        "elapsed_s": round(elapsed, 3),
        "audio_bytes": len(audio),
        "has_url": bool(audio_out.get("url")),
        "first_event": body.get("output", {}).get("finish_reason"),
    }


async def http_sse() -> dict:
    chunks: list[bytes] = []
    events = 0
    ttff = None
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        async with client.stream(
            "POST",
            BASE + HTTP_PATH,
            json=payload("pcm"),
            headers={"X-DashScope-SSE": "enable"},
        ) as response:
            status = response.status_code
            ctype = response.headers.get("content-type", "")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = json.loads(line[5:].strip())
                events += 1
                b64 = data.get("output", {}).get("audio", {}).get("data") or ""
                if b64:
                    if ttff is None:
                        ttff = time.perf_counter() - started
                    chunks.append(base64.b64decode(b64))
    audio = b"".join(chunks)
    return {
        "ok": status == 200 and events > 0 and len(audio) > 0,
        "status": status,
        "content_type": ctype.split(";")[0],
        "events": events,
        "ttff_s": round(ttff, 3) if ttff else None,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "audio_bytes": len(audio),
    }


async def ws_duplex() -> dict:
    url = (BASE.replace("https://", "wss://").replace("http://", "ws://")) + WS_PATH
    task_id = str(uuid.uuid4())
    started = time.perf_counter()
    ttff = None
    audio_bytes = 0
    frames = 0
    events: list[str] = []
    kwargs: dict[str, object] = {"max_size": 8 * 1024 * 1024}
    # websockets >= 14 honours proxy env vars; earlier versions have no kwarg.
    if "proxy" in inspect.signature(websockets.connect).parameters:
        kwargs["proxy"] = None
    async with websockets.connect(url, **kwargs) as ws:
        await ws.send(
            json.dumps(
                {
                    "header": {
                        "action": "run-task",
                        "task_id": task_id,
                        "streaming": "duplex",
                    },
                    "payload": {
                        "task_group": "audio",
                        "task": "tts",
                        "function": "SpeechSynthesizer",
                        "model": MODEL,
                        "parameters": {
                            "voice": VOICE,
                            "format": "pcm",
                            "sample_rate": 24000,
                        },
                        "input": {},
                    },
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "header": {
                        "action": "continue-task",
                        "task_id": task_id,
                        "streaming": "duplex",
                    },
                    "payload": {"input": {"text": TEXT}},
                }
            )
        )
        await ws.send(
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
            message = await asyncio.wait_for(ws.recv(), timeout=180)
            if isinstance(message, bytes):
                if ttff is None:
                    ttff = time.perf_counter() - started
                audio_bytes += len(message)
                frames += 1
                continue
            event = json.loads(message)
            name = event.get("header", {}).get("event", "")
            events.append(name)
            if name in {"task-finished", "task-failed"}:
                return {
                    "ok": name == "task-finished" and audio_bytes > 0,
                    "events": events,
                    "frames": frames,
                    "audio_bytes": audio_bytes,
                    "ttff_s": round(ttff, 3) if ttff else None,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "error": event.get("header", {}).get("error_message"),
                }


async def main() -> int:
    results = {}
    for name, fn in (
        ("http", http_plain),
        ("sse", http_sse),
        ("ws", ws_duplex),
    ):
        try:
            results[name] = await fn()
        except Exception as exc:  # noqa: BLE001 - report, keep going
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({name: results[name]}, ensure_ascii=False), flush=True)
    print(json.dumps({"ok": all(r.get("ok") for r in results.values())}))
    return 0 if results["ws"].get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
