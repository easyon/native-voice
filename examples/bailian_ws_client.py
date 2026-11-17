"""Minimal Bailian-compatible WebSocket client for Native Voice.

TTS:
    python examples/bailian_ws_client.py --key "$NV_KEY" tts --text "你好，世界" --out out.wav

ASR:
    python examples/bailian_ws_client.py --key "$NV_KEY" asr --wav audio/asr_test.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
import wave

import numpy as np
import websockets


async def _recv_json(ws) -> dict:
    msg = await ws.recv()
    if isinstance(msg, bytes):
        raise RuntimeError("expected a JSON event, got binary audio")
    return json.loads(msg)


async def tts(args) -> None:
    task_id = str(uuid.uuid4())
    run = {
        "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
        "payload": {
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "model": "cosyvoice-local",
            "input": {},
            "parameters": {
                "text_type": "PlainText",
                "voice": args.voice,
                "format": "pcm",
                "sample_rate": 24000,
            },
        },
    }
    async with websockets.connect(args.url, extra_headers={"Authorization": f"Bearer {args.key}"}) as ws:
        await ws.send(json.dumps(run, ensure_ascii=False))
        print(await _recv_json(ws))
        await ws.send(
            json.dumps(
                {
                    "header": {"action": "continue-task", "task_id": task_id, "streaming": "duplex"},
                    "payload": {"input": {"text": args.text}},
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
        audio = bytearray()
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                audio.extend(msg)
                continue
            obj = json.loads(msg)
            header = obj.get("header", {})
            if header.get("event") == "result-generated":
                kind = obj["payload"]["output"].get("type")
                print(f"  {kind} index={obj['payload']['output']['sentence'].get('index')}")
            elif header.get("event") == "task-finished":
                print("finished", obj.get("payload"))
                break
            elif header.get("event") == "task-failed":
                raise RuntimeError(header)
    with wave.open(args.out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(bytes(audio))
    print(f"saved {len(audio) / 2 / 24000:.2f}s to {args.out}")


async def asr(args) -> None:
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly

    data, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        g = gcd(int(sr), 16000)
        data = resample_poly(data, 16000 // g, int(sr) // g).astype(np.float32)
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()
    task_id = str(uuid.uuid4())
    run = {
        "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": "paraformer-local",
            "input": {},
            "parameters": {"format": "pcm", "sample_rate": 16000},
        },
    }
    async with websockets.connect(args.url, extra_headers={"Authorization": f"Bearer {args.key}"}) as ws:
        await ws.send(json.dumps(run, ensure_ascii=False))
        print(await _recv_json(ws))
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
            obj = await _recv_json(ws)
            header = obj.get("header", {})
            if header.get("event") == "result-generated":
                s = obj["payload"]["output"]["sentence"]
                print(f"  sentence={s['sentence_id']} end={s['sentence_end']} text={s['text']!r}")
            elif header.get("event") == "task-finished":
                print("finished", obj.get("payload"))
                break
            elif header.get("event") == "task-failed":
                raise RuntimeError(header)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8090/api-ws/v1/inference")
    parser.add_argument("--key", required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tts")
    t.add_argument("--text", required=True)
    t.add_argument("--voice", default="default")
    t.add_argument("--out", default="tts_out.wav")
    a = sub.add_parser("asr")
    a.add_argument("--wav", required=True)
    args = parser.parse_args()
    asyncio.run(tts(args) if args.cmd == "tts" else asr(args))


if __name__ == "__main__":
    main()
