#!/usr/bin/env python3
"""Replay a WAV file through the Bailian-compatible ASR WebSocket API."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
import wave
from pathlib import Path

try:
    from websockets.sync.client import connect
except Exception as exc:  # noqa: BLE001
    print("websockets is required: pip install websockets", file=sys.stderr)
    raise SystemExit(1) from exc


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1:
            raise SystemExit("only mono WAV is supported")
        if reader.getsampwidth() != 2:
            raise SystemExit("only 16-bit WAV is supported")
        if reader.getframerate() != 16_000:
            raise SystemExit(
                f"WAV sample rate is {reader.getframerate()}; expected 16000"
            )
        return reader.readframes(reader.getnframes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="ws://host:port/api-ws/v1/inference")
    parser.add_argument("model", help="configured model key, e.g. fun-asr-realtime")
    parser.add_argument("audio", type=Path, help="16 kHz mono 16-bit WAV")
    parser.add_argument("--api-key", default="", help="Bearer token")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="sleep between frames instead of sending as fast as possible",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    pcm = read_pcm(args.audio)
    task_id = uuid.uuid4().hex
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    started = time.monotonic()

    with connect(args.url, additional_headers=headers, open_timeout=10) as websocket:
        websocket.send(
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
                        "model": args.model,
                        "parameters": {
                            "format": "pcm",
                            "sample_rate": 16_000,
                            "max_sentence_silence": 1_300,
                        },
                        "input": {},
                    },
                }
            )
        )
        event = json.loads(websocket.recv(timeout=args.timeout))
        if event["header"]["event"] != "task-started":
            print(json.dumps(event, ensure_ascii=False), file=sys.stderr)
            return 1
        print(f"task-started ({time.monotonic() - started:.3f}s)")

        step = max(1, args.chunk_ms * 32)
        for offset in range(0, len(pcm), step):
            websocket.send(pcm[offset : offset + step])
            if args.realtime:
                time.sleep(args.chunk_ms / 1000.0)
        websocket.send(
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

        first_partial = None
        while True:
            message = websocket.recv(timeout=args.timeout)
            if isinstance(message, bytes):
                continue
            event = json.loads(message)
            name = event["header"]["event"]
            if name == "result-generated":
                sentence = event["payload"]["output"]["sentence"]
                now = time.monotonic() - started
                if sentence["sentence_begin"] and first_partial is None:
                    first_partial = now
                    print(f"first-partial {now:.3f}s: {sentence['text']}")
                if sentence["sentence_end"]:
                    usage = event["payload"].get("usage") or {}
                    print(
                        f"sentence {sentence['sentence_id']} "
                        f"[{sentence['begin_time']}-{sentence['end_time']}ms] "
                        f"usage={usage.get('duration')}s: {sentence['text']}"
                    )
                elif not sentence["sentence_begin"]:
                    print(f"partial: {sentence['text']}")
            elif name == "task-finished":
                print(f"task-finished ({time.monotonic() - started:.3f}s)")
                return 0
            elif name == "task-failed":
                print(
                    "task-failed: "
                    f"{event['header'].get('error_code')} "
                    f"{event['header'].get('error_message')}",
                    file=sys.stderr,
                )
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
