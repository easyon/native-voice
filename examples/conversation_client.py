"""Simulate the browser full-duplex client with a WAV file.

    python examples/conversation_client.py --key "$NV_KEY" --wav audio/asr_test.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import wave

import numpy as np
import requests
import websockets


def load_pcm16k(path: str) -> bytes:
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        g = gcd(int(sr), 16000)
        data = resample_poly(data, 16000 // g, int(sr) // g)
    # Leading/trailing silence so the VAD can calibrate and endpoint.
    audio = np.concatenate([np.zeros(16000, dtype=np.float32), data, np.zeros(16000, dtype=np.float32)])
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


async def main_async(args) -> None:
    base = args.http.rstrip("/")
    ticket = requests.post(
        base + "/v1/realtime/ticket", headers={"Authorization": f"Bearer {args.key}"}, timeout=5
    ).json()["ticket"]
    ws_url = args.ws.rstrip("/") + "/ws/conversation"
    pcm = load_pcm16k(args.wav)
    audio_out = bytearray()
    backchannel = bytearray()
    in_backchannel = False
    text = ""
    turn_id = None
    playback_started = False
    got_metrics = False
    got_playback = False

    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "auth", "ticket": ticket}))
        ready = json.loads(await ws.recv())
        print("ready", ready)
        await ws.send(json.dumps({"type": "session.start", "voice": args.voice}))
        step = 640  # 20 ms
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i + step])
            if args.realtime:
                # Backchannel/endpoint policies use the audio timeline, but a
                # paced upload also exercises the real duplex timing (and lets
                # the low-volume ack arrive while the user is still talking).
                await asyncio.sleep(step / 16000)

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                if in_backchannel:
                    backchannel.extend(msg)
                else:
                    audio_out.extend(msg)
                    if turn_id and not playback_started:
                        playback_started = True
                        await ws.send(json.dumps({"type": "playback.started", "turn_id": turn_id}))
                continue
            event = json.loads(msg)
            kind = event.get("type")
            if kind == "asr.partial":
                print(f"\r[识别] {event['text']}", end="", flush=True)
            elif kind == "asr.final":
                print(f"\n你: {event['text']}")
            elif kind == "assistant.text.delta":
                text += event["text"]
                print(event["text"], end="", flush=True)
            elif kind == "assistant.text.done":
                print()
            elif kind == "tts.audio.start":
                turn_id = event["turn_id"]
                playback_started = False
            elif kind == "assistant.audio.all_done":
                if turn_id:
                    await asyncio.sleep(0.2)
                    await ws.send(json.dumps({"type": "playback.done", "turn_id": turn_id}))
            elif kind == "assistant.backchannel.start":
                in_backchannel = True
                print(f"\n[附和] start {event.get('backchannel_id')}")
            elif kind == "assistant.backchannel.done":
                in_backchannel = False
                print(f"[附和] done {len(backchannel) / 2 / 24000:.2f}s")
            elif kind == "metrics.turn":
                got_metrics = True
                print("metrics.turn", {k: v for k, v in event.items() if k != "type"})
            elif kind == "metrics.playback":
                got_playback = True
                print("metrics.playback", {k: v for k, v in event.items() if k != "type"})
            elif kind == "assistant.interrupted":
                print("[interrupted]", event.get("reason"))
            elif kind == "error":
                print("[error]", event)
                break
            if got_metrics and got_playback:
                break

    if backchannel:
        bc_path = str(args.out).replace(".wav", "_backchannel.wav")
        with wave.open(bc_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(bytes(backchannel))
        print(f"saved {len(backchannel) / 2 / 24000:.2f}s backchannel to {bc_path}")

    if audio_out:
        with wave.open(args.out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(bytes(audio_out))
        print(f"saved {len(audio_out) / 2 / 24000:.2f}s audio to {args.out}")
    print("reply:", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", default="http://127.0.0.1:8090")
    parser.add_argument("--ws", default="ws://127.0.0.1:8090")
    parser.add_argument("--key", required=True)
    parser.add_argument("--wav", default="audio/asr_test.wav")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--out", default="conversation_out.wav")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace PCM at 20 ms/frame instead of uploading the whole file at once",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
