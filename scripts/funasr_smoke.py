#!/usr/bin/env python3
"""Direct FunASR smoke test: load, VAD, partial/final latency and RTF.

This bypasses the gateway so a target host can validate model files, CUDA and
the FunASR API before deploying the service.

    python scripts/funasr_smoke.py \
      --model-dir ./models/Fun-ASR-Nano-2512 \
      --vad-dir ./models/fsmn-vad \
      --audio sample.wav --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from funasr_runtime.core.audio import decode_to_pcm16k


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--vad-dir", default="")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--architecture",
        default="fun-asr-nano",
        choices=["fun-asr-nano", "sensevoice", "paraformer"],
    )
    parser.add_argument("--language", default="")
    parser.add_argument("--hotwords", default="")
    parser.add_argument("--max-sentence-silence", type=int, default=1_300)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="simulate realtime partial decoding (VAD + bounded windows)",
    )
    parser.add_argument("--partial-interval-ms", type=int, default=960)
    parser.add_argument("--partial-window-s", type=float, default=8.0)
    args = parser.parse_args()

    try:
        import torch
        from funasr import AutoModel
    except Exception as exc:  # noqa: BLE001
        print(
            "FunASR is required: pip install -e 'packages/asr_runtime[funasr]'",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return 1

    audio_path = Path(args.audio)
    pcm = decode_to_pcm16k(
        audio_path.read_bytes(),
        audio_path.suffix.lstrip(".").lower() or None,
    )
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    duration_s = len(audio) / 16_000.0
    print(f"audio: {duration_s:.2f}s")

    load_started = time.monotonic()
    kwargs = {
        "model": args.model_dir,
        "device": args.device,
        "disable_update": True,
    }
    if args.architecture == "fun-asr-nano":
        kwargs["trust_remote_code"] = True
    model = AutoModel(**kwargs)
    vad = None
    if args.vad_dir:
        from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

        vad_model = AutoModel(
            model=args.vad_dir,
            device="cpu",
            disable_update=True,
        )
        vad = DynamicStreamingVAD(
            vad_model,
            sample_rate=16_000,
            silence_schedule=[(float("inf"), args.max_sentence_silence)],
        )
    print(f"load: {time.monotonic() - load_started:.2f}s")

    if args.stream:
        return _stream(args, model, vad, audio, torch)

    segments = [(0, int(duration_s * 1000))]
    if vad is not None:
        segments = vad.process(torch.from_numpy(audio).float())
    total_decode = 0.0
    texts = []
    for begin_ms, end_ms in segments:
        chunk = audio[
            int(begin_ms * 16) : min(len(audio), int(end_ms * 16))
        ]
        if len(chunk) < 1_600:
            continue
        started = time.monotonic()
        text = _generate(model, chunk, args)
        elapsed = time.monotonic() - started
        total_decode += elapsed
        texts.append(text)
        print(f"[{begin_ms}-{end_ms}ms] {elapsed:.3f}s: {text}")
    print(f"text: {''.join(texts)}")
    print(
        f"decode_total={total_decode:.3f}s "
        f"rtf={total_decode / max(duration_s, 0.001):.3f}"
    )
    return 0


def _stream(args, model, vad, audio, torch) -> int:
    if vad is None:
        print("--stream requires --vad-dir", file=sys.stderr)
        return 1
    chunk_samples = int(16_000 * 0.2)
    segment_start: int | None = None
    last_decode_ms = 0
    last_text = ""
    first_partial = None
    started = time.monotonic()
    finals = []
    for offset in range(0, len(audio), chunk_samples):
        chunk = audio[offset : offset + chunk_samples]
        segments = vad.feed(torch.from_numpy(chunk).float())
        current_ms = int((offset + len(chunk)) / 16)
        for begin_ms, end_ms in segments:
            segment_audio = audio[int(begin_ms * 16) : int(end_ms * 16)]
            text = _generate(model, segment_audio, args)
            finals.append(text)
            print(f"final [{begin_ms}-{end_ms}ms]: {text}")
            segment_start = None
        if segment_start is None and vad.current_speech_start is not None:
            segment_start = int(vad.current_speech_start)
            last_decode_ms = segment_start
            last_text = ""
        if segment_start is not None and current_ms - last_decode_ms >= args.partial_interval_ms:
            start_sample = int(segment_start * 16)
            if args.partial_window_s > 0:
                start_sample = max(
                    start_sample,
                    int(current_ms * 16) - int(args.partial_window_s * 16_000),
                )
            window = audio[start_sample : int(current_ms * 16)]
            text = _generate(model, window, args)
            last_decode_ms = current_ms
            if text and text != last_text:
                last_text = text
                if first_partial is None:
                    first_partial = time.monotonic() - started
                print(f"partial @{current_ms}ms: {text}")
    if vad.current_speech_start is not None:
        text = _generate(model, audio[int(vad.current_speech_start * 16) :], args)
        finals.append(text)
        print(f"final tail: {text}")
    print(f"first_partial={first_partial}")
    print(f"text: {''.join(finals)}")
    print(f"wall={time.monotonic() - started:.3f}s")
    return 0


def _generate(model, audio: np.ndarray, args) -> str:
    kwargs = {"cache": {}, "batch_size": 1}
    if args.language:
        kwargs["language"] = args.language
    if args.hotwords:
        kwargs["hotwords"] = [
            word.strip() for word in args.hotwords.split(",") if word.strip()
        ]
    results = model.generate(input=[audio], **kwargs)
    if not results:
        return ""
    return str(results[0].get("text") or "").strip()


if __name__ == "__main__":
    raise SystemExit(main())
