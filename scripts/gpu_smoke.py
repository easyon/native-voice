#!/usr/bin/env python3
"""GPU acceptance smoke test for a local CosyVoice3 checkpoint.

Runs entirely outside the gateway so a model/CUDA problem is not confused with
an API problem. It loads the official CosyVoice class, synthesizes one
utterance through the same streaming path the runtime uses, and reports the
metrics that matter for real-time calls: first-chunk latency, RTF, and audio
duration.

Example (on the GPU host)::

    COSYVOICE_REPO=/data/CosyVoice \
    python scripts/gpu_smoke.py \
        --model-dir /data/models/Fun-CosyVoice3-0.5B \
        --prompt-wav /data/CosyVoice/asset/zero_shot_prompt.wav \
        --prompt-text 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。' \
        --text '你好，我是你的语音助手，很高兴为你服务。' \
        --out /tmp/smoke.wav

Batch/accelerator comparison (run the same command with and without
``--vllm``/``--trt`` and compare aggregate audio seconds per wall second)::

    python scripts/gpu_smoke.py ... --vllm --concurrency 4 --tasks 16
"""

from __future__ import annotations

import argparse
import concurrent.futures
import inspect
import json
import os
import statistics
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", default="你好，我是你的语音助手，很高兴为你服务。")
    parser.add_argument(
        "--prompt-wav",
        default="",
        help="reference audio; when omitted the checkpoint's SFT voices are used",
    )
    parser.add_argument("--prompt-text", default="", help="transcript of --prompt-wav")
    parser.add_argument("--spk-id", default="")
    parser.add_argument(
        "--cache-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="register zero-shot prompt features once, as the runtime does",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="load the LLM through vLLM (continuous batching); "
        "same flag as engine.cosyvoice.load_vllm",
    )
    parser.add_argument(
        "--trt",
        action="store_true",
        help="compile/use the TensorRT flow decoder estimator; "
        "same flag as engine.cosyvoice.load_trt",
    )
    parser.add_argument("--trt-concurrent", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="parallel requests; >1 is how batching gains are measured",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=0,
        help="total requests in concurrency mode (default concurrency*2)",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="TF32 matmuls (matches the runtime default)",
    )
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = os.environ.get("COSYVOICE_REPO", "")
    if repo:
        sys.path.insert(0, repo)
        sys.path.insert(0, str(Path(repo) / "third_party" / "Matcha-TTS"))

    import numpy as np
    import torch
    from cosyvoice.cli.cosyvoice import CosyVoice3

    try:
        from cosyvoice_runtime.engines.cosyvoice_compat import (
            install_wav_loader,
            install_wetext_assets,
        )

        install_wav_loader()
        install_wetext_assets()
    except ImportError:
        # Running the upstream checkout without the runtime installed: the
        # script still works when torchaudio.TorchCodec is available.
        pass

    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "error": "CUDA is not available"}))
        return 2

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    load_kwargs = {
        "fp16": args.fp16,
        "load_vllm": args.vllm,
        "load_trt": args.trt,
        "trt_concurrent": args.trt_concurrent,
    }
    params = inspect.signature(CosyVoice3).parameters
    if args.vllm and "load_vllm" not in params:
        print(json.dumps({"ok": False, "error": "this CosyVoice build has no load_vllm"}))
        return 2
    if args.trt and "load_trt" not in params:
        print(json.dumps({"ok": False, "error": "this CosyVoice build has no load_trt"}))
        return 2
    load_kwargs = {k: v for k, v in load_kwargs.items() if k in params}

    load_start = time.perf_counter()
    model = CosyVoice3(args.model_dir, **load_kwargs)
    load_s = time.perf_counter() - load_start
    sample_rate = int(model.sample_rate)

    if not args.prompt_wav and not args.spk_id:
        voices = []
        try:
            voices = model.list_available_spks()
        except Exception:  # noqa: BLE001 - optional for zero-shot checkpoints
            pass
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "need --prompt-wav or --spk-id",
                    "sft_voices": voices,
                }
            )
        )
        return 2

    zero_shot_spk_id = ""
    if args.cache_prompt and args.prompt_wav:
        zero_shot_spk_id = "smoke-v1"
        register_start = time.perf_counter()
        model.add_zero_shot_spk(args.prompt_text, args.prompt_wav, zero_shot_spk_id)
        print(
            json.dumps(
                {
                    "prompt_cache_register_s": round(
                        time.perf_counter() - register_start, 3
                    )
                }
            )
        )

    def one() -> dict:
        chunks: list[np.ndarray] = []
        first_chunk_s: float | None = None
        start = time.perf_counter()
        try:
            if args.prompt_wav:
                generator = model.inference_zero_shot(
                    args.text,
                    args.prompt_text,
                    args.prompt_wav,
                    zero_shot_spk_id=zero_shot_spk_id,
                    stream=args.stream,
                )
            else:
                generator = model.inference_sft(
                    args.text, args.spk_id, stream=args.stream
                )
            for output in generator:
                now = time.perf_counter()
                if first_chunk_s is None:
                    first_chunk_s = now - start
                speech = output["tts_speech"]
                if hasattr(speech, "detach"):
                    speech = speech.detach().cpu().float().numpy()
                chunks.append(np.asarray(speech, dtype=np.float32).reshape(-1))
        except Exception as exc:  # noqa: BLE001 - reported in the summary
            return {"error": f"{type(exc).__name__}: {exc}"}
        elapsed = time.perf_counter() - start
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return {
            "first_chunk_s": round(first_chunk_s or 0.0, 4),
            "elapsed_s": round(elapsed, 4),
            "audio_s": round(audio.size / sample_rate, 4),
            "audio": audio,
        }

    results: list[dict] = []
    concurrency = max(1, args.concurrency)
    tasks = args.tasks if args.tasks > 0 else concurrency * 2
    if concurrency == 1:
        for run in range(args.runs):
            torch.cuda.reset_peak_memory_stats()
            result = one()
            result["run"] = run
            result["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if "error" not in result:
                result["rtf"] = (
                    round(result["elapsed_s"] / result["audio_s"], 4)
                    if result["audio_s"]
                    else None
                )
            results.append(result)
            if args.out and run == 0 and "audio" in result:
                import soundfile as sf

                sf.write(args.out, result.pop("audio"), sample_rate)
    else:
        torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency
        ) as pool:
            results = list(pool.map(lambda _: one(), range(tasks)))
        wall_s = time.perf_counter() - wall_start
        peak_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        errors = [r["error"] for r in results if "error" in r]
        done = [r for r in results if "error" not in r]
        for result in done:
            result.pop("audio", None)
        ttff = [r["first_chunk_s"] * 1000 for r in done]
        rtfs = [r["elapsed_s"] / r["audio_s"] for r in done if r["audio_s"]]
        total_audio = sum(r["audio_s"] for r in done)
        summary = {
            "ok": not errors,
            "mode": "concurrent",
            "concurrency": concurrency,
            "tasks": tasks,
            "failures": len(errors),
            "errors": errors[:5],
            "wall_s": round(wall_s, 4),
            "aggregate_audio_per_wall_s": round(total_audio / wall_s, 4)
            if wall_s
            else None,
            "ttff_ms_mean": round(statistics.fmean(ttff), 1) if ttff else None,
            "ttff_ms_p50": round(statistics.median(ttff), 1) if ttff else None,
            "rtf_mean": round(statistics.fmean(rtfs), 4) if rtfs else None,
            "peak_gpu_mb": peak_mb,
        }
        print(
            json.dumps(
                {
                    **summary,
                    "model_dir": args.model_dir,
                    "fp16": args.fp16,
                    "vllm": args.vllm,
                    "trt": args.trt,
                    "trt_concurrent": args.trt_concurrent,
                    "tf32": args.tf32,
                    "sample_rate": sample_rate,
                    "load_s": round(load_s, 3),
                },
                ensure_ascii=False,
            )
        )
        return 0 if summary["ok"] else 1

    for result in results:
        result.pop("audio", None)
    print(
        json.dumps(
            {
                "ok": all("error" not in r for r in results),
                "model_dir": args.model_dir,
                "fp16": args.fp16,
                "vllm": args.vllm,
                "trt": args.trt,
                "trt_concurrent": args.trt_concurrent,
                "tf32": args.tf32,
                "cache_prompt": args.cache_prompt,
                "sample_rate": sample_rate,
                "load_s": round(load_s, 3),
                "runs": results,
            },
            ensure_ascii=False,
        )
    )
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
