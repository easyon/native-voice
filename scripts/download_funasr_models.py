#!/usr/bin/env python3
"""Download FunASR model artifacts from ModelScope.

Private deployments should mount these directories into the runtime and set
``engine.funasr.allow_download=false``. This script performs the download
explicitly, outside the service, so a restart never stalls on the network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTORCH_MODELS = {
    # name -> ModelScope repo id
    "fun-asr-nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    "paraformer": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "sensevoice": "iic/SenseVoiceSmall",
}

GGUF_MODELS = {
    # name -> (ModelScope repo id, files)
    "fsmn-vad": ("FunAudioLLM/fsmn-vad-GGUF", ["fsmn-vad.gguf"]),
    "paraformer": ("FunAudioLLM/Paraformer-GGUF", ["paraformer-q8.gguf"]),
    "sensevoice": (
        "FunAudioLLM/SenseVoiceSmall-GGUF",
        ["sensevoice-small-q8.gguf"],
    ),
    "fun-asr-nano": (
        "FunAudioLLM/Fun-ASR-Nano-GGUF",
        ["funasr-encoder-f16.gguf", "qwen3-0.6b-q8_0.gguf"],
    ),
}


def _parse_models(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names or "all" in names:
        return list(PYTORCH_ALIASES) + ["gguf"]
    unknown = [name for name in names if name not in PYTORCH_ALIASES and name != "gguf"]
    if unknown:
        raise SystemExit(
            f"unknown model(s): {', '.join(unknown)}; "
            f"choose from {', '.join(PYTORCH_ALIASES)}, gguf, all"
        )
    return names


PYTORCH_ALIASES = ("fun-asr-nano", "fsmn-vad", "punc", "paraformer", "sensevoice")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="fun-asr-nano,fsmn-vad",
        help="comma list: fun-asr-nano, fsmn-vad, punc, paraformer, sensevoice, gguf, all",
    )
    parser.add_argument("--dest", default="./models", help="output directory")
    parser.add_argument(
        "--revision",
        default="master",
        help="ModelScope revision (default master)",
    )
    args = parser.parse_args()

    try:
        from modelscope import snapshot_download
        try:
            # modelscope >= 1.40
            from modelscope.hub.file_download import (
                model_file_download as file_download,
            )
        except ImportError:  # older modelscope
            from modelscope.hub.file_download import file_download
    except Exception as exc:  # noqa: BLE001
        print(
            "modelscope is required: pip install -e 'packages/asr_runtime[funasr]'",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return 1

    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    selected = _parse_models(args.models)

    for name in selected:
        if name == "gguf":
            continue
        repo = PYTORCH_MODELS[name]
        target = dest / _pytorch_dir(name)
        print(f"[pytorch] {repo} -> {target}")
        snapshot_download(
            repo,
            revision=args.revision,
            local_dir=str(target),
        )

    if "gguf" in selected:
        gguf_dir = dest / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        if set(selected) >= set(PYTORCH_ALIASES) or set(selected) == {"gguf"}:
            wanted = list(GGUF_MODELS)
        else:
            wanted = [name for name in selected if name in GGUF_MODELS]
        for name in dict.fromkeys(wanted):
            repo, files = GGUF_MODELS[name]
            for filename in files:
                print(f"[gguf] {repo}/{filename} -> {gguf_dir / filename}")
                file_download(
                    repo,
                    filename,
                    revision=args.revision,
                    local_dir=str(gguf_dir),
                )

    print("\nDone. Point configs/funasr.*.yaml at these directories:")
    print(f"  model_root: {dest}")
    for name in PYTORCH_ALIASES:
        print(f"  {name}: {dest / _pytorch_dir(name)}")
    print(f"  GGUF files: {dest / 'gguf'}")
    return 0


def _pytorch_dir(name: str) -> str:
    return {
        "fun-asr-nano": "Fun-ASR-Nano-2512",
        "fsmn-vad": "fsmn-vad",
        "punc": "punc",
        "paraformer": "Paraformer",
        "sensevoice": "SenseVoiceSmall",
    }[name]


if __name__ == "__main__":
    raise SystemExit(main())

