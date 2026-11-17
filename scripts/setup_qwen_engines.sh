#!/usr/bin/env bash
# Create isolated venvs for the optional Qwen3 ASR/TTS engines (Linux dev).
#   scripts/setup_qwen_engines.sh tts --download
#   scripts/setup_qwen_engines.sh asr --download
#   scripts/setup_qwen_engines.sh both
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KIND="${1:-both}"
DOWNLOAD="${2:-}"
UV="${UV:-uv}"
PY_VERSION="${PY_VERSION:-3.10}"

new_env() {
  local name="$1"
  local dir="$ROOT/envs/$name"
  local py="$dir/bin/python"
  if [ ! -x "$py" ]; then
    "$UV" venv "$dir" --python "$PY_VERSION"
  fi
  "$UV" pip install --python "$py" -e "$ROOT"
  # cu121 keeps compatibility with the 8 GB laptop driver; pip prefers the
  # +cu121 local version over the CPU wheel from PyPI.
  "$UV" pip install --python "$py" "torch==2.5.1" "torchaudio==2.5.1" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu121
  echo "$py"
}

if [ "$KIND" = "tts" ] || [ "$KIND" = "both" ]; then
  PY="$(new_env qwen3tts)"
  "$UV" pip install --python "$PY" \
    "transformers==4.57.3" "accelerate==1.12.0" \
    "librosa>=0.10,<0.11" "soundfile>=0.12" "sox" "onnxruntime>=1.18,<1.20" "einops>=0.8,<1"
  "$UV" pip install --python "$PY" "qwen-tts==0.1.1" --no-deps
fi

if [ "$KIND" = "asr" ] || [ "$KIND" = "both" ]; then
  PY="$(new_env qwen3asr)"
  "$UV" pip install --python "$PY" \
    "transformers==4.57.6" "accelerate==1.12.0" \
    "nagisa==0.2.11" "soynlp==0.0.493" "librosa>=0.10,<0.11" "soundfile>=0.12"
  "$UV" pip install --python "$PY" "qwen-asr==0.0.6" --no-deps
fi

if [ "$DOWNLOAD" = "--download" ]; then
  MS="$ROOT/.venv/bin/modelscope"
  [ -x "$MS" ] || MS="$ROOT/.venv/Scripts/modelscope.exe"
  [ -x "$MS" ] || { echo "modelscope CLI not found in .venv; run uv sync --extra gpu first" >&2; exit 1; }
  if [ "$KIND" = "tts" ] || [ "$KIND" = "both" ]; then
    "$MS" download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
      --local_dir "$ROOT/models/Qwen3-TTS-12Hz-0.6B-Base"
  fi
  if [ "$KIND" = "asr" ] || [ "$KIND" = "both" ]; then
    "$MS" download --model Qwen/Qwen3-ASR-0.6B \
      --local_dir "$ROOT/models/Qwen3-ASR-0.6B"
  fi
fi

echo "done. select with:"
echo "  BACKEND=worker ASR_PROVIDER=qwen3asr TTS_PROVIDER=qwen3tts scripts/serve_dev.sh"
