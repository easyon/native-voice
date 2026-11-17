#!/usr/bin/env bash
# Dev gateway. Usage:
#   BACKEND=worker ASR_PROVIDER=paraformer TTS_PROVIDER=qwen3tts scripts/serve_dev.sh
#   LLM_DEVICE=cpu scripts/serve_dev.sh        # llama.cpp -ngl 0 + CPU binary (no LLM VRAM)
#   LLM_DEVICE=gpu scripts/serve_dev.sh        # llama.cpp -ngl 99 + CUDA binary (default)
#
# Provider combos, least compute first (RTX 4070 Laptop 8 GB; see docs/models.md):
#   ASR_PROVIDER=paraformer TTS_PROVIDER=cosyvoice3 scripts/serve_dev.sh     # default, ~3.5 GB, TTS RTF 0.38-0.52
#   ASR_PROVIDER=paraformer TTS_PROVIDER=qwen3tts scripts/serve_dev.sh       # ~3.8 GB, TTS RTF 1.6-1.7 (transformers)
#   ASR_PROVIDER=qwen3asr TTS_PROVIDER=qwen3tts scripts/serve_dev.sh         # ~4.5 GB
#   ASR_PROVIDER=paraformer TTS_PROVIDER=qwen3tts_vllm scripts/serve_dev.sh  # needs deploy/vllm-omni running first; TTS alone 4.5 GB
#   ASR_PROVIDER=paraformer TTS_PROVIDER=cosyvoice3_vllm scripts/serve_dev.sh # same image, CosyVoice3 on :8092 (--profile cosyvoice3)
#   ASR_PROVIDER=qwen3asr TTS_PROVIDER=qwen3tts_vllm scripts/serve_dev.sh    # ~6.1 GB

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
ARGS=(serve --config configs/service.yaml --backend "${BACKEND:-local}" --dev-key)
[ -n "${ASR_PROVIDER:-}" ] && ARGS+=(--asr-provider "$ASR_PROVIDER")
[ -n "${TTS_PROVIDER:-}" ] && ARGS+=(--tts-provider "$TTS_PROVIDER")
case "${LLM_DEVICE:-}" in
  cpu)
    ARGS+=(--llm-gpu-layers 0)
    if [ -x "$ROOT/bin/llama/llama-server.exe" ]; then
      ARGS+=(--llm-server bin/llama/llama-server.exe)
    fi
    ;;
  gpu)
    ARGS+=(--llm-gpu-layers 99)
    if [ -x "$ROOT/bin/llama-cuda/llama-server.exe" ]; then
      ARGS+=(--llm-server bin/llama-cuda/llama-server.exe)
    fi
    ;;
  "")
    ;;
  *)
    echo "LLM_DEVICE must be 'cpu' or 'gpu' (got '$LLM_DEVICE')" >&2
    exit 2
    ;;
esac
if [ -n "${LLM_SERVER:-}" ]; then
  ARGS+=(--llm-server "$LLM_SERVER")
fi
if [ -x "$ROOT/.venv/Scripts/native-voice.exe" ]; then
  exec "$ROOT/.venv/Scripts/native-voice.exe" "${ARGS[@]}" "$@"
fi
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT/.venv/bin/native-voice" ]; then
    exec "$ROOT/.venv/bin/native-voice" "${ARGS[@]}" "$@"
  elif [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON=python
  fi
fi
exec "$PYTHON" -m native_voice.cli.main "${ARGS[@]}" "$@"
