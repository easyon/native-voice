#!/usr/bin/env bash
# One-command bare-metal deployment for a single GPU host.
#
# It creates (or reuses) a venv, installs the runtime and the verified engine
# dependency set, fetches the checkpoint, writes a GPU config from the
# maintained template, runs the model smoke test, and starts the service.
#
#   bash deploy/bootstrap-gpu.sh                      # deploy and start
#   bash deploy/bootstrap-gpu.sh --no-start           # prepare only
#   bash deploy/bootstrap-gpu.sh --vllm --trt         # + accelerator stack
#   MODEL_DIR=/data/models/Fun-CosyVoice3-0.5B COSYVOICE_REPO=/data/CosyVoice \
#     bash deploy/bootstrap-gpu.sh
#
# Defaults keep large assets on /data (override with MODEL_DIR/COSYVOICE_REPO):
#   VENV=$HOME/.venvs/cosyvoice  MODEL_DIR=/data/models/Fun-CosyVoice3-0.5B
#   COSYVOICE_REPO=/data/CosyVoice  MODEL_SOURCE=auto (ModelScope, then hf-mirror)
#   LOAD_VLLM=0 LOAD_TRT=0 (enable with --vllm/--trt after GPU validation)
#
# Assumes: NVIDIA driver + CUDA 12.x, Python 3.10-3.12, git, network access to
# PyPI and the model mirror. For air-gapped hosts use the Docker image instead
# (docs/ops/cosyvoice-runtime-image.md).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-$HOME/.venvs/cosyvoice}
PYTHON=${PYTHON:-python3}
PORT=${PORT:-8000}
MODEL_DIR=${MODEL_DIR:-/data/models/Fun-CosyVoice3-0.5B}
MODEL_REPO=${MODEL_REPO:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}
MODEL_SOURCE=${MODEL_SOURCE:-auto}
MODEL_BASE=${MODEL_BASE:-}
COSYVOICE_REPO=${COSYVOICE_REPO:-/data/CosyVoice}
COSYVOICE_URL=${COSYVOICE_URL:-https://github.com/FunAudioLLM/CosyVoice}
CONFIG=${CONFIG:-$ROOT/configs/runtime.local.gpu.yaml}
MAX_CONCURRENT=${MAX_CONCURRENT:-2}
# Optional accelerator stack: see requirements-accel.lock. Off until verified
# on the target GPU; enabling turns on vLLM batching and/or the TensorRT flow
# decoder, the vendor-supported fixes for upstream's single CUDA stream.
LOAD_VLLM=${LOAD_VLLM:-0}
LOAD_TRT=${LOAD_TRT:-0}
TRT_CONCURRENT=${TRT_CONCURRENT:-2}
SMOKE_CONCURRENCY=${SMOKE_CONCURRENCY:-0}
START=1
SKIP_MODEL=0

for arg in "$@"; do
  case "$arg" in
    --no-start) START=0 ;;
    --skip-model) SKIP_MODEL=1 ;;
    --vllm) LOAD_VLLM=1 ;;
    --trt) LOAD_TRT=1 ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;p}' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

as_bool() { [ "$1" = "1" ] && printf 'true' || printf 'false'; }

log() { printf '[bootstrap] %s\n' "$*"; }
die() { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; install the NVIDIA driver first"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# --- 1. Python environment -------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  log "creating venv at $VENV"
  "$PYTHON" -m venv "$VENV" || die "python venv failed; Python 3.10+ is required"
fi
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip

if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  log "installing torch/torchaudio (cu128 wheels)"
  "$PY" -m pip install -q torch==2.10.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128
else
  log "reusing the torch already installed in $VENV"
fi
"$PY" - <<'PYCHECK' || die "torch/CUDA check failed; install torch 2.10.0+cu128 or fix the driver"
import sys, torch
print(f"[bootstrap] python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"cuda {torch.version.cuda}, cudnn {torch.backends.cudnn.version()}")
assert torch.cuda.is_available(), "torch cannot see CUDA (driver or CPU-only wheel)"
assert torch.version.cuda and torch.version.cuda.startswith("12."), \
    f"torch was built for CUDA {torch.version.cuda}; this stack is verified on cu12x"
PYCHECK

log "installing runtime packages"
"$PY" -m pip install -q -e "$ROOT/packages/common" -e "$ROOT/packages/runtime"
log "installing engine dependencies (pinned)"
"$PY" -m pip install -q -r "$ROOT/requirements-cosyvoice.lock"

when_accel() {
  if [ "$LOAD_VLLM" = "1" ]; then
    log "installing vLLM stack (requirements-vllm.lock)"
    "$PY" -m pip install -q -r "$ROOT/requirements-vllm.lock" \
      || die "vllm install failed; it pins torch 2.10.0 and may require a different wheel for this CUDA"
    "$PY" -c 'import torch, transformers, vllm; print(f"[bootstrap] vllm {vllm.__version__}, torch {torch.__version__}, transformers {transformers.__version__}")' \
      || die "vllm import failed on this GPU/driver"
  fi
  if [ "$LOAD_TRT" = "1" ]; then
    log "installing TensorRT runtime (requirements-trt.lock)"
    "$PY" -m pip install -q -r "$ROOT/requirements-trt.lock" \
      || die "tensorrt install failed"
    "$PY" -c 'import tensorrt; print(f"[bootstrap] tensorrt {tensorrt.__version__}")' \
      || die "tensorrt import failed"
  fi
}
when_accel

# --- 2. Upstream CosyVoice source ------------------------------------------
# Point COSYVOICE_REPO at a clone you already have; it is reused as-is.
if [ ! -f "$COSYVOICE_REPO/cosyvoice/cli/cosyvoice.py" ]; then
  command -v git >/dev/null 2>&1 || die "git not found; clone CosyVoice to $COSYVOICE_REPO manually"
  log "cloning CosyVoice into $COSYVOICE_REPO"
  mkdir -p "$(dirname "$COSYVOICE_REPO")"
  git clone --depth 1 "$COSYVOICE_URL" "$COSYVOICE_REPO"
fi
if [ -d "$COSYVOICE_REPO/.git" ] && [ ! -d "$COSYVOICE_REPO/third_party/Matcha-TTS/matcha" ]; then
  log "initializing CosyVoice submodules (Matcha-TTS)"
  git -C "$COSYVOICE_REPO" submodule update --init --recursive --depth 1
fi
export COSYVOICE_REPO
export PYTHONPATH="$COSYVOICE_REPO:$COSYVOICE_REPO/third_party/Matcha-TTS:${PYTHONPATH:-}"

# --- 3. Model assets -------------------------------------------------------
if [ "$SKIP_MODEL" = "0" ]; then
  need_fetch=0
  include_args=()
  if [ ! -f "$MODEL_DIR/llm.pt" ] || [ ! -f "$MODEL_DIR/flow.pt" ]; then
    need_fetch=1
  fi
  if [ "$LOAD_TRT" = "1" ] \
     && [ ! -f "$MODEL_DIR/flow.decoder.estimator.fp32.onnx" ] \
     && ! ls "$MODEL_DIR"/flow.decoder.estimator.fp32.*.plan >/dev/null 2>&1; then
    need_fetch=1
    include_args=(--include flow.decoder.estimator.fp32.onnx)
  fi
  if [ "$need_fetch" = "1" ]; then
    log "fetching $MODEL_REPO -> $MODEL_DIR (source=$MODEL_SOURCE, parallel ranges)"
    base_args=()
    if [ -n "$MODEL_BASE" ]; then base_args=(--base "$MODEL_BASE"); fi
    "$PY" "$ROOT/scripts/fetch_model.py" \
      --repo "$MODEL_REPO" --source "$MODEL_SOURCE" \
      "${base_args[@]}" "${include_args[@]}" --dest "$MODEL_DIR"
  else
    log "model already present at $MODEL_DIR"
  fi
fi
PROMPT_WAV=${PROMPT_WAV:-$COSYVOICE_REPO/asset/zero_shot_prompt.wav}
PROMPT_TEXT=${PROMPT_TEXT:-'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'}
[ -f "$PROMPT_WAV" ] || die "prompt wav not found: $PROMPT_WAV (set PROMPT_WAV)"
[ -f "$MODEL_DIR/speech_tokenizer_v3.onnx" ] || die "checkpoint incomplete: $MODEL_DIR"

# vLLM and TensorRT write their generated artifacts inside the model dir.
if [ "$LOAD_VLLM" = "1" ] && [ ! -d "$MODEL_DIR/vllm" ] && [ ! -w "$MODEL_DIR" ]; then
  die "LOAD_VLLM=1 exports $MODEL_DIR/vllm on first start; make MODEL_DIR writable or pre-export that directory"
fi
if [ "$LOAD_TRT" = "1" ] && ! ls "$MODEL_DIR"/flow.decoder.estimator.fp32.*.plan >/dev/null 2>&1 && [ ! -w "$MODEL_DIR" ]; then
  die "LOAD_TRT=1 builds the plan inside $MODEL_DIR; make it writable or prebuild the plan"
fi

# --- 4. GPU library path ---------------------------------------------------
# gpu_env.sh inspects whichever interpreter it is told about; point it at the
# venv so it finds the torch that the engine will actually import.
COSYVOICE_PYTHON="$PY"
export COSYVOICE_PYTHON
# shellcheck disable=SC1091
. "$ROOT/scripts/gpu_env.sh"

# --- 5. Config from the maintained template --------------------------------
sed -e "s#^  port: 8000#  port: $PORT#" \
    -e "s#artifact: .*#artifact: $MODEL_DIR#" \
    -e "s#prompt_wav: .*#prompt_wav: $PROMPT_WAV#" \
    -e "s#prompt_text: .*#prompt_text: \"$PROMPT_TEXT\"#" \
    -e "s#max_concurrent: [0-9]*#max_concurrent: $MAX_CONCURRENT#" \
    -e "s#^    load_vllm: .*#    load_vllm: $(as_bool "$LOAD_VLLM")#" \
    -e "s#^    load_trt: .*#    load_trt: $(as_bool "$LOAD_TRT")#" \
    -e "s#^    trt_concurrent: [0-9]*#    trt_concurrent: $TRT_CONCURRENT#" \
    "$ROOT/configs/runtime.gpu.yaml" > "$CONFIG"
log "wrote $CONFIG"

# --- 6. Smoke test ---------------------------------------------------------
smoke_args=(
  --model-dir "$MODEL_DIR" --prompt-wav "$PROMPT_WAV"
  --prompt-text "$PROMPT_TEXT" --cache-prompt
  --text '你好，我是你的语音助手，很高兴为你服务。' --runs 1
)
if [ "$LOAD_VLLM" = "1" ]; then smoke_args+=(--vllm); fi
if [ "$LOAD_TRT" = "1" ]; then
  smoke_args+=(--trt --trt-concurrent "$TRT_CONCURRENT")
fi
log "running model smoke test (load + one utterance)"
"$PY" "$ROOT/scripts/gpu_smoke.py" "${smoke_args[@]}" || \
  die "smoke test failed; fix the model/CUDA issue before serving traffic"

if [ "${SMOKE_CONCURRENCY:-0}" -gt 1 ]; then
  log "running concurrency smoke test (c=$SMOKE_CONCURRENCY)"
  "$PY" "$ROOT/scripts/gpu_smoke.py" "${smoke_args[@]}" \
    --concurrency "$SMOKE_CONCURRENCY" \
    --tasks "$((SMOKE_CONCURRENCY * 2))" || \
    die "concurrency smoke test failed"
fi

if [ "$START" = "0" ]; then
  log "prepared; start with: $VENV/bin/cosyvoice-runtime -c $CONFIG"
  exit 0
fi

log "starting runtime on 0.0.0.0:$PORT (docs: /docs disabled; probes: /healthz /readyz)"
exec "$VENV/bin/cosyvoice-runtime" -c "$CONFIG"
