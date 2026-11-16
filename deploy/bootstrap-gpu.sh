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
# Offline bundle built by deploy/prepare-bundle.sh (no network on the target):
#   bash deploy/bootstrap-gpu.sh --bundle /data/bundles/cosyvoice-20261113
#   bash deploy/bootstrap-gpu.sh --bundle ... --no-start --skip-smoke
#
# A bundle built with --venv-from also carries a prebuilt virtualenv. It is
# extracted to BUNDLE_VENV_TARGET and no pip step runs at all, which is the
# fastest repeatable deployment (the venv is byte-identical everywhere):
#   bash deploy/bootstrap-gpu.sh --bundle /data/bundles/cosyvoice-20261113 --no-start
#   VENV=/data/venvs/other bash deploy/bootstrap-gpu.sh --bundle ...
# The base interpreter recorded in the venv must exist at the same path, and
# the extracted venv is re-pointed at this checkout's packages/ automatically.
#
# Reuse an interpreter that already has a compatible CUDA PyTorch (a venv,
# conda env, or a pre-provisioned image). This never creates a venv and never
# replaces torch:
#   bash deploy/bootstrap-gpu.sh --reuse-python /opt/conda/bin/python
#   bash deploy/bootstrap-gpu.sh --reuse-python ... --no-install   # verify only
#
# Mirror override for mainland China (pip and torch wheels):
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#   TORCH_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cu128 \
#     bash deploy/bootstrap-gpu.sh
#
# Defaults keep large assets on /data (override with MODEL_DIR/COSYVOICE_REPO):
#   VENV=$HOME/.venvs/cosyvoice  MODEL_DIR=/data/models/Fun-CosyVoice3-0.5B
#   COSYVOICE_REPO=/data/CosyVoice  MODEL_SOURCE=auto (ModelScope, then hf-mirror)
#   LOAD_VLLM=0 LOAD_TRT=0 (enable with --vllm/--trt after GPU validation)
#   MAX_CONCURRENT=2 (4 when both --vllm and --trt are passed)
#
# Assumes: NVIDIA driver + CUDA 12.x, Python 3.10-3.12, git, network access to
# PyPI and the model mirror (unless --bundle). prepare-bundle.sh compiles the
# sdist-only packages into wheels, so a bundle normally needs no compiler on
# the target; if that build failed, the target needs Python headers and g++
# (bootstrap installs them via apt). For a fully hermetic host use the Docker
# image (docs/ops/tts-runtime-image.md).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# Remember whether the operator set these explicitly: a bundle only fills in
# defaults, it must not override an intentional path.
MODEL_DIR_GIVEN=${MODEL_DIR:-}
COSYVOICE_REPO_GIVEN=${COSYVOICE_REPO:-}
VENV_GIVEN=${VENV:-}
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
MAX_CONCURRENT_GIVEN=${MAX_CONCURRENT:-}
MAX_CONCURRENT=${MAX_CONCURRENT:-2}
# Optional accelerator stack: see requirements-accel.lock. Off until verified
# on the target GPU; enabling turns on vLLM batching and/or the TensorRT flow
# decoder, the vendor-supported fixes for upstream's single CUDA stream.
LOAD_VLLM=${LOAD_VLLM:-0}
LOAD_TRT=${LOAD_TRT:-0}
TRT_CONCURRENT=${TRT_CONCURRENT:-2}
SMOKE_CONCURRENCY=${SMOKE_CONCURRENCY:-0}
# pip honors PIP_INDEX_URL automatically; torch needs an explicit index.
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
BUNDLE=${BUNDLE:-}
REUSE_PYTHON=${REUSE_PYTHON:-}
NO_INSTALL=${NO_INSTALL:-0}
ALLOW_NETWORK=${ALLOW_NETWORK:-0}
SKIP_SMOKE=${SKIP_SMOKE:-0}
START=1
SKIP_MODEL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --no-start) START=0 ;;
    --skip-model) SKIP_MODEL=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --vllm) LOAD_VLLM=1 ;;
    --trt) LOAD_TRT=1 ;;
    --bundle) BUNDLE=${2:?--bundle needs a directory}; shift ;;
    --bundle=*) BUNDLE=${1#--bundle=} ;;
    --reuse-python) REUSE_PYTHON=${2:?--reuse-python needs an interpreter}; shift ;;
    --reuse-python=*) REUSE_PYTHON=${1#--reuse-python=} ;;
    --no-install) NO_INSTALL=1 ;;
    --allow-network) ALLOW_NETWORK=1 ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;p}' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

as_bool() { [ "$1" = "1" ] && printf 'true' || printf 'false'; }

log() { printf '[bootstrap] %s\n' "$*"; }
warn() { printf '[bootstrap] WARNING: %s\n' "$*" >&2; }
die() { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

# Extract a packaged venv and make it usable at $target.
#
# A venv is not relocatable by itself: pyvenv.cfg's `home` and bin/python* point
# at the absolute base interpreter, and the console scripts carry the path of
# the venv they were created in. The base interpreter must therefore exist at
# the recorded path (same conda env / same system python); the venv path itself
# only appears in shebangs and .pth files, which are rewritten here. The two
# repo distributions were installed editable on the bundling host, so they are
# re-pointed at this checkout - an offline, dependency-free local install.
extract_venv() {
  local archive=$1 target=$2 base_home=${BUNDLE_VENV_BASE_HOME:-}
  local source=${BUNDLE_VENV_SOURCE:-}
  if [ -n "$base_home" ]; then
    if [ ! -x "$base_home/python3" ] && [ ! -x "$base_home/python" ]; then
      die "the bundled venv was built against $base_home, which does not exist on this host; install that interpreter at the same path (it supplies torch) and rerun"
    fi
    log "venv base interpreter: $base_home"
  fi
  if [ -x "$target/bin/python" ]; then
    log "reusing the existing venv at $target (remove it to re-extract the bundle's copy)"
  else
    mkdir -p "$target"
    log "extracting $archive -> $target"
    case "$archive" in
      *.zst)
        command -v zstd >/dev/null 2>&1 \
          || die "zstd is required to extract $archive; install zstd or rebuild the bundle with --tar-store"
        zstd -q -dc "$archive" | tar -C "$target" --strip-components=1 -xf - ;;
      *.gz) tar -C "$target" --strip-components=1 -xzf "$archive" ;;
      *)    tar -C "$target" --strip-components=1 -xf "$archive" ;;
    esac
    [ -x "$target/bin/python" ] || die "extracted venv has no $target/bin/python"
  fi
  if [ -n "$source" ] && [ "$source" != "$target" ]; then
    log "rewriting $source -> $target in console scripts and .pth files"
    find "$target/bin" -maxdepth 1 -type f -print0 2>/dev/null \
      | while IFS= read -r -d '' f; do
          head -n 1 "$f" 2>/dev/null | grep -q '^#!' || continue
          sed -i "1s|$source|$target|g" "$f"
        done
    find "$target/lib" -name '*.pth' -type f -print0 2>/dev/null \
      | while IFS= read -r -d '' f; do
          sed -i "s|$source|$target|g" "$f"
        done
  fi
  # Editable installs of packages/{common,tts_runtime} still point at the bundling
  # host's checkout. --no-deps keeps this offline and resolver-free.
  log "re-pointing packages/common and packages/tts_runtime at $ROOT"
  PIP_NO_INDEX=1 "$target/bin/python" -m pip install -q --disable-pip-version-check \
    --no-deps --no-build-isolation \
    -e "$ROOT/packages/common" -e "$ROOT/packages/tts_runtime" \
    || die "could not install this checkout into the extracted venv (is setuptools present in the base interpreter?)"
}

# --- 0. Offline bundle ------------------------------------------------------
WHEELS=""
VENV_ARCHIVE=""
if [ -n "$BUNDLE" ]; then
  [ -d "$BUNDLE" ] || die "bundle directory not found: $BUNDLE"
  BUNDLE=$(cd "$BUNDLE" && pwd -P)
  [ -f "$BUNDLE/manifest.env" ] \
    || die "$BUNDLE/manifest.env missing; build it with deploy/prepare-bundle.sh"
  # shellcheck disable=SC1090
  . "$BUNDLE/manifest.env"
  [ "${BUNDLE_FORMAT:-0}" = "1" ] || die "unsupported bundle format in $BUNDLE/manifest.env"
  if [ -z "$MODEL_DIR_GIVEN" ] && [ "${BUNDLE_HAS_MODEL:-0}" = "1" ]; then
    MODEL_DIR="$BUNDLE/${BUNDLE_MODEL_SUBDIR:-model}"
  fi
  if [ -z "$COSYVOICE_REPO_GIVEN" ] && [ "${BUNDLE_HAS_COSYVOICE:-1}" = "1" ]; then
    COSYVOICE_REPO="$BUNDLE/${BUNDLE_COSYVOICE_SUBDIR:-CosyVoice}"
  fi
  WHEELS="$BUNDLE/wheels"
  log "bundle: repo ${BUNDLE_REPO_COMMIT:-?} cosyvoice ${BUNDLE_COSYVOICE_COMMIT:-?} wheels python ${BUNDLE_WHEEL_PYTHON:-?}"
  [ "${BUNDLE_REPO_DIRTY:-0}" = "1" ] && warn "bundle was built from a repo with uncommitted changes"
  if [ "${BUNDLE_HAS_VENV:-0}" = "1" ]; then
    if [ -n "$REUSE_PYTHON" ]; then
      log "bundle ships a prebuilt venv, but --reuse-python wins; ignoring it"
    else
      VENV_ARCHIVE="$BUNDLE/${BUNDLE_VENV_ARCHIVE:-venv/venv.tar.zst}"
      [ -f "$VENV_ARCHIVE" ] \
        || die "manifest lists a prebuilt venv but $VENV_ARCHIVE is missing; re-copy the bundle"
      if [ -z "$VENV_GIVEN" ]; then
        VENV=${BUNDLE_VENV_TARGET:-/data/venvs/cosyvoice}
      fi
      log "bundle: prebuilt venv (python ${BUNDLE_VENV_PYTHON:-?}) for $VENV; no pip install will run"
    fi
  fi
  if [ "$ALLOW_NETWORK" != "1" ] && [ "$NO_INSTALL" != "1" ] && [ -z "$VENV_ARCHIVE" ]; then
    [ -d "$WHEELS" ] || die "bundle has no wheels directory; rebuild with wheels or pass --allow-network"
    export PIP_NO_INDEX=1
    export PIP_FIND_LINKS="$WHEELS"
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_NO_INPUT=1
    log "offline mode: pip uses only $WHEELS (pass --allow-network to permit downloads)"
  fi
  # Fail before the multi-minute model load when the bundle cannot possibly
  # satisfy the requested accelerator stack.
  if [ "$ALLOW_NETWORK" != "1" ] && [ "$NO_INSTALL" != "1" ]; then
    if [ "$LOAD_VLLM" = "1" ] && [ "${BUNDLE_LOAD_VLLM:-1}" != "1" ]; then
      die "bundle has no vLLM wheels; rebuild with --with-vllm or pass --allow-network"
    fi
    if [ "$LOAD_TRT" = "1" ] && [ "${BUNDLE_LOAD_TRT:-1}" != "1" ]; then
      die "bundle has no TensorRT wheels; rebuild with --with-trt or pass --allow-network"
    fi
  fi
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; install the NVIDIA driver first"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Memory is a sanity check, not a tuning input: for this 0.5B stack the
# binding limit is TTFF/RTF and the flow decoder's serialization, not VRAM.
# The smoke test prints the measured peak; size concurrency from that.
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
  | awk 'NR==1{gsub(/[[:space:]]/,""); print}') || GPU_MEM_MIB=0
case "$GPU_MEM_MIB" in ''|*[!0-9]*) GPU_MEM_MIB=0 ;; esac
if [ "$GPU_MEM_MIB" -gt 0 ] \
   && { [ "$LOAD_VLLM" = "1" ] || [ "$LOAD_TRT" = "1" ]; } \
   && [ "$GPU_MEM_MIB" -lt 16384 ]; then
  warn "GPU reports ${GPU_MEM_MIB} MiB; vLLM+TensorRT on this stack want >=16 GiB (vLLM KV cache + TRT workspace + CUDA context). Verify with the smoke test's peak_gpu_mb"
fi

# --- 1. Python environment -------------------------------------------------
if [ -n "$REUSE_PYTHON" ]; then
  if [ -x "$REUSE_PYTHON" ]; then
    PY=$REUSE_PYTHON
  elif command -v "$REUSE_PYTHON" >/dev/null 2>&1; then
    PY=$(command -v "$REUSE_PYTHON")
  else
    die "reuse interpreter not found or not executable: $REUSE_PYTHON"
  fi
  log "reusing interpreter $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])') ); no venv will be created"
  if [ "$NO_INSTALL" != "1" ]; then
    "$PY" -m pip --version >/dev/null 2>&1 || die "pip is not available in $PY"
  fi
else
  if [ -n "$VENV_ARCHIVE" ] && [ "$NO_INSTALL" != "1" ]; then
    extract_venv "$VENV_ARCHIVE" "$VENV"
    # The extracted venv is the exact environment that was validated; every
    # later step turns into a verification instead of a pip install.
    NO_INSTALL=1
    log "using the bundle's prebuilt venv; all remaining steps verify, none install"
  fi
  if [ "$NO_INSTALL" = "1" ] && [ ! -x "$VENV/bin/python" ]; then
    die "--no-install needs an existing environment; use --reuse-python or create $VENV first"
  fi
  if [ ! -x "$VENV/bin/python" ]; then
    log "creating venv at $VENV"
    "$PYTHON" -m venv "$VENV" || die "python venv failed; Python 3.10+ is required"
  fi
  PY="$VENV/bin/python"
  if [ "$NO_INSTALL" != "1" ]; then
    if [ -n "$BUNDLE" ] && [ "$ALLOW_NETWORK" != "1" ]; then
      "$PY" -m pip install -q --upgrade pip --no-index --find-links "$WHEELS" \
        || warn "could not upgrade pip from the bundle; continuing"
    else
      "$PY" -m pip install -q --upgrade pip
    fi
  fi
fi

if [ "$NO_INSTALL" = "1" ]; then
  "$PY" -c 'import torch' >/dev/null 2>&1 \
    || die "--no-install: torch is not importable in $PY"
elif [ -n "$REUSE_PYTHON" ]; then
  "$PY" -c 'import torch' >/dev/null 2>&1 \
    || die "--reuse-python never installs torch; install a CUDA 12.x PyTorch in that environment first"
  log "reusing torch $("$PY" -c 'import torch; print(torch.__version__)')"
else
  if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
    if [ -n "$BUNDLE" ] && [ "$ALLOW_NETWORK" != "1" ]; then
      log "installing torch/torchaudio from the bundle"
      "$PY" -m pip install -q torch==2.10.0 torchaudio==2.10.0 \
        --no-index --find-links "$WHEELS" \
        || die "torch/torchaudio wheels are missing from $WHEELS"
    else
      log "installing torch/torchaudio (cu128 wheels)"
      "$PY" -m pip install -q torch==2.10.0 torchaudio==2.10.0 \
        --index-url "$TORCH_INDEX_URL"
    fi
  else
    log "reusing the torch already installed in $VENV"
  fi
fi

# The wheel set was resolved for one Python minor version; cp312 wheels do not
# install into 3.11. Only relevant when this run will actually use pip.
if [ -n "$BUNDLE" ] && [ "${BUNDLE_HAS_WHEELS:-0}" = "1" ] \
   && [ "$NO_INSTALL" != "1" ] && [ "$ALLOW_NETWORK" != "1" ]; then
  wheel_py=${BUNDLE_WHEEL_PYTHON:-}
  got_py=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
  if [ -n "$wheel_py" ] && [ "$wheel_py" != "$got_py" ]; then
    die "bundle wheels target Python $wheel_py, but $PY is $got_py; rebuild the bundle with the same Python minor version"
  fi
  if [ -n "${BUNDLE_WHEEL_PLATFORM:-}" ] && [ "${BUNDLE_WHEEL_PLATFORM}" != "$(uname -sm)" ]; then
    die "bundle wheels target ${BUNDLE_WHEEL_PLATFORM}, but this host is $(uname -sm); rebuild on a matching platform"
  fi
fi

"$PY" - <<'PYCHECK' || die "torch/CUDA check failed; install torch 2.10.0+cu128 or fix the driver"
import sys, torch
print(f"[bootstrap] python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"cuda {torch.version.cuda}, cudnn {torch.backends.cudnn.version()}")
assert torch.cuda.is_available(), "torch cannot see CUDA (driver or CPU-only wheel)"
assert torch.version.cuda and torch.version.cuda.startswith("12."), \
    f"torch was built for CUDA {torch.version.cuda}; this stack is verified on cu12x"
PYCHECK

# pyworld (imported by the checkpoint's dataset.processor tags) and
# openai-whisper build native code. Bare-metal hosts often have neither the
# Python headers nor a compiler; install the distro packages when missing.
# --no-install never builds anything, and a bundle whose wheels are all
# prebuilt (prepare-bundle compiles the sdists) needs no toolchain either.
need_build_tools=1
if [ -n "$BUNDLE" ] && [ "${BUNDLE_BINARY_WHEELS:-0}" = "1" ]; then
  need_build_tools=0
  log "bundle ships binary wheels only; skipping Python headers/g++ setup"
fi
if [ "$NO_INSTALL" != "1" ] && [ "$need_build_tools" = "1" ]; then
has_python_headers() {
  "$PY" -c 'import os, sysconfig; raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)' 2>/dev/null
}
if ! has_python_headers; then
  if command -v apt-get >/dev/null 2>&1; then
    pydev=$("$PY" -c 'import sys; print(f"python{sys.version_info[0]}.{sys.version_info[1]}-dev")')
    log "installing $pydev for pyworld's C++ extension"
    apt-get update >/dev/null 2>&1 || true
    apt-get install -y "$pydev" >/dev/null 2>&1 \
      || apt-get install -y python3-dev >/dev/null 2>&1 \
      || die "install $pydev (or python3-dev) and rerun"
    has_python_headers || die "Python.h still missing after installing $pydev"
  else
    die "Python.h not found; install the Python development headers for $("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  fi
fi
if ! command -v g++ >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    log "installing build-essential for native extensions"
    apt-get update >/dev/null 2>&1 || true
    apt-get install -y build-essential >/dev/null 2>&1 || die "install build-essential and rerun"
  else
    die "g++ not found; install a C++ toolchain and rerun"
  fi
fi
fi

if [ "$NO_INSTALL" = "1" ]; then
  # A pre-provisioned image: verify instead of touching pip. The list covers
  # what the runtime imports on the gateway and engine paths.
  log "--no-install: verifying the existing environment"
  "$PY" - <<'PYCHECK' || die "the existing environment is incomplete; install the missing packages or drop --no-install"
import importlib.util

required = [
    "tts_runtime", "torch", "torchaudio", "transformers", "onnxruntime",
    "librosa", "soundfile", "fastapi", "uvicorn", "yaml", "numpy",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing modules: " + ", ".join(missing))
print("[bootstrap] environment imports verified")
PYCHECK
else
log "installing runtime packages"
"$PY" -m pip install -q -e "$ROOT/packages/common" -e "$ROOT/packages/tts_runtime"
# whisper's sdist needs pkg_resources at build time; setuptools>=81 dropped it.
# Install it un-isolated against a pinned setuptools instead of letting pip
# resolve a build environment that cannot build it. See requirements-cosyvoice.lock.
log "installing openai-whisper (no build isolation, setuptools<81)"
"$PY" -m pip install -q "setuptools<81" wheel
"$PY" -m pip install -q --no-build-isolation "openai-whisper==20240930"
log "installing engine dependencies (pinned)"
"$PY" -m pip install -q -r "$ROOT/requirements-cosyvoice.lock"
fi

when_accel() {
  if [ "$LOAD_VLLM" = "1" ]; then
    if [ "$NO_INSTALL" != "1" ]; then
      log "installing vLLM stack (requirements-vllm.lock)"
      "$PY" -m pip install -q -r "$ROOT/requirements-vllm.lock" \
        || die "vllm install failed; it pins torch 2.10.0 and may require a different wheel for this CUDA"
    fi
    "$PY" -c 'import torch, transformers, vllm; print(f"[bootstrap] vllm {vllm.__version__}, torch {torch.__version__}, transformers {transformers.__version__}")' \
      || die "vllm import failed on this GPU/driver"
  fi
  if [ "$LOAD_TRT" = "1" ]; then
    if [ "$NO_INSTALL" != "1" ]; then
      log "installing TensorRT runtime (requirements-trt.lock)"
      "$PY" -m pip install -q -r "$ROOT/requirements-trt.lock" \
        || die "tensorrt install failed"
    fi
    "$PY" -c 'import tensorrt; print(f"[bootstrap] tensorrt {tensorrt.__version__}")' \
      || die "tensorrt import failed"
  fi
}
when_accel

# The console script lives in the interpreter's scripts dir; a reused conda
# env and a venv put it in different places.
SCRIPTS=$("$PY" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')
RUNTIME_BIN="$SCRIPTS/tts-runtime"
if [ ! -x "$RUNTIME_BIN" ]; then
  RUNTIME_BIN=$(command -v tts-runtime 2>/dev/null || true)
fi
if [ -z "$RUNTIME_BIN" ] || [ ! -x "$RUNTIME_BIN" ]; then
  die "tts-runtime entry point not found near $PY; reinstall with pip install -e packages/tts_runtime"
fi
log "runtime entry point: $RUNTIME_BIN"

# --- 2. Upstream CosyVoice source ------------------------------------------
# Point COSYVOICE_REPO at a clone you already have; it is reused as-is.
if [ ! -f "$COSYVOICE_REPO/cosyvoice/cli/cosyvoice.py" ]; then
  if [ -n "$BUNDLE" ] && [ "$ALLOW_NETWORK" != "1" ]; then
    die "bundle has no upstream CosyVoice checkout at $COSYVOICE_REPO; rebuild the bundle or set COSYVOICE_REPO"
  fi
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
    if [ -n "$BUNDLE" ] && [ "$ALLOW_NETWORK" != "1" ]; then
      die "checkpoint at $MODEL_DIR is incomplete and the bundle is offline; rebuild the bundle or pass --allow-network"
    fi
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
# Upstream's single shared CUDA stream is what limits a non-accelerated
# process to ~2 concurrent requests. With vLLM batching the LLM decode and
# TensorRT running several flow contexts, 4 is a better starting point; it is
# still an admission cap, not a promise. Measure with scripts/loadtest.py.
if [ -z "$MAX_CONCURRENT_GIVEN" ] && [ "$LOAD_VLLM" = "1" ] && [ "$LOAD_TRT" = "1" ]; then
  MAX_CONCURRENT=4
  log "vLLM + TRT: defaulting max_concurrent to 4 (override with MAX_CONCURRENT=<n>)"
fi
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
if [ "$SKIP_SMOKE" = "1" ]; then
  log "skipping model smoke test (--skip-smoke)"
else
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
fi

if [ "$START" = "0" ]; then
  log "prepared; start with:"
  log "  COSYVOICE_PYTHON=$PY COSYVOICE_REPO=$COSYVOICE_REPO . $ROOT/scripts/gpu_env.sh"
  log "  $RUNTIME_BIN -c $CONFIG"
  exit 0
fi

log "starting runtime on 0.0.0.0:$PORT (docs: /docs disabled; probes: /healthz /readyz)"
exec "$RUNTIME_BIN" -c "$CONFIG"
