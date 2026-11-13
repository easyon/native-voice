#!/usr/bin/env bash
# Build an offline deployment bundle: checkpoint + upstream CosyVoice + wheels.
#
#   bash deploy/prepare-bundle.sh --dest /data/bundles/cosyvoice-20261113
#   bash deploy/prepare-bundle.sh --dest ... --with-trt --with-vllm
#   bash deploy/prepare-bundle.sh --dest ... --all          # + accelerator stacks
#   bash deploy/prepare-bundle.sh --dest ... --no-wheels   # assets only
#   bash deploy/prepare-bundle.sh --dest ... \
#     --model-dir /data/models/Fun-CosyVoice3-0.5B         # reuse a warm model dir
#
# Fast path for repeated deployments: package an already validated venv, so a
# target host extracts it instead of running pip at all (typically minutes
# instead of tens of minutes, and no wheelhouse is needed):
#
#   bash deploy/prepare-bundle.sh --dest /data/bundles/cosyvoice-20261113 \
#     --venv-from /data/venvs/cosyvoice --no-wheels
#
# The venv must have been built for the same Python minor version and against a
# CUDA torch that exists at the same path on the target (usually the shared
# conda env). The bundle records the base interpreter so bootstrap can refuse
# an incompatible host instead of half-working.
#
# The bundle is copied as a directory (or --tar archive) to a target host:
#
#   tar -C /data/bundles -czf cosyvoice-bundle.tar.gz cosyvoice-20261113
#   scp cosyvoice-bundle.tar.gz target:/data/bundles/
#   bash /data/bundles/cosyvoice-20261113/repo/deploy/bootstrap-gpu.sh \
#     --bundle /data/bundles/cosyvoice-20261113 --no-start
#
# Layout:
#   model/          fetch_model.py output (llm/flow/vocoder + ONNX for TRT)
#   CosyVoice/      upstream checkout at a pinned commit + Matcha-TTS submodule
#   wheels/         every pip distribution the target needs (Linux, same Python)
#   venv/           optional prebuilt virtualenv (--venv-from) + packages.txt
#   repo/           this repository (so the target needs no git clone)
#   manifest.env    versions/commits consumed by bootstrap-gpu.sh --bundle
#
# Constraints:
#   * Run this on Linux x86_64 with the SAME Python major.minor as the target.
#     pip resolves wheels for the interpreter running this script; cp312 wheels
#     do not install into 3.11.
#   * Build-from-source packages (openai-whisper, pyworld) are compiled into
#     binary wheels when this host has a toolchain; the target then installs
#     them like any other wheel and needs no g++/Python headers. If the build
#     fails the sdist stays in the bundle and the target compiles it instead.
#   * The TensorRT plan is GPU-architecture specific and is built on the target
#     (or prebuilt per GPU type). The vLLM export is written to model/vllm on
#     first start and needs a writable model directory.
#
# Networked prep host required. On the target, bootstrap --bundle uses
# PIP_NO_INDEX and never downloads unless --allow-network is passed.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=""
PYTHON=${PYTHON:-python3}
MODEL_REPO=${MODEL_REPO:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}
MODEL_SOURCE=${MODEL_SOURCE:-auto}
MODEL_BASE=${MODEL_BASE:-}
# Copy an existing checkpoint directory instead of downloading it. Useful when
# it already contains the TensorRT plan and the vLLM export.
MODEL_SRC=${MODEL_SRC:-}
COSYVOICE_URL=${COSYVOICE_URL:-https://github.com/FunAudioLLM/CosyVoice}
# Pinned in requirements-cosyvoice.lock; keep the bundle reproducible.
COSYVOICE_REF=${COSYVOICE_REF:-074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
WITH_MODEL=1
WITH_COSYVOICE=1
WITH_WHEELS=1
WITH_REPO=1
WITH_TORCH_WHEELS=1
WITH_VLLM=0
WITH_TRT=0
DO_CHECKSUM=0
MAKE_TAR=0
DRY_RUN=0
# Prebuilt-venv fast path. VENV_SRC is the directory to package; VENV_TARGET is
# where bootstrap-gpu.sh will extract it on the target host.
VENV_SRC=${VENV_SRC:-}
VENV_TARGET=${VENV_TARGET:-/data/venvs/cosyvoice}
WHEELS_GIVEN=${WHEELS_GIVEN:-0}
TAR_STORE=${TAR_STORE:-0}

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST=${2:?--dest needs a directory}; shift ;;
    --dest=*) DEST=${1#--dest=} ;;
    --python) PYTHON=${2:?--python needs an interpreter}; shift ;;
    --python=*) PYTHON=${1#--python=} ;;
    --model-repo) MODEL_REPO=${2:?}; shift ;;
    --model-source) MODEL_SOURCE=${2:?}; shift ;;
    --model-base) MODEL_BASE=${2:?}; shift ;;
    --model-dir) MODEL_SRC=${2:?--model-dir needs a directory}; shift ;;
    --model-dir=*) MODEL_SRC=${1#--model-dir=} ;;
    --cosyvoice-url) COSYVOICE_URL=${2:?}; shift ;;
    --cosyvoice-ref) COSYVOICE_REF=${2:?}; shift ;;
    --torch-index-url) TORCH_INDEX_URL=${2:?}; shift ;;
    --no-model) WITH_MODEL=0 ;;
    --no-cosyvoice) WITH_COSYVOICE=0 ;;
    --no-wheels) WITH_WHEELS=0 ;;
    --with-wheels) WITH_WHEELS=1; WHEELS_GIVEN=1 ;;
    --no-repo) WITH_REPO=0 ;;
    --no-torch-wheels) WITH_TORCH_WHEELS=0 ;;
    --venv-from) VENV_SRC=${2:?--venv-from needs a venv directory}; shift ;;
    --venv-from=*) VENV_SRC=${1#--venv-from=} ;;
    --venv-target) VENV_TARGET=${2:?--venv-target needs a path}; shift ;;
    --venv-target=*) VENV_TARGET=${1#--venv-target=} ;;
    --tar-store) TAR_STORE=1 ;;
    --with-vllm) WITH_VLLM=1 ;;
    --with-trt) WITH_TRT=1 ;;
    --all) WITH_VLLM=1; WITH_TRT=1 ;;
    --checksum) DO_CHECKSUM=1 ;;
    --tar) MAKE_TAR=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;p}' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

log() { printf '[bundle] %s\n' "$*"; }
warn() { printf '[bundle] WARNING: %s\n' "$*" >&2; }
die() { printf '[bundle] ERROR: %s\n' "$*" >&2; exit 1; }

[ -n "$DEST" ] || die "--dest is required (put it outside the repository)"
# A prebuilt venv already contains every distribution, so the wheelhouse is
# redundant unless the operator explicitly asks for both.
if [ -n "$VENV_SRC" ] && [ "$WHEELS_GIVEN" = "0" ]; then
  WITH_WHEELS=0
fi
if [ -z "$VENV_SRC" ] && [ "$WITH_WHEELS" = "0" ] \
   && { [ "$WITH_VLLM" = "1" ] || [ "$WITH_TRT" = "1" ]; }; then
  warn "--with-vllm/--with-trt have no effect with --no-wheels"
fi
for cmd in git tar; do
  command -v "$cmd" >/dev/null 2>&1 || die "$cmd is required"
done
if [ "$WITH_WHEELS" = "1" ]; then
  command -v "$PYTHON" >/dev/null 2>&1 || die "python not found: $PYTHON"
  "$PYTHON" -m pip --version >/dev/null 2>&1 \
    || die "$PYTHON has no pip; install pip or pass --python"
fi
if [ -n "$VENV_SRC" ]; then
  [ -d "$VENV_SRC" ] || die "--venv-from is not a directory: $VENV_SRC"
  [ -x "$VENV_SRC/bin/python" ] || die "$VENV_SRC is not a virtualenv (no bin/python)"
  [ -f "$VENV_SRC/pyvenv.cfg" ] \
    || die "$VENV_SRC has no pyvenv.cfg; a conda env cannot be copied this way (copy the whole prefix instead)"
  case "$VENV_TARGET" in
    /*) ;;
    *) die "--venv-target must be an absolute path: $VENV_TARGET" ;;
  esac
fi

mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd -P)
case "$DEST/" in
  "$ROOT/"*) die "bundle destination must be outside the repository: $DEST" ;;
esac
mkdir -p "$DEST"

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '[bundle] dry-run:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

# pip download is separate because it needs the wheel directory and options.
pip_download() {
  local dest=$1
  shift
  run "$PYTHON" -m pip download -q --disable-pip-version-check -d "$dest" "$@"
}

# whisper publishes only an sdist, and its build imports pkg_resources, which
# setuptools>=81 removed. A metadata build during `pip download` can therefore
# fail on a prep host whose setuptools is too new. Try pip first; on failure
# fetch the exact sdist from PyPI's JSON API and verify its published sha256.
fetch_whisper_sdist() {
  local dest=$1
  if [ "$DRY_RUN" = "1" ]; then
    run "$PYTHON" -m pip download -q --no-deps -d "$dest" "openai-whisper==20240930"
    return 0
  fi
  if [ "$PYTHON" -m pip download -q --disable-pip-version-check \
      -d "$dest" "openai-whisper==20240930"; then
    return 0
  fi
  warn "pip could not download the openai-whisper sdist; fetching it directly"
  "$PYTHON" - "$dest" <<'PY'
import hashlib
import json
import os
import sys
import urllib.request

dest = sys.argv[1]
version = "20240930"
url = f"https://pypi.org/pypi/openai-whisper/{version}/json"
with urllib.request.urlopen(url, timeout=60) as resp:
    info = json.load(resp)
sdist = next((u for u in info["urls"] if u["packagetype"] == "sdist"), None)
if sdist is None:
    raise SystemExit("no sdist published for openai-whisper")
target = os.path.join(dest, sdist["filename"])
with urllib.request.urlopen(sdist["url"], timeout=300) as resp, open(target, "wb") as out:
    while True:
        block = resp.read(1024 * 1024)
        if not block:
            break
        out.write(block)
digest = hashlib.sha256()
with open(target, "rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
expected = (sdist.get("digests") or {}).get("sha256", "")
if expected and digest.hexdigest() != expected:
    os.remove(target)
    raise SystemExit(f"sha256 mismatch for {target}")
print(f"[bundle] downloaded {target}")
PY
  # pip never resolved the sdist's dependencies in this path; fetch the ones
  # the other locks do not already cover.
  "$PYTHON" -m pip download -q --disable-pip-version-check --no-deps -d "$dest" \
    tiktoken triton numba more-itertools tqdm \
    || die "could not download whisper's runtime dependencies"
}

# Compile the sdist-only packages into binary wheels so the target installs
# them like any other wheel and needs no compiler. Uses a disposable venv
# seeded from the bundle itself (same numpy as the runtime). Failure is not
# fatal: the sdist stays and bootstrap falls back to building on the target.
build_binary_wheels() {
  local wheels=$1
  local sdists=() f
  # sdist filenames keep the project spelling ("openai-whisper-...",
  # "antlr4-python3-runtime-..."); accept both separators.
  for f in "$wheels"/pyworld-*.tar.gz "$wheels"/openai[-_]whisper-*.tar.gz \
           "$wheels"/antlr4[-_]python3[-_]runtime-*.tar.gz; do
    if [ -f "$f" ]; then sdists+=("$f"); fi
  done
  [ "${#sdists[@]}" -gt 0 ] || return 0
  if [ "$DRY_RUN" = "1" ]; then
    log "dry-run: build binary wheels for ${#sdists[@]} sdist-only package(s)"
    return 0
  fi
  local venv="$DEST/.build-venv" bpy
  rm -rf "$venv"
  "$PYTHON" -m venv "$venv" \
    || { warn "cannot create a wheel-build venv; target will compile the sdists"; return 0; }
  bpy="$venv/bin/python"
  if ! "$bpy" -m pip install -q --disable-pip-version-check --no-index \
       --find-links "$wheels" "setuptools<81" wheel numpy==1.26.4; then
    warn "cannot seed the wheel-build venv from the bundle; target will compile the sdists"
    rm -rf "$venv"
    return 0
  fi
  if "$bpy" -m pip wheel -q --disable-pip-version-check --no-build-isolation \
       --no-deps -w "$wheels" "${sdists[@]}"; then
    local name
    # Wheel filenames normalize dashes to underscores; sdist names do not.
    for name in pyworld openai_whisper antlr4_python3_runtime; do
      if compgen -G "$wheels/$name-*.whl" >/dev/null; then
        rm -f "$wheels/$name"-*.tar.gz \
              "$wheels/${name//_/-}"-*.tar.gz
      fi
    done
    log "built binary wheels for the sdist-only packages"
  else
    warn "sdist build failed (missing g++ or Python headers?); target will compile them"
  fi
  rm -rf "$venv"
}

log "destination: $DEST"
[ "$DRY_RUN" = "1" ] && log "dry run: nothing will be downloaded or written except directories"

if [ "$WITH_MODEL" = "1" ]; then
  mkdir -p "$DEST/model"
  if [ -n "$MODEL_SRC" ]; then
    [ -d "$MODEL_SRC" ] || die "--model-dir is not a directory: $MODEL_SRC"
    case "$(cd "$MODEL_SRC" && pwd -P)/" in
      "$DEST/"*|"$ROOT/"*) die "--model-dir must be outside the bundle and the repository: $MODEL_SRC" ;;
    esac
    case "$DEST/" in
      "$(cd "$MODEL_SRC" && pwd -P)/"*) die "--dest must not be inside --model-dir: $DEST" ;;
    esac
    for f in llm.pt flow.pt speech_tokenizer_v3.onnx; do
      [ -f "$MODEL_SRC/$f" ] || die "source checkpoint missing $f: $MODEL_SRC"
    done
    if [ "$WITH_TRT" = "1" ] \
       && [ ! -f "$MODEL_SRC/flow.decoder.estimator.fp32.onnx" ] \
       && ! ls "$MODEL_SRC"/flow.decoder.estimator.fp32.*.plan >/dev/null 2>&1; then
      die "load_trt needs flow.decoder.estimator.fp32.onnx or a prebuilt .plan in $MODEL_SRC"
    fi
    log "checkpoint: copying the warm directory $MODEL_SRC"
    # The upstream plan path is literally flow.decoder.estimator.fp32.mygpu.plan:
    # the filename does not record the GPU model, so a plan built on another
    # card would be copied silently and only fail at TensorRT deserialization.
    if ls "$MODEL_SRC"/flow.decoder.estimator.fp32.*.plan >/dev/null 2>&1; then
      warn "copying a prebuilt TensorRT .plan; it only loads on the exact GPU model,"
      warn "MIG slice and TensorRT version that produced it. Rebuild the plan on the"
      warn "target host (or its identical GPU) and use this copy only for that fleet."
    fi
    if [ "$DRY_RUN" = "1" ]; then
      log "dry-run: tar -C $MODEL_SRC -cf - . | tar -C $DEST/model -xf -"
    else
      ( cd "$MODEL_SRC" && tar -cf - . ) | tar -C "$DEST/model" -xf -
    fi
  else
    include_args=()
    if [ "$WITH_TRT" = "1" ]; then
      include_args=(--include flow.decoder.estimator.fp32.onnx)
    fi
    base_args=()
    if [ -n "$MODEL_BASE" ]; then
      base_args=(--base "$MODEL_BASE")
    fi
    log "checkpoint: $MODEL_REPO (source=$MODEL_SOURCE)"
    run "$PYTHON" "$ROOT/scripts/fetch_model.py" \
      --repo "$MODEL_REPO" --source "$MODEL_SOURCE" \
      "${base_args[@]}" "${include_args[@]}" --dest "$DEST/model"
  fi
  if [ "$DRY_RUN" = "0" ]; then
    for f in llm.pt flow.pt speech_tokenizer_v3.onnx; do
      [ -f "$DEST/model/$f" ] || die "checkpoint incomplete: $DEST/model/$f is missing"
    done
    if [ "$WITH_TRT" = "1" ] \
       && [ ! -f "$DEST/model/flow.decoder.estimator.fp32.onnx" ] \
       && ! ls "$DEST/model"/flow.decoder.estimator.fp32.*.plan >/dev/null 2>&1; then
      die "load_trt needs flow.decoder.estimator.fp32.onnx or a prebuilt .plan in $DEST/model"
    fi
  fi
else
  log "skipping checkpoint (--no-model)"
fi

if [ "$WITH_COSYVOICE" = "1" ]; then
  if [ -f "$DEST/CosyVoice/cosyvoice/cli/cosyvoice.py" ]; then
    log "reusing existing upstream checkout at $DEST/CosyVoice"
  else
    log "cloning upstream CosyVoice (ref=${COSYVOICE_REF:-default branch})"
    run git clone --depth 1 "$COSYVOICE_URL" "$DEST/CosyVoice"
    if [ -n "$COSYVOICE_REF" ]; then
      if ! run git -C "$DEST/CosyVoice" fetch --depth 1 origin "$COSYVOICE_REF"; then
        die "cannot fetch $COSYVOICE_REF from $COSYVOICE_URL; use a full clone or a reachable mirror"
      fi
      run git -C "$DEST/CosyVoice" checkout -q FETCH_HEAD
    fi
    run git -C "$DEST/CosyVoice" submodule update --init --recursive --depth 1
  fi
  if [ "$DRY_RUN" = "0" ]; then
    [ -f "$DEST/CosyVoice/cosyvoice/cli/cosyvoice.py" ] \
      || die "upstream checkout incomplete: cosyvoice/cli/cosyvoice.py missing"
    [ -d "$DEST/CosyVoice/third_party/Matcha-TTS/matcha" ] \
      || warn "Matcha-TTS submodule missing; HyperPyYAML will fail to resolve matcha.utils"
  fi
else
  log "skipping upstream CosyVoice (--no-cosyvoice)"
fi

if [ "$WITH_WHEELS" = "1" ]; then
  mkdir -p "$DEST/wheels"
  # The wheel set is tied to the interpreter that runs pip download. Record it
  # so bootstrap can refuse an incompatible target instead of failing mid-pip.
  WHEEL_PYTHON=$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
  log "wheels: runtime + engine stack for Python $WHEEL_PYTHON ($(uname -sm))"
  pip_download "$DEST/wheels" -r "$ROOT/requirements-runtime.lock"
  if [ "$WITH_TORCH_WHEELS" = "1" ]; then
    log "wheels: torch/torchaudio from $TORCH_INDEX_URL"
    pip_download "$DEST/wheels" \
      torch==2.10.0 torchaudio==2.10.0 --index-url "$TORCH_INDEX_URL"
  fi
  pip_download "$DEST/wheels" -r "$ROOT/requirements-cosyvoice.lock"
  # pip itself, setuptools<81 and wheel are needed to build whisper/pyworld
  # offline; whisper has no wheel, so keep its sdist too.
  pip_download "$DEST/wheels" --no-deps pip "setuptools<81" wheel
  fetch_whisper_sdist "$DEST/wheels"
  # scripts/loadtest.py is used for on-host acceptance and imports httpx.
  pip_download "$DEST/wheels" httpx
  if [ "$WITH_VLLM" = "1" ]; then
    log "wheels: vLLM stack"
    pip_download "$DEST/wheels" -r "$ROOT/requirements-vllm.lock"
  fi
  if [ "$WITH_TRT" = "1" ]; then
    log "wheels: TensorRT stack"
    pip_download "$DEST/wheels" -r "$ROOT/requirements-trt.lock"
  fi
  build_binary_wheels "$DEST/wheels"
  if [ "$DRY_RUN" = "0" ]; then
    count=$(find "$DEST/wheels" -maxdepth 1 -type f | wc -l)
    [ "$count" -gt 0 ] || die "no distributions were downloaded into $DEST/wheels"
    if [ "$DO_CHECKSUM" = "1" ]; then
      log "writing wheels/SHA256SUMS"
      "$PYTHON" - "$DEST/wheels" <<'PY'
import hashlib
import os
import sys

root = sys.argv[1]
lines = []
for name in sorted(os.listdir(root)):
    path = os.path.join(root, name)
    if not os.path.isfile(path) or name == "SHA256SUMS":
        continue
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    lines.append(f"{digest.hexdigest()}  {name}")
with open(os.path.join(root, "SHA256SUMS"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
PY
    fi
  fi
else
  log "skipping wheels (--no-wheels)"
fi

VENV_BASE_HOME=""
VENV_PYVER=""
VENV_ARCHIVE=""
VENV_HAS_VLLM=0
VENV_HAS_TRT=0
if [ -n "$VENV_SRC" ]; then
  VENV_SRC=$(cd "$VENV_SRC" && pwd -P)
  case "$VENV_SRC/" in
    "$ROOT/"*) die "--venv-from must be outside the repository (it is a build artifact)" ;;
    "$DEST/"*) die "--venv-from must not be inside the bundle destination" ;;
  esac
  # pyvenv.cfg records the base interpreter a --system-site-packages venv leans
  # on. Copying the venv to a host without that base cannot work, so record it
  # and let bootstrap refuse early instead of failing on `import torch`.
  VENV_BASE_HOME=$(sed -n 's/^home *= *//p' "$VENV_SRC/pyvenv.cfg" | head -n 1)
  VENV_PYVER=$("$VENV_SRC/bin/python" \
    -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  log "venv: packaging $VENV_SRC (python $VENV_PYVER, base ${VENV_BASE_HOME:-?}) for $VENV_TARGET"
  mkdir -p "$DEST/venv"
  if [ "$DRY_RUN" = "0" ]; then
    # Audit trail: diff two bundles' package sets without extracting them.
    "$VENV_SRC/bin/python" -m pip list --local --format=freeze \
      > "$DEST/venv/packages.txt" 2>/dev/null || true
    # Record which accelerators the venv actually contains, so bootstrap can
    # reject --vllm/--trt instead of starting an engine that cannot import.
    grep -qi '^vllm==' "$DEST/venv/packages.txt" && VENV_HAS_VLLM=1 || VENV_HAS_VLLM=0
    grep -qi '^tensorrt-cu12==' "$DEST/venv/packages.txt" && VENV_HAS_TRT=1 || VENV_HAS_TRT=0
    log "venv: contents include vllm=$VENV_HAS_VLLM tensorrt=$VENV_HAS_TRT"
    if [ "$TAR_STORE" = "1" ]; then
      archive="$DEST/venv/venv.tar"
      log "venv: archiving without compression (--tar-store)"
      tar -C "$(dirname "$VENV_SRC")" -cf "$archive" "$(basename "$VENV_SRC")"
    elif command -v zstd >/dev/null 2>&1; then
      archive="$DEST/venv/venv.tar.zst"
      log "venv: archiving with zstd (multithreaded)"
      tar -C "$(dirname "$VENV_SRC")" -cf - "$(basename "$VENV_SRC")" \
        | zstd -q -3 -T0 -o "$archive"
    else
      archive="$DEST/venv/venv.tar.gz"
      log "venv: archiving with gzip (install zstd for a faster archive)"
      tar -C "$(dirname "$VENV_SRC")" -czf "$archive" "$(basename "$VENV_SRC")"
    fi
    VENV_ARCHIVE=${archive#"$DEST/"}
    [ -s "$archive" ] || die "venv archive is empty: $archive"
    log "venv: $(du -h "$archive" | cut -f1) at $archive"
  fi
fi

if [ "$WITH_REPO" = "1" ]; then
  mkdir -p "$DEST/repo"
  log "copying this repository (without .git/.venv/models/data) into repo/"
  if [ "$DRY_RUN" = "0" ]; then
    tar -C "$ROOT" \
      --exclude=./.git --exclude=./.venv --exclude=./venv \
      --exclude=./node_modules --exclude=./models --exclude=./data \
      --exclude='*/__pycache__' --exclude='*/.pytest_cache' \
      --exclude='*/.ruff_cache' --exclude='*/.mypy_cache' \
      --exclude='*.egg-info' --exclude='*.pyc' --exclude=./dist --exclude=./build \
      -cf - . | tar -C "$DEST/repo" -xf -
    [ -f "$DEST/repo/deploy/bootstrap-gpu.sh" ] \
      || die "repository copy incomplete: deploy/bootstrap-gpu.sh missing"
  fi
else
  log "skipping repository copy (--no-repo)"
fi

if [ "$DRY_RUN" = "0" ]; then
  if [ "$DO_CHECKSUM" = "1" ]; then
    log "writing bundle SHA256SUMS (model hashing may take minutes)"
    "$PYTHON" - "$DEST" <<'PY'
import hashlib
import os
import sys

root = sys.argv[1]
lines = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if d not in {".git"})
    for name in sorted(filenames):
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, root)
        if rel in {"SHA256SUMS", "manifest.env"} or rel.endswith("SHA256SUMS"):
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {rel}")
with open(os.path.join(root, "SHA256SUMS"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print(f"[bundle] checksummed {len(lines)} files")
PY
  fi

  repo_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
  if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null | head -n 1)" ]; then
    repo_dirty=1
  else
    repo_dirty=0
  fi
  cosy_commit=""
  if [ -d "$DEST/CosyVoice/.git" ]; then
    cosy_commit=$(git -C "$DEST/CosyVoice" rev-parse HEAD 2>/dev/null || true)
  fi
  # Prefix every key with BUNDLE_ so bootstrap can source this file without
  # clobbering the operator's environment (MODEL_REPO, LOAD_VLLM, ...).
  # Use %q so values containing spaces (WHEEL_PLATFORM) survive sourcing.
  emit() { printf '%s=%q\n' "$1" "$2"; }
  {
    emit BUNDLE_FORMAT 1
    emit BUNDLE_CREATED_UTC "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    emit BUNDLE_REPO_COMMIT "$repo_commit"
    emit BUNDLE_REPO_DIRTY "$repo_dirty"
    emit BUNDLE_MODEL_REPO "$MODEL_REPO"
    emit BUNDLE_MODEL_SOURCE "$([ -n "$MODEL_SRC" ] && printf local || printf '%s' "$MODEL_SOURCE")"
    emit BUNDLE_MODEL_SUBDIR model
    emit BUNDLE_COSYVOICE_SUBDIR CosyVoice
    emit BUNDLE_COSYVOICE_COMMIT "${cosy_commit:-unknown}"
    emit BUNDLE_WHEEL_PYTHON "${WHEEL_PYTHON:-}"
    emit BUNDLE_WHEEL_PLATFORM "$(uname -sm)"
    emit BUNDLE_TORCH_INDEX_URL "$TORCH_INDEX_URL"
    emit BUNDLE_LOAD_VLLM "$(if [ -n "$VENV_SRC" ]; then
      printf '%s' "$VENV_HAS_VLLM"
    elif [ "$WITH_WHEELS" = "1" ]; then
      printf '%s' "$WITH_VLLM"
    else
      printf 0
    fi)"
    emit BUNDLE_LOAD_TRT "$(if [ -n "$VENV_SRC" ]; then
      printf '%s' "$VENV_HAS_TRT"
    elif [ "$WITH_WHEELS" = "1" ]; then
      printf '%s' "$WITH_TRT"
    else
      printf 0
    fi)"
    emit BUNDLE_HAS_MODEL "$WITH_MODEL"
    emit BUNDLE_HAS_COSYVOICE "$WITH_COSYVOICE"
    emit BUNDLE_HAS_WHEELS "$WITH_WHEELS"
    emit BUNDLE_HAS_VENV "$([ -n "$VENV_ARCHIVE" ] && printf 1 || printf 0)"
    emit BUNDLE_VENV_ARCHIVE "$VENV_ARCHIVE"
    emit BUNDLE_VENV_TARGET "$VENV_TARGET"
    emit BUNDLE_VENV_PYTHON "$VENV_PYVER"
    emit BUNDLE_VENV_BASE_HOME "$VENV_BASE_HOME"
    emit BUNDLE_VENV_SOURCE "$VENV_SRC"
    # 1 when every distribution in wheels/ is a wheel: bootstrap can then skip
    # installing Python headers and g++ on the target.
    if [ "$WITH_WHEELS" = "1" ] && [ "$DRY_RUN" = "0" ] \
       && ! compgen -G "$DEST/wheels/*.tar.gz" >/dev/null \
       && ! compgen -G "$DEST/wheels/*.zip" >/dev/null; then
      emit BUNDLE_BINARY_WHEELS 1
    else
      emit BUNDLE_BINARY_WHEELS 0
    fi
  } > "$DEST/manifest.env"
  log "wrote $DEST/manifest.env"

  if [ "$MAKE_TAR" = "1" ]; then
    # zstd is several times faster than gzip on multi-GB checkpoint files.
    base=$(basename "$DEST")
    if command -v zstd >/dev/null 2>&1; then
      arch="$DEST.tar.zst"
      log "archiving to $arch (zstd, multithreaded)"
      if ! ZSTD_CLEVEL=${ZSTD_CLEVEL:-3} ZSTD_NBTHREADS=0 \
           tar -C "$(dirname "$DEST")" --zstd -cf "$arch" "$base"; then
        warn "tar --zstd failed; falling back to gzip"
        rm -f "$arch"
        arch="$DEST.tar.gz"
        tar -C "$(dirname "$DEST")" -czf "$arch" "$base"
      fi
    else
      arch="$DEST.tar.gz"
      log "archiving to $arch (gzip; install zstd for a faster archive)"
      tar -C "$(dirname "$DEST")" -czf "$arch" "$base"
    fi
    log "archive: $arch"
  fi
fi

log "bundle complete"
log "next: copy $DEST to the target and run"
log "  bash $DEST/repo/deploy/bootstrap-gpu.sh --bundle $DEST --no-start"
log "  # with a pre-provisioned interpreter:"
log "  bash $DEST/repo/deploy/bootstrap-gpu.sh --bundle $DEST --no-start \\"
log "    --reuse-python /path/to/venv/bin/python"
