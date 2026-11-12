#!/bin/sh
# Export the CUDA/cuDNN libraries bundled with torch to LD_LIBRARY_PATH.
#
# Usage (must be sourced, not executed):
#   . scripts/gpu_env.sh
#
# Why this exists: onnxruntime-gpu does not link CUDA/cuDNN itself. It dlopen()s
# libcublasLt.so.12 and libcudnn.so.9 at session creation. Those libraries ship
# inside the conda pytorch environment (or, for pip torch wheels, under
# site-packages/nvidia/*/lib). Without this file the engine fails with
# "libcublasLt.so.12: cannot open shared object file" even though torch works.
#
# The script is intentionally POSIX-sh compatible so the container entrypoint
# can source it under /bin/sh. It is a no-op on a CPU-only image.
#
# Overridable:
#   COSYVOICE_PYTHON             interpreter used to locate torch (default: python)
#   COSYVOICE_EXTRA_LIB_DIRS     colon-separated directories prepended first
#   COSYVOICE_GPU_ENV_VERBOSE=1  print the resolved library path

_cv_gpu_put() {
    if [ -z "$1" ] || [ ! -d "$1" ]; then
        return 0
    fi
    case ":$_cv_path:" in
        *":$1:"*) return 0 ;;
    esac
    _cv_path="${_cv_path:+$_cv_path:}$1"
}

_cv_gpu_env() {
    _cv_py="${COSYVOICE_PYTHON:-python}"
    if ! command -v "$_cv_py" >/dev/null 2>&1; then
        _cv_py="python3"
    fi
    if ! command -v "$_cv_py" >/dev/null 2>&1; then
        return 0
    fi

    # -I keeps PYTHONPATH/user-site from changing which torch is inspected. A
    # venv created with --system-site-packages still sees the conda torch.
    _cv_info=$("$_cv_py" -I -c '
import os
import sys
import sysconfig

try:
    import torch
except Exception:
    sys.exit(0)

torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
site = os.path.dirname(torch_dir)
prefix = site.replace(os.sep, "/").split("/lib/python")[0]
print("TORCH_LIB=" + os.path.join(torch_dir, "lib"))
print("TORCH_SP=" + site)
print("PREFIX_LIB=" + os.path.join(prefix, "lib"))
print("PURELIB=" + sysconfig.get_paths()["purelib"])
print("TORCH_VERSION=" + torch.__version__)
' 2>/dev/null) || _cv_info=""

    _cv_torch_lib=$(printf '%s\n' "$_cv_info" | sed -n 's/^TORCH_LIB=//p')
    _cv_torch_sp=$(printf '%s\n' "$_cv_info" | sed -n 's/^TORCH_SP=//p')
    _cv_prefix_lib=$(printf '%s\n' "$_cv_info" | sed -n 's/^PREFIX_LIB=//p')
    _cv_purelib=$(printf '%s\n' "$_cv_info" | sed -n 's/^PURELIB=//p')
    _cv_torch_version=$(printf '%s\n' "$_cv_info" | sed -n 's/^TORCH_VERSION=//p')

    _cv_path=""

    # Append in priority order; the first matching directory wins at load time.
    _cv_old_ifs=$IFS
    IFS=:
    for _cv_dir in ${COSYVOICE_EXTRA_LIB_DIRS:-}; do
        _cv_gpu_put "$_cv_dir"
    done
    IFS=$_cv_old_ifs

    _cv_gpu_put "$_cv_torch_lib"
    for _cv_dir in "$_cv_torch_sp"/nvidia/*/lib; do
        _cv_gpu_put "$_cv_dir"
    done
    _cv_gpu_put "$_cv_prefix_lib"
    for _cv_dir in "$_cv_purelib"/nvidia/*/lib; do
        _cv_gpu_put "$_cv_dir"
    done
    _cv_gpu_put "/usr/local/cuda/lib64"
    _cv_gpu_put "/usr/local/nvidia/lib64"

    if [ -n "$_cv_path" ]; then
        LD_LIBRARY_PATH="${_cv_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export LD_LIBRARY_PATH
    fi

    if [ -z "${CUDA_HOME:-}" ] && [ -d "/usr/local/cuda" ]; then
        CUDA_HOME=/usr/local/cuda
        export CUDA_HOME
    fi

    if [ "${COSYVOICE_GPU_ENV_VERBOSE:-0}" = "1" ]; then
        printf 'gpu_env: torch=%s\n' "${_cv_torch_version:-<not importable>}" >&2
        printf 'gpu_env: LD_LIBRARY_PATH=%s\n' "${_cv_path:-<empty>}" >&2
    fi

    unset _cv_py _cv_info _cv_torch_lib _cv_torch_sp _cv_prefix_lib \
        _cv_purelib _cv_torch_version _cv_path _cv_old_ifs _cv_dir
    return 0
}

_cv_gpu_env
unset -f _cv_gpu_env 2>/dev/null || true
unset -f _cv_gpu_put 2>/dev/null || true
