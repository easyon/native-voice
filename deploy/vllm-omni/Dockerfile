# Derived image for the Qwen3-TTS streaming server.
#
# The upstream vllm/vllm-omni:nightly image pulled on 2026-11-10 had two
# corrupted/truncated files (the pull was interrupted by a full C: drive):
#
#   1. z3/lib/libz3.so was 4 MiB instead of ~22.6 MiB, which caused
#      SIGBUS on import and "undefined symbol" errors during torch.compile.
#   2. triton/backends/nvidia/bin/ptxas segfaulted, so Triton reported
#      "Cannot find ptxas". The CUDA toolkit's ptxas is intact.
#
# Reinstalling the exact z3-solver version already present in the image
# replaces the truncated library. Pinning to the installed version keeps the
# Python bindings and libz3.so API in sync when the base image is updated.
FROM vllm/vllm-omni:nightly

ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

RUN Z3_VERSION="$(python -c 'import importlib.metadata as m; print(m.version("z3-solver"))')" \
 && echo "Reinstalling z3-solver==${Z3_VERSION}" \
 && uv pip install --system --no-cache-dir --force-reinstall "z3-solver==${Z3_VERSION}" \
 && python -c "import z3; print('z3', z3.get_version_string(), 'OK')"
