# syntax=docker/dockerfile:1.7
#
# Multi-stage image for cosyvoice-runtime.
#
# Default target (last stage) is the CPU gateway, so the existing
# `docker build -t cosyvoice-runtime .` workflow and CI keep working.
# Build the in-process GPU image explicitly:
#
#   docker build --target gpu -t cosyvoice-runtime:0.1.0-gpu .
#   docker build --target gpu --build-arg WITH_ACCEL=1 \
#     -t cosyvoice-runtime:0.1.0-gpu-accel .   # + vLLM / TensorRT
#
# The accelerator stack is off by default: it is the vendor-supported path for
# single-GPU batching (docs/design/cosyvoice-runtime.md 8.5.10) but must be
# validated on the target GPU, and it changes transformers (4.51.3 -> 4.57.1).
# The image still ships no model weights or generated artifacts; prebuild
# <artifact>/vllm and the TensorRT plan, because models are mounted read-only.
#
# The GPU stage deliberately does NOT pip-install torch. It creates a conda
# environment with pytorch 2.10.0 + pytorch-cuda 12.8 (the stack validated on
# A100/CUDA 12.8), then a Python 3.12 venv on top of it with
# --system-site-packages, matching the host that was verified. All Python
# libraries are pinned in requirements-runtime.lock / requirements-cosyvoice.lock.
#
# Pin the base images by digest in production, e.g.
#   --build-arg CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:...
#   --build-arg PYTHON_IMAGE=python:3.12-slim@sha256:...

ARG PYTHON_IMAGE=python:3.12-slim
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ARG MICROMAMBA_VERSION=1.5.10

# ---------------------------------------------------------------------------
# Build the two local wheels once; both final images install the same files.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS wheelbuilder

WORKDIR /src
COPY packages/common ./packages/common
COPY packages/runtime ./packages/runtime

RUN python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels \
        ./packages/common ./packages/runtime

# ---------------------------------------------------------------------------
# GPU: conda torch 2.10.0 + CUDA 12.8, then a venv with the locked pip deps.
# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS gpu-torch

ARG MICROMAMBA_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# bzip2/ca-certificates/curl are needed to bootstrap micromamba. ffmpeg is used
# by the gateway for mp3/resampling; libsndfile1 backs the soundfile loader.
# build-essential compiles pyworld, which upstream imports at load time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bzip2 \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Pinned micromamba. For an air-gapped build, mirror this tarball and pass
# --build-arg MICROMAMBA_URL=https://registry.internal/.../micromamba.tar.bz2.
ARG MICROMAMBA_URL=""
RUN url="${MICROMAMBA_URL:-https://micro.mamba.pm/api/micromamba/linux-64/${MICROMAMBA_VERSION}}"; \
    curl -fsSL "${url}" | tar -xj -C /usr/local bin/micromamba

COPY deploy/docker/conda-torch.yaml /tmp/conda-torch.yaml
RUN micromamba create -y -p /opt/conda/envs/torch -f /tmp/conda-torch.yaml \
    && micromamba clean --all --yes

# Venv on top of the conda interpreter. torch/torchaudio are inherited from
# conda and are never installed by pip; the assertion below fails the build if
# anything replaced them.
RUN /opt/conda/envs/torch/bin/python -m venv --system-site-packages /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

COPY requirements-runtime.lock requirements-cosyvoice.lock /tmp/
# openai-whisper's sdist imports pkg_resources at build time, which
# setuptools>=81 removed. Pin build tooling and build it un-isolated; it is
# excluded from requirements-cosyvoice.lock for that reason.
RUN pip install --no-cache-dir "setuptools<81" wheel \
    && pip install --no-cache-dir --no-build-isolation "openai-whisper==20240930" \
    && pip install --no-cache-dir -r /tmp/requirements-runtime.lock \
    && pip install --no-cache-dir -r /tmp/requirements-cosyvoice.lock

COPY --from=wheelbuilder /wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/*.whl

# Fail fast on a wrong torch/CUDA/onnxruntime combination. This runs without a
# GPU: it validates versions and that the shared libraries load.
COPY scripts/gpu_env.sh /opt/cosyvoice/gpu_env.sh
RUN python -c "import torch, torchaudio, onnxruntime as ort, transformers as tf, soundfile; \
assert torch.__version__.split('+')[0] == '2.10.0', torch.__version__; \
assert (torch.version.cuda or '') == '12.8', torch.version.cuda; \
assert torchaudio.__version__.split('+')[0] == '2.10.0', torchaudio.__version__; \
assert ort.__version__ == '1.20.2', ort.__version__; \
assert tf.__version__ == '4.51.3', tf.__version__; \
assert 'CUDAExecutionProvider' in ort.get_available_providers(), ort.get_available_providers()"

# onnxruntime-gpu does not link the CUDA/cuDNN libraries it loads at runtime;
# gpu_env.sh exports the torch/conda library directories. Verify that the
# loader can resolve the sonames that ORT 1.20.2's CUDA EP needs.
RUN . /opt/cosyvoice/gpu_env.sh \
    && for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 \
                    libcufft.so.11 libcurand.so.10 libcudnn.so.9; do \
        python -c "import ctypes, sys; ctypes.CDLL(sys.argv[1])" "$lib" || exit 1; \
    done

# ---------------------------------------------------------------------------
# GPU runtime image. Model assets and the upstream CosyVoice checkout are
# mounted at runtime, never baked into the image.
# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS gpu

# HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE keep the model path hermetic: the
# checkpoint must be complete on disk. Override to 0 only for debugging.
# The cache variables keep numba/torch/matplotlib writable under a read-only
# root filesystem.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    COSYVOICE_RUNTIME_CONFIG=/app/configs/runtime.gpu.yaml \
    COSYVOICE_REPO=/opt/CosyVoice \
    COSYVOICE_GPU_ENV=/opt/cosyvoice/gpu_env.sh \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/mpl \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    TORCH_HOME=/tmp/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=gpu-torch /opt/conda/envs/torch /opt/conda/envs/torch
COPY --from=gpu-torch /opt/venv /opt/venv

# Optional accelerator stack (vLLM continuous batching + TensorRT flow
# decoder). Enabled with --build-arg WITH_ACCEL=1. Requirements are installed
# into the venv; torch stays the conda 2.10.0+cu128 build, which satisfies
# vllm's torch==2.10.0 pin, so pip does not replace it. Version assertions in
# gpu-torch stay strict for the baseline image; this stage prints what it got.
ARG WITH_ACCEL=0
COPY requirements-vllm.lock requirements-trt.lock requirements-accel.lock /tmp/
RUN if [ "$WITH_ACCEL" = "1" ]; then \
        pip install --no-cache-dir -r /tmp/requirements-accel.lock \
        && python -c "import importlib.metadata as m; \
            print('vllm', m.version('vllm'), 'tensorrt', m.version('tensorrt-cu12'))"; \
    fi \
    && rm -f /tmp/requirements-vllm.lock /tmp/requirements-trt.lock \
             /tmp/requirements-accel.lock

WORKDIR /app

COPY configs ./configs
COPY scripts/gpu_env.sh /opt/cosyvoice/gpu_env.sh
COPY deploy/docker/entrypoint.sh /usr/local/bin/cosyvoice-entrypoint

# Mount points: /models is read-only model storage, /opt/CosyVoice is the
# read-only upstream checkout (CosyVoice + third_party/Matcha-TTS).
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && chmod 0755 /usr/local/bin/cosyvoice-entrypoint /opt/cosyvoice/gpu_env.sh \
    && mkdir -p /app/data/audio /models /data/models /data/cvtest /data/prompts \
        /opt/CosyVoice /tmp/.cache \
    && chown -R app:app /app /tmp/.cache

USER 10001:10001

EXPOSE 8000
STOPSIGNAL SIGTERM

# /readyz returns 200 only after the engine has preloaded and is not draining.
# Model loading can take minutes on a cold node; Kubernetes should still use
# the chart's startupProbe instead of relying on this Docker healthcheck.
HEALTHCHECK --interval=15s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(os.environ.get('COSYVOICE_HEALTHCHECK_URL', 'http://127.0.0.1:8000/readyz'), timeout=4)"

ENTRYPOINT ["/usr/local/bin/cosyvoice-entrypoint"]
CMD ["cosyvoice-runtime"]

# ---------------------------------------------------------------------------
# CPU gateway image (default target). No torch, no CUDA, mock engine or a
# remote model service.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    COSYVOICE_RUNTIME_CONFIG=/app/configs/runtime.yaml \
    COSYVOICE_GPU_ENV=/opt/cosyvoice/gpu_env.sh

WORKDIR /app

# ffmpeg is used for mp3/opus encoding, resampling and pitch shifting.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.lock /tmp/requirements-runtime.lock
RUN pip install --no-cache-dir -r /tmp/requirements-runtime.lock

COPY --from=wheelbuilder /wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/*.whl

COPY configs ./configs
COPY scripts/gpu_env.sh /opt/cosyvoice/gpu_env.sh
COPY deploy/docker/entrypoint.sh /usr/local/bin/cosyvoice-entrypoint

# Non-root runtime. The local audio store and /tmp must remain writable when
# readOnlyRootFilesystem=true is used in Kubernetes.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && chmod 0755 /usr/local/bin/cosyvoice-entrypoint /opt/cosyvoice/gpu_env.sh \
    && mkdir -p /app/data/audio /tmp/.cache \
    && chown -R app:app /app /tmp/.cache

USER 10001:10001

EXPOSE 8000
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(os.environ.get('COSYVOICE_HEALTHCHECK_URL', 'http://127.0.0.1:8000/readyz'), timeout=2)"

ENTRYPOINT ["/usr/local/bin/cosyvoice-entrypoint"]
CMD ["cosyvoice-runtime"]
