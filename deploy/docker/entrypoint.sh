#!/bin/sh
# Container entrypoint: prepare the dynamic loader and PYTHONPATH, then exec
# the requested command (default: tts-runtime).
#
# The image does not bake in the upstream CosyVoice checkout. Mount it
# read-only and point COSYVOICE_REPO at it:
#   -v /data/cvtest:/opt/CosyVoice:ro
#
# Setting PYTHONPATH here keeps the upstream code and third_party/Matcha-TTS
# importable without an editable install, matching how scripts/gpu_smoke.py
# runs on the host.

set -eu

gpu_env="${COSYVOICE_GPU_ENV:-/opt/cosyvoice/gpu_env.sh}"
if [ -r "$gpu_env" ]; then
    # shellcheck disable=SC1090
    . "$gpu_env"
else
    echo "entrypoint: GPU env file not found: $gpu_env" >&2
fi

repo="${COSYVOICE_REPO:-/opt/CosyVoice}"
if [ -d "$repo/cosyvoice" ]; then
    matcha="${COSYVOICE_MATCHA_REPO:-$repo/third_party/Matcha-TTS}"
    if [ -d "$matcha" ]; then
        PYTHONPATH="$repo:$matcha${PYTHONPATH:+:$PYTHONPATH}"
    else
        PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
        echo "entrypoint: Matcha-TTS not found under $matcha" >&2
    fi
    export PYTHONPATH
elif [ "${COSYVOICE_ENGINE_TYPE:-}" = "cosyvoice" ]; then
    echo "entrypoint: COSYVOICE_REPO=$repo does not contain a cosyvoice/ package" >&2
fi

if [ "$#" -eq 0 ]; then
    set -- tts-runtime
fi

# Allow `docker run IMAGE --port 9000` as a shorthand for the CLI.
case "$1" in
    -*) set -- tts-runtime "$@" ;;
esac

exec "$@"
