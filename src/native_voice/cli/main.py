"""native-voice CLI: serve, manage API keys."""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="native-voice")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the gateway")
    serve.add_argument("--config", default="configs/service.yaml")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--backend", choices=["local", "fake", "worker"], default="local")
    serve.add_argument("--asr-provider", default=None, help="paraformer | qwen3asr")
    serve.add_argument("--tts-provider", default=None, help="cosyvoice3 | qwen3tts")
    serve.add_argument(
        "--llm-gpu-layers",
        type=int,
        default=None,
        help="llama.cpp -ngl: 0 = CPU only, 99 = all layers on GPU (default: config)",
    )
    serve.add_argument(
        "--llm-server",
        default=None,
        help="llama-server binary, e.g. bin/llama/llama-server.exe (CPU) or bin/llama-cuda/...",
    )
    serve.add_argument("--dev-key", action="store_true", help="create and print a dev key if empty")
    serve.set_defaults(func=_serve)

    key = sub.add_parser("key", help="manage API keys")
    key.add_argument("action", choices=["create", "list", "disable", "enable"])
    key.add_argument("--config", default="configs/service.yaml")
    key.add_argument("--name", default="default")
    key.add_argument("--scopes", default="asr,tts,conversation")
    key.add_argument("--rpm", type=int, default=None)
    key.add_argument("--max-concurrent", type=int, default=None)
    key.add_argument("--key-id", default=None)
    key.set_defaults(func=_key)

    args = parser.parse_args(argv)
    args.func(args)


def _serve(args) -> None:
    from .serve import run

    run(args)


def _key(args) -> None:
    from .keys import run

    run(args)


if __name__ == "__main__":  # python -m native_voice.cli.main
    main()
