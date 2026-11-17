"""Run one GPU engine: python -m native_voice.engine.cli asr|tts --config ..."""
from __future__ import annotations

import argparse
import logging

import uvicorn

from ..config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="native-voice-engine")
    parser.add_argument("kind", choices=["asr", "tts"])
    parser.add_argument("--config", default="configs/service.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--provider", default=None, help="override engine.asr_provider / engine.tts_provider")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    logging.basicConfig(level=getattr(logging, cfg.logging.level.upper(), logging.INFO))
    if args.kind == "asr":
        from .asr_server import create_asr_app

        app = create_asr_app(cfg, args.provider)
        port = args.port or 8101
    else:
        from .tts_server import create_tts_app

        app = create_tts_app(cfg, args.provider)
        port = args.port or 8102
    uvicorn.run(app, host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
