"""Serve command."""
from __future__ import annotations

import logging
from pathlib import Path

import uvicorn

from ..api.app import create_app
from ..config import load_config


def run(args) -> None:
    cfg = load_config(args.config)
    if args.host:
        cfg.server.host = args.host
    if args.port:
        cfg.server.port = args.port
    if args.dev_key:
        cfg.auth.dev_key = True
    if getattr(args, "asr_provider", None):
        cfg.engine.asr_provider = args.asr_provider
    if getattr(args, "tts_provider", None):
        cfg.engine.tts_provider = args.tts_provider
    if getattr(args, "llm_gpu_layers", None) is not None:
        cfg.llm.n_gpu_layers = args.llm_gpu_layers
    if getattr(args, "llm_server", None):
        cfg.llm.server = Path(args.llm_server).expanduser()
        if not cfg.llm.server.is_absolute():
            cfg.llm.server = (cfg.root / cfg.llm.server).resolve()
    logging.basicConfig(level=getattr(logging, cfg.logging.level.upper(), logging.INFO))
    llm = None
    if args.backend == "fake":
        from ..backends.fake import FakeLLM

        llm = FakeLLM()
    app = create_app(cfg, backend_kind=args.backend, config_path=args.config, llm=llm)
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="warning")
