"""FastAPI application factory."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..config.schema import ServiceConfig
from .lifespan import lifespan
from .routers import conversation, health, inference, metrics, tickets, voices


def create_app(
    cfg: ServiceConfig,
    backend_kind: str = "local",
    config_path: str | None = None,
    llm=None,
) -> FastAPI:
    app = FastAPI(
        title="Native Voice",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.backend_kind = backend_kind
    app.state.config_path = config_path
    app.state.injected_llm = llm
    if cfg.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.server.cors_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-DashScope-WorkSpace"],
        )
    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(voices.router)
    app.include_router(tickets.router)
    app.include_router(inference.router)
    app.include_router(conversation.router)

    web_dir = Path(__file__).resolve().parents[3] / "web"
    if web_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse("/ui/")

    return app
