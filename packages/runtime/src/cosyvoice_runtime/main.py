from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from cosyvoice_common.errors import APIError, internal_error
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .api import files, health, http, ws
from .context import bind_context
from .logging_config import configure_logging
from .runtime import Runtime
from .settings import RuntimeConfig, load_config

logger = logging.getLogger("cosyvoice_runtime.main")


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Reject oversized JSON bodies, including chunked requests."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max(1, int(max_bytes))

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                pass

        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        started = False

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _BodyTooLarge:
            if not started:
                await self._reject(send)
            else:
                raise

    @staticmethod
    async def _reject(send: Any) -> None:
        body = json.dumps(
            {
                "code": "RequestTooLarge",
                "message": "request body is too large",
                "retryable": False,
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def create_app(
    config: RuntimeConfig | None = None,
    *,
    config_path: str | Path | None = None,
) -> FastAPI:
    """Build the ASGI app.

    ``config`` wins over ``config_path``; when both are omitted the path comes
    from ``COSYVOICE_RUNTIME_CONFIG`` / ``configs/runtime.yaml``. Tests pass a
    ``RuntimeConfig`` directly, while the CLI loads YAML.
    """
    runtime = Runtime(config or load_config(config_path))
    configure_logging(
        runtime.config.logging,
        service=runtime.config.service_name,
        version=runtime.config.version,
        environment=runtime.config.environment,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ffmpeg and WAV/resampling work runs in this bounded pool. A per-call
        # asyncio.to_thread() would otherwise share the loop's tiny default
        # executor and serialize streaming under load.
        executor = ThreadPoolExecutor(
            max_workers=max(4, int(runtime.config.server.worker_threads)),
            thread_name_prefix="cosyvoice-io",
        )
        asyncio.get_running_loop().set_default_executor(executor)
        try:
            await runtime.startup()
            yield
        finally:
            await runtime.shutdown()
            executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="CosyVoice Runtime",
        version=runtime.config.version,
        description="Bailian-compatible CosyVoice HTTP / SSE / WebSocket gateway",
        lifespan=lifespan,
        docs_url="/docs" if runtime.config.server.docs_enabled else None,
        redoc_url="/redoc" if runtime.config.server.docs_enabled else None,
        openapi_url="/openapi.json" if runtime.config.server.docs_enabled else None,
    )
    app.state.runtime = runtime

    if runtime.config.server.trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=runtime.config.server.trusted_hosts,
        )
    if runtime.config.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=runtime.config.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context_and_access_log(request: Request, call_next):
        raw_request_id = request.headers.get("X-Request-Id")
        request_id = (
            "".join(ch for ch in raw_request_id if ch.isprintable()).strip()[:128]
            if raw_request_id
            else ""
        ) or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.monotonic()
        with bind_context(request_id=request_id):
            response = await call_next(request)
            response.headers.setdefault("X-Request-Id", request_id)
            if runtime.config.logging.access_log:
                logger.info(
                    "access: %s %s status=%s duration_ms=%.2f client=%s",
                    request.method,
                    request.url.path,
                    response.status_code,
                    (time.monotonic() - started) * 1000,
                    request.client.host if request.client else "-",
                )
            return response

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        headers = {}
        if exc.retry_after is not None:
            headers["Retry-After"] = str(exc.retry_after)
        return JSONResponse(
            exc.to_dict(request_id),
            status_code=exc.status_code,
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()))
        detail = first.get("msg", "invalid request")
        message = f"{location}: {detail}" if location else detail
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        return JSONResponse(
            {
                "code": "InvalidParameter",
                "message": message,
                "retryable": False,
                "request_id": request_id,
            },
            status_code=400,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        code = {
            400: "InvalidParameter",
            401: "Unauthorized",
            403: "AccessDenied",
            404: "NotFound",
            405: "MethodNotAllowed",
            413: "RequestTooLarge",
            429: "Throttling",
            503: "ServiceUnavailable",
        }.get(exc.status_code, "InternalError")
        return JSONResponse(
            {
                "code": code,
                "message": str(exc.detail),
                "retryable": exc.status_code >= 500,
                "request_id": request_id,
            },
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception(
            "unhandled error: %s %s",
            request.method,
            request.url.path,
        )
        error = internal_error()
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        return JSONResponse(error.to_dict(request_id), status_code=500)

    app.include_router(health.router)
    app.include_router(http.router)
    app.include_router(ws.router)
    app.include_router(files.router)

    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=runtime.config.server.max_request_bytes,
    )
    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn --factory cosyvoice_runtime.main:app_factory``."""
    return create_app()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cosyvoice-runtime",
        description="Bailian-compatible CosyVoice HTTP/SSE/WebSocket runtime.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="config path (default: $COSYVOICE_RUNTIME_CONFIG or configs/runtime.yaml)",
    )
    parser.add_argument("--host", default=None, help="override server.host")
    parser.add_argument("--port", type=int, default=None, help="override server.port")
    parser.add_argument("--version", action="version", version="cosyvoice-runtime 0.1.0")
    return parser


def run() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    app = create_app(config)
    ssl_kwargs = {}
    if config.server.ssl_certfile and config.server.ssl_keyfile:
        ssl_kwargs = {
            "ssl_certfile": config.server.ssl_certfile,
            "ssl_keyfile": config.server.ssl_keyfile,
        }
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_config=None,
        log_level=config.logging.level.lower(),
        access_log=config.logging.access_log,
        proxy_headers=config.server.proxy_headers,
        forwarded_allow_ips=config.server.forwarded_allow_ips,
        ws_max_size=config.server.max_ws_message_bytes,
        timeout_keep_alive=config.server.keepalive_timeout_s,
        timeout_graceful_shutdown=config.limits.shutdown_timeout_s,
        **ssl_kwargs,
    )


if __name__ == "__main__":
    run()
