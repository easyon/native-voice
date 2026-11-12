from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from .context import ContextFilter, context_snapshot
from .settings import LoggingConfig


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(context_snapshot())
        for key in ("service", "version", "environment"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(
    config: LoggingConfig,
    *,
    service: str,
    version: str,
    environment: str,
) -> None:
    level = (config.level or "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())
    if config.format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s "
                "[request_id=%(request_id)s tenant=%(tenant)s "
                "task_id=%(task_id)s] %(message)s",
                defaults={
                    "request_id": "-",
                    "tenant": "-",
                    "task_id": "-",
                },
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Uvicorn's default handlers would bypass our formatter and duplicate
    # output. Its access log is replaced by the gateway access middleware.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        if name == "uvicorn.access":
            logger.disabled = not config.access_log

    for name in ("cosyvoice_runtime", "cosyvoice_common"):
        logging.getLogger(name).setLevel(level)

    logging.getLogger("cosyvoice_runtime").info(
        "logging configured: format=%s level=%s service=%s version=%s env=%s",
        config.format,
        level,
        service,
        version,
        environment,
        extra={
            "service": service,
            "version": version,
            "environment": environment,
        },
    )
