from __future__ import annotations

import json
import logging
import sys
from typing import Any

from .context import ContextFilter, context_snapshot
from .settings import LoggingConfig


class JsonFormatter(logging.Formatter):
    def __init__(self, *, service: str, version: str, environment: str) -> None:
        super().__init__()
        self._static: dict[str, Any] = {
            "service": service,
            "version": version,
            "environment": environment,
        }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            **self._static,
            **context_snapshot(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extras = getattr(record, "extra_fields", None)
        if isinstance(extras, dict):
            payload.update(extras)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(
    config: LoggingConfig,
    *,
    service: str,
    version: str,
    environment: str,
) -> None:
    level = getattr(logging, config.level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    if config.format == "json":
        handler.setFormatter(
            JsonFormatter(
                service=service,
                version=version,
                environment=environment,
            )
        )
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s "
                "request_id=%(request_id)s tenant=%(tenant)s task_id=%(task_id)s "
                "%(message)s"
            )
        )
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    # uvicorn installs its own handlers; route them through the same formatter.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers[:] = []
        logger.propagate = True
    for name in ("funasr_runtime", "cosyvoice_common"):
        logging.getLogger(name).setLevel(level)
