from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cosyvoice_request_id", default=None
)
_tenant: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cosyvoice_tenant", default=None
)
_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cosyvoice_task_id", default=None
)
_model: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cosyvoice_model", default=None
)
_voice: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cosyvoice_voice", default=None
)
_api: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cosyvoice_api", default=None
)

_FIELDS = {
    "request_id": _request_id,
    "tenant": _tenant,
    "task_id": _task_id,
    "model": _model,
    "voice": _voice,
    "api": _api,
}


@contextmanager
def bind_context(**values: Any) -> Iterator[None]:
    """Bind request/task fields to the current async context.

    Context variables are per-task, so concurrent requests do not bleed into
    each other. The context manager resets the values on exit.
    """
    tokens = []
    for name, value in values.items():
        var = _FIELDS.get(name)
        if var is None or value is None:
            continue
        tokens.append((var, var.set(str(value))))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def context_snapshot() -> dict[str, str]:
    return {
        name: value
        for name, var in _FIELDS.items()
        if (value := var.get()) is not None
    }


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for name, value in context_snapshot().items():
            if not hasattr(record, name):
                setattr(record, name, value)
        return True
