from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_FIELDS = {
    "request_id": "request_id",
    "tenant": "tenant",
    "task_id": "task_id",
    "model": "model",
    "api": "api",
}

_values: dict[str, contextvars.ContextVar[str]] = {
    name: contextvars.ContextVar(name, default="") for name in _FIELDS
}


@contextmanager
def bind_context(**values: Any) -> Iterator[None]:
    tokens = []
    for name, value in values.items():
        variable = _values.get(name)
        if variable is not None and value is not None:
            tokens.append((variable, variable.set(str(value))))
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def context_snapshot() -> dict[str, str]:
    return {
        output: _values[name].get()
        for name, output in _FIELDS.items()
        if _values[name].get()
    }


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        snapshot = context_snapshot()
        for name in _FIELDS.values():
            if not hasattr(record, name):
                setattr(record, name, snapshot.get(name, ""))
        return True
