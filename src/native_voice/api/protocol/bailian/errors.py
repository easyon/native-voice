"""Protocol error types. Codes mirror the Bailian task-failed vocabulary."""
from __future__ import annotations


class ProtocolError(Exception):
    """Malformed envelope/action; the connection may be closed."""


class TaskError(Exception):
    """A task-level failure that must be reported as task-failed."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
