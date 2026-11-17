"""Backwards-compatible import; the implementation moved to utils.async_stream."""
from ...utils.async_stream import iter_in_thread

__all__ = ["iter_in_thread"]
