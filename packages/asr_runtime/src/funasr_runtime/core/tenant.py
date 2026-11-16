from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tenant:
    id: str
    max_concurrent_streams: int | None = None
    requests_per_minute: int | None = None
    audio_seconds_per_minute: int | None = None
