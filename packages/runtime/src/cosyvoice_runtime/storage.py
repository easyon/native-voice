from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from .settings import StorageConfig

logger = logging.getLogger("cosyvoice_runtime.storage")


def sign_local_file(filename: str, expires_at: int, secret: str) -> str:
    message = f"{filename}:{expires_at}".encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


@dataclass(slots=True)
class StoredAudio:
    url: str
    id: str
    expires_at: int


class AudioStore(ABC):
    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    @abstractmethod
    async def save(self, data: bytes, fmt: str, request_id: str) -> StoredAudio | None:
        raise NotImplementedError


class NullAudioStore(AudioStore):
    async def save(self, data: bytes, fmt: str, request_id: str) -> StoredAudio | None:
        return None


class LocalAudioStore(AudioStore):
    """Local filesystem store with TTL cleanup and HMAC-signed URLs.

    This is suitable for single-node/on-prem deployments. For multi-replica
    gateways use S3/MinIO and presigned URLs so every pod can read the object.
    """

    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        self._dir = Path(config.dir).resolve()
        self._cleanup_task: asyncio.Task[None] | None = None

    async def startup(self) -> None:
        await asyncio.to_thread(self._dir.mkdir, parents=True, exist_ok=True)
        if self.config.ttl_s > 0 and self.config.cleanup_interval_s > 0:
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(),
                name="audio-store-cleanup",
            )

    async def shutdown(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def save(self, data: bytes, fmt: str, request_id: str) -> StoredAudio | None:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", request_id) or uuid.uuid4().hex
        ext = "pcm" if fmt == "pcm" else fmt
        filename = f"audio_{safe_id}.{ext}"
        path = self._dir / filename
        temp = self._dir / f".{filename}.{uuid.uuid4().hex}.tmp"
        await asyncio.to_thread(self._write_atomic, path, temp, data)
        expires_at = int(time.time()) + int(self.config.ttl_s)
        return StoredAudio(
            url=self._url(filename, expires_at),
            id=f"audio_{safe_id}",
            expires_at=expires_at,
        )

    @staticmethod
    def _write_atomic(path: Path, temp: Path, data: bytes) -> None:
        temp.write_bytes(data)
        os.replace(temp, path)

    def _url(self, filename: str, expires_at: int) -> str:
        base = f"{self.config.base_url.rstrip('/')}/files/{quote(filename)}"
        if not self.config.signing_secret:
            return base
        signature = sign_local_file(filename, expires_at, self.config.signing_secret)
        return f"{base}?exp={expires_at}&sig={signature}"

    async def _cleanup_loop(self) -> None:
        interval = max(1.0, float(self.config.cleanup_interval_s))
        while True:
            await asyncio.sleep(interval)
            try:
                removed = await asyncio.to_thread(self._cleanup_once)
                if removed:
                    logger.info("removed %s expired audio files", removed)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - cleanup must not kill the process
                logger.exception("audio cleanup failed")

    def _cleanup_once(self) -> int:
        cutoff = time.time() - max(0, int(self.config.ttl_s))
        removed = 0
        for path in self._dir.glob("audio_*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                logger.debug("could not remove expired file: %s", path)
        return removed


def create_store(config: StorageConfig) -> AudioStore:
    if config.type == "none":
        return NullAudioStore()
    if config.type == "local":
        return LocalAudioStore(config)
    raise ValueError(f"unknown storage type: {config.type!r}; supported: local, none")
