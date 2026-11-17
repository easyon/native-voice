"""Remote ASR session: one internal WebSocket per utterance."""
from __future__ import annotations

import json

import numpy as np
import websockets

from ..base import ASRBackend, ASRSession


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


class WorkerASRSession(ASRSession):
    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self.ws = None

    async def _ensure(self) -> None:
        if self.ws is None:
            self.ws = await websockets.connect(self.url, extra_headers=_headers(self.token), max_size=None)
            await self.ws.send(json.dumps({"op": "open"}))
            ready = json.loads(await self.ws.recv())
            if ready.get("op") != "ready":
                raise RuntimeError(f"ASR engine handshake failed: {ready}")

    async def reset(self, seed: np.ndarray | None = None) -> None:
        await self._ensure()
        await self.ws.send(json.dumps({"op": "reset"}))
        await self.ws.recv()
        if seed is not None and len(seed):
            await self.feed(seed)

    async def feed(self, audio: np.ndarray) -> str:
        await self._ensure()
        await self.ws.send(np.asarray(audio, dtype="<f4").tobytes())
        msg = json.loads(await self.ws.recv())
        if msg.get("op") == "error":
            raise RuntimeError(msg.get("message", "ASR engine error"))
        return str(msg.get("text", ""))

    async def finalize(self) -> str:
        await self._ensure()
        await self.ws.send(json.dumps({"op": "finalize"}))
        msg = json.loads(await self.ws.recv())
        return str(msg.get("text", ""))

    async def close(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.send(json.dumps({"op": "close"}))
            except Exception:
                pass
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None


class WorkerASRBackend(ASRBackend):
    def __init__(self, url: str, token: str = "", http_url: str = ""):
        self.url = url
        self.token = token
        self.http_url = http_url.rstrip("/")
        self._ready = False

    async def start(self) -> None:
        if self.http_url:
            import httpx

            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(self.http_url + "/readyz")
                    if resp.status_code != 200 or not resp.json().get("ready"):
                        raise RuntimeError(f"ASR engine not ready: {resp.status_code}")
            except Exception as exc:
                raise RuntimeError(f"ASR engine unreachable at {self.http_url}: {exc!r}") from exc
        self._ready = True

    async def close(self) -> None:
        self._ready = False

    def create_session(self) -> ASRSession:
        return WorkerASRSession(self.url, self.token)

    @property
    def ready(self) -> bool:
        return self._ready
