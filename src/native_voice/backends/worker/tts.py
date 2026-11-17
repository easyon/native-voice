"""Remote TTS job: one internal WebSocket per sentence, priority-aware."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import numpy as np
import websockets

from ...config.schema import VoiceProfile
from ...service.voice_store import VoiceStore
from ..base import TTSBackend


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


class WorkerTTSBackend(TTSBackend):
    def __init__(
        self,
        ws_url: str,
        http_url: str,
        token: str,
        voices: dict[str, VoiceProfile],
        voice_store: VoiceStore | None = None,
    ):
        self.ws_url = ws_url
        self.http_url = http_url.rstrip("/")
        self.token = token
        self.voice_store = voice_store
        self._ready = False
        self._voices = [
            {"name": name, "description": p.description or name} for name, p in voices.items()
        ] or [{"name": "default", "description": "default"}]

    async def start(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(self.http_url + "/readyz")
                if resp.status_code != 200 or not resp.json().get("ready"):
                    raise RuntimeError(f"TTS engine not ready: {resp.status_code} {resp.text[:200]}")
        except Exception as exc:
            raise RuntimeError(f"TTS engine unreachable at {self.http_url}: {exc!r}") from exc
        self._ready = True

    async def close(self) -> None:
        self._ready = False

    def voices(self) -> list[dict]:
        items = list(self._voices)
        if self.voice_store is not None:
            existing = {v["name"] for v in items}
            for rec in self.voice_store.list():
                if rec["id"] not in existing:
                    items.append(
                        {
                            "name": rec["id"],
                            "description": rec.get("description") or f"克隆 · {rec.get('name', rec['id'])}",
                        }
                    )
        return items

    async def add_voice(self, name: str, prompt_wav, prompt_text: str, description: str = "") -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.http_url + "/voices",
                json={
                    "id": name,
                    "prompt_wav": str(prompt_wav),
                    "prompt_text": prompt_text,
                    "description": description,
                },
                headers=_headers(self.token),
            )
        if resp.status_code != 200:
            raise RuntimeError(f"TTS engine voice add failed: {resp.status_code} {resp.text[:300]}")

    async def remove_voice(self, name: str) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.delete(self.http_url + f"/voices/{name}", headers=_headers(self.token))

    @property
    def ready(self) -> bool:
        return self._ready

    async def synthesize(
        self, text: str, voice: str, cancel, priority: int = 1
    ) -> AsyncIterator[np.ndarray]:
        ws = await websockets.connect(self.ws_url, extra_headers=_headers(self.token), max_size=None)
        try:
            await ws.send(
                json.dumps(
                    {"op": "synthesize", "text": text, "voice": voice, "priority": priority},
                    ensure_ascii=False,
                )
            )
            while True:
                if cancel is not None and cancel.is_set():
                    try:
                        await ws.send(json.dumps({"op": "cancel"}))
                    except Exception:
                        pass
                    return
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if isinstance(msg, bytes):
                    yield np.frombuffer(msg, dtype="<f4").astype(np.float32)
                    continue
                obj = json.loads(msg)
                op = obj.get("op")
                if op == "done":
                    return
                if op == "error":
                    raise RuntimeError(obj.get("message", "TTS engine error"))
        finally:
            try:
                await ws.close()
            except Exception:
                pass
