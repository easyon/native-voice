from __future__ import annotations

import logging
from typing import Annotated

from cosyvoice_common.errors import APIError
from fastapi import APIRouter, File, Form, Header, Request, UploadFile

from .deps import authenticate, runtime_from_request

logger = logging.getLogger("cosyvoice_runtime.voices")

router = APIRouter()

_CHUNK = 64 * 1024


def _not_found(message: str = "voice not found") -> APIError:
    return APIError("NotFound", message, 404)


def _voice_body(info, *, cloned: bool) -> dict[str, object]:
    body = info.to_dict()
    body.pop("transcript", None)
    body.update({"cloned": cloned, "mode": "zero_shot" if cloned else None})
    return body


@router.get("/api/v1/voices")
async def list_voices(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    runtime = runtime_from_request(request)
    authenticate(authorization, runtime.config.auth)

    voices: list[dict[str, object]] = []
    cloned_ids = {info.id for info in runtime.cloned_voices.list()}
    for voice_id, voice in sorted(runtime.config.voices.items()):
        # Runtime.startup registers cloned voices in the same config dict, so
        # skip them here and emit them once below with cloned=true.
        if voice_id in cloned_ids:
            continue
        voices.append(
            {
                "id": voice_id,
                "name": voice_id,
                "mode": voice.mode,
                "cloned": False,
                "duration_s": None,
                "created_at": None,
                "source": None,
            }
        )
    for info in runtime.cloned_voices.list():
        voices.append(_voice_body(info, cloned=True))
    return {"voices": voices}


@router.post("/api/v1/voices/clone")
async def clone_voice(
    request: Request,
    audio: Annotated[UploadFile, File()],
    transcript: Annotated[str, Form()] = "",
    name: Annotated[str, Form()] = "",
    source: Annotated[str, Form()] = "upload",
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    runtime = runtime_from_request(request)
    tenant = authenticate(authorization, runtime.config.auth)
    if not runtime.config.cloning.enabled:
        raise APIError("AccessDenied", "voice cloning is disabled", 403)

    data = await _read_limited(audio, runtime.config.cloning.max_upload_bytes)
    try:
        info = await runtime.cloned_voices.create(
            data,
            audio.filename or "prompt.wav",
            transcript,
            name=name,
            source=source,
        )
    finally:
        await audio.close()

    voice = runtime.cloned_voices.voice_config(info.id)
    if voice is not None:
        runtime.registry.add_voice(info.id, voice)
        cache = getattr(runtime.engine, "cache_zero_shot_voice", None)
        if cache is not None:
            try:
                await cache(voice)
            except Exception:  # noqa: BLE001 - latency optimization only
                logger.warning(
                    "failed to cache prompt features for cloned voice %s",
                    info.id,
                    exc_info=True,
                )
    return {"voice": _voice_body(info, cloned=True), "tenant": tenant.id}


@router.delete("/api/v1/voices/{voice_id}")
async def delete_voice(
    voice_id: str,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    runtime = runtime_from_request(request)
    authenticate(authorization, runtime.config.auth)
    if not await runtime.cloned_voices.delete(voice_id):
        raise _not_found()
    runtime.registry.remove_voice(voice_id)
    return {"deleted": True, "id": voice_id}


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise APIError(
                "RequestTooLarge",
                f"audio file exceeds {limit} bytes",
                413,
            )
        chunks.append(chunk)
    if not chunks:
        raise APIError("InvalidParameter", "audio file is empty", 400)
    return b"".join(chunks)
