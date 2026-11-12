from __future__ import annotations

import asyncio
import hmac
import re
import time
from pathlib import Path
from typing import Annotated

from cosyvoice_common.audio import content_type
from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from ..storage import sign_local_file
from .deps import runtime_from_request

router = APIRouter()

_FILENAME = re.compile(r"^audio_[A-Za-z0-9_.-]+\.(?:pcm|wav|mp3|opus)$")


@router.get("/files/{filename:path}")
async def download_audio(
    filename: str,
    request: Request,
    exp: Annotated[int | None, Query()] = None,
    sig: Annotated[str | None, Query()] = None,
):
    runtime = runtime_from_request(request)
    storage = runtime.config.storage
    if storage.type != "local" or not storage.serve_files:
        return JSONResponse({"code": "NotFound", "message": "not found"}, status_code=404)

    # The path parameter is intentionally strict: no slashes, no traversal,
    # only files produced by LocalAudioStore.
    if not _FILENAME.match(filename) or Path(filename).name != filename:
        return JSONResponse({"code": "NotFound", "message": "not found"}, status_code=404)

    now = int(time.time())
    if storage.signing_secret:
        if exp is None or sig is None or exp < now:
            return JSONResponse(
                {"code": "AccessDenied", "message": "invalid or expired URL"},
                status_code=403,
            )
        expected = sign_local_file(filename, exp, storage.signing_secret)
        if not hmac.compare_digest(sig, expected):
            return JSONResponse(
                {"code": "AccessDenied", "message": "invalid or expired URL"},
                status_code=403,
            )

    path = await asyncio.to_thread(
        lambda: Path(storage.dir).resolve() / filename
    )
    if not await asyncio.to_thread(path.is_file):
        return JSONResponse({"code": "NotFound", "message": "not found"}, status_code=404)

    extension = filename.rsplit(".", 1)[-1]
    max_age = max(0, int(exp - now)) if exp else max(0, int(storage.ttl_s))
    return FileResponse(
        path,
        media_type=content_type(extension),
        headers={
            "Cache-Control": f"private, max-age={max_age}",
            "Content-Disposition": f'inline; filename="{filename}"',
        },
    )
