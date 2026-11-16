from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from .deps import runtime_from_request

router = APIRouter()

# One self-contained file with inline CSS/JS: an air-gapped ASR host needs no
# Node build step. Keep it next to the package, not in a separate frontend.
_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


@router.get("/ui", include_in_schema=False)
@router.get("/ui/", include_in_schema=False)
async def ui(request: Request) -> FileResponse:
    runtime = runtime_from_request(request)
    if not runtime.config.server.ui_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    if not _INDEX.is_file():  # pragma: no cover - packaging error
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(
        _INDEX,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )
