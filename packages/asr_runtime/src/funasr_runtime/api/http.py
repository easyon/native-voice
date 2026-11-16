from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

from cosyvoice_common.errors import (
    APIError,
    invalid_parameter,
    unsupported_parameter,
)
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from ..core.audio import decode_to_pcm16k
from ..protocol import openai
from ..protocol.bailian import AsrParameters
from ..schemas import TranscriptSegment
from .deps import authenticate, runtime_from_request

logger = logging.getLogger("funasr_runtime.http")
router = APIRouter()

_RESPONSE_FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}


@router.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: Annotated[UploadFile, File()],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    temperature: Annotated[float | None, Form()] = None,
) -> Response:
    runtime = runtime_from_request(request)
    try:
        authenticate(request.headers.get("authorization"), runtime.config.auth)
        if response_format not in _RESPONSE_FORMATS:
            raise unsupported_parameter(
                f"unsupported response_format: {response_format}; "
                f"supported: {', '.join(sorted(_RESPONSE_FORMATS))}"
            )
        model_key, model_config = runtime.resolve_offline_model(model)
        data = await file.read()
        if len(data) > runtime.config.offline.max_upload_bytes:
            raise APIError(
                "RequestTooLarge",
                "uploaded audio exceeds offline.max_upload_bytes",
                413,
            )
        if not data:
            raise invalid_parameter("uploaded file is empty")

        suffix = Path(file.filename or "").suffix.lower().lstrip(".") or None
        pcm = await asyncio.to_thread(decode_to_pcm16k, data, suffix)
        if not pcm:
            raise invalid_parameter("uploaded audio contains no samples")
        duration_ms = len(pcm) // 32
        if duration_ms > runtime.config.offline.max_duration_s * 1000:
            raise APIError(
                "AudioTooLong",
                "uploaded audio exceeds offline.max_duration_s",
                400,
            )

        parameters = AsrParameters(
            format="pcm",
            sample_rate=16_000,
            language_hints=[language] if language else [],
            hotwords=prompt,
        )
        params = runtime.build_params(
            model_key,
            model_config,
            parameters,
            context=prompt or "",
        )
        params.timestamps = response_format in {"verbose_json", "srt", "vtt"}
        transcript = await runtime.engine.transcribe(model_config, pcm, params)
        if not transcript.segments and transcript.duration_ms > 0:
            transcript.segments = [
                TranscriptSegment(
                    text=transcript.text,
                    begin_time=0,
                    end_time=transcript.duration_ms,
                )
            ]

        runtime.metrics.request("http", "ok")
        if response_format == "text":
            return PlainTextResponse(transcript.text)
        if response_format == "verbose_json":
            return JSONResponse(openai.verbose_json(transcript, language))
        if response_format == "srt":
            return PlainTextResponse(
                openai.srt(transcript.segments),
                media_type="application/x-subrip",
            )
        if response_format == "vtt":
            return PlainTextResponse(
                openai.vtt(transcript.segments),
                media_type="text/vtt",
            )
        return JSONResponse({"text": transcript.text})
    except APIError as exc:
        runtime.metrics.error(exc.code, api="http")
        runtime.metrics.request("http", "error")
        return JSONResponse(
            openai.error_body(
                exc.message,
                type_=(
                    "authentication_error"
                    if exc.status_code == 401
                    else "invalid_request_error"
                    if exc.status_code < 500
                    else "server_error"
                ),
                code=exc.code,
            ),
            status_code=exc.status_code,
        )
    except Exception as exc:  # noqa: BLE001 - map engine failures to OpenAI shape
        logger.exception("transcription failed")
        runtime.metrics.error("InternalError", api="http")
        runtime.metrics.request("http", "error")
        return JSONResponse(
            openai.error_body(str(exc) or "internal error", type_="server_error"),
            status_code=500,
        )
