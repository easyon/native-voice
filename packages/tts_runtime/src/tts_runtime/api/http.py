from __future__ import annotations

import base64
import logging
import time
import uuid
from typing import Annotated

from cosyvoice_common.audio import (
    encode_audio_async,
    encode_stream_chunk_async,
)
from cosyvoice_common.errors import APIError, internal_error
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..context import bind_context
from ..core.sentence_pipeline import (
    SentenceAudio,
    SentenceDone,
    SentenceStart,
    stream_sentences,
)
from ..core.tenant import Tenant
from ..core.timeout import deadline
from ..protocol import bailian
from ..schemas import SpeechSynthesizerRequest, SynthesisRequest
from .deps import authenticate, runtime_from_request

logger = logging.getLogger("tts_runtime.http")
router = APIRouter()


def _request_id(value: str | None) -> str:
    if value:
        cleaned = "".join(ch for ch in value if ch.isprintable()).strip()[:128]
        if cleaned:
            return cleaned
    return uuid.uuid4().hex


@router.post("/api/v1/services/audio/tts/SpeechSynthesizer")
async def speech_synthesizer(
    payload: SpeechSynthesizerRequest,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    x_dashscope_sse: Annotated[str | None, Header(alias="X-DashScope-SSE")] = None,
):
    runtime = runtime_from_request(request)
    tenant = authenticate(authorization, runtime.config.auth)
    request.state.tenant_id = tenant.id
    request_id = _request_id(request.headers.get("X-Request-Id"))
    request.state.request_id = request_id
    streaming = (x_dashscope_sse or "").strip().lower() == "enable"
    request_deadline_s = max(
        1.0,
        min(
            float(runtime.config.server.request_timeout_s),
            float(runtime.config.limits.max_request_s),
        ),
    )

    await runtime.lifecycle.enter()
    stream_returned = False
    try:
        with bind_context(
            request_id=request_id,
            tenant=tenant.id,
            api="sse" if streaming else "http",
        ):
            tts = runtime.build_request(
                payload.model,
                payload.input,
                request_id,
                streaming=streaming,
                api="sse" if streaming else "http",
            )
            runtime.limiter.consume_characters(
                tenant,
                runtime.orchestrator.characters(tts),
            )
            logger.info(
                "http tts: model=%s voice=%s stream=%s",
                tts.model,
                tts.voice,
                streaming,
            )

            if streaming:
                response = StreamingResponse(
                    _sse(runtime, tts, request, tenant),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
                stream_returned = True
                return response

            async with runtime.limiter.slot(tenant):
                async with deadline(request_deadline_s):
                    pcm, source_rate = await runtime.orchestrator.synthesize_full(tts)
                    audio = await encode_audio_async(
                        pcm,
                        source_rate,
                        tts.format,
                        sample_rate=tts.sample_rate,
                        volume=tts.volume,
                        pitch=tts.pitch,
                        bit_rate=tts.bit_rate,
                    )
                    stored = await runtime.store.save(audio, tts.format, tts.request_id)
            runtime.metrics.request("http", "ok")
            return JSONResponse(
                bailian.http_success(
                    tts.request_id,
                    audio_b64="" if stored else base64.b64encode(audio).decode("ascii"),
                    stored=stored,
                    characters=runtime.orchestrator.characters(tts),
                )
            )
    except APIError as exc:
        runtime.metrics.error(exc.code, api="sse" if streaming else "http")
        runtime.metrics.request("sse" if streaming else "http", "error")
        raise
    except Exception:
        runtime.metrics.error("InternalError", api="sse" if streaming else "http")
        runtime.metrics.request("sse" if streaming else "http", "error")
        raise
    finally:
        if not stream_returned:
            await runtime.lifecycle.leave()


async def _sse(
    runtime,
    tts: SynthesisRequest,
    http_request: Request,
    tenant: Tenant,
):
    collect_audio = runtime.config.storage.type != "none"
    all_pcm = bytearray()
    request_deadline_s = max(
        1.0,
        min(
            float(runtime.config.server.request_timeout_s),
            float(runtime.config.limits.max_request_s),
        ),
    )
    source_rate = tts.sample_rate
    characters = 0
    disconnected = False
    stored = None
    first_audio = True
    started = time.monotonic()
    pipeline = stream_sentences(runtime.orchestrator, tts)
    try:
        with bind_context(
            request_id=tts.request_id,
            tenant=tenant.id,
            model=tts.model,
            voice=tts.voice,
            api="sse",
        ):
            async with runtime.limiter.slot(tenant):
                async with deadline(request_deadline_s):
                    async for event in pipeline:
                        if await http_request.is_disconnected():
                            disconnected = True
                            break

                        if isinstance(event, SentenceStart):
                            yield bailian.sse(
                                bailian.sentence_begin(
                                    tts.request_id,
                                    event.index,
                                    event.text,
                                    words=[],
                                )
                            )
                        elif isinstance(event, SentenceAudio):
                            source_rate = event.chunk.sample_rate
                            if collect_audio:
                                if (
                                    len(all_pcm) + len(event.chunk.pcm)
                                    > runtime.config.limits.max_stream_audio_bytes
                                ):
                                    collect_audio = False
                                    all_pcm.clear()
                                    logger.warning(
                                        "SSE audio too large to store; "
                                        "continuing without download URL"
                                    )
                                else:
                                    all_pcm.extend(event.chunk.pcm)
                            encoded = await encode_stream_chunk_async(
                                event.chunk.pcm,
                                event.chunk.sample_rate,
                                tts.format,
                                sample_rate=tts.sample_rate,
                                volume=tts.volume,
                                pitch=tts.pitch,
                                bit_rate=tts.bit_rate,
                                first=first_audio,
                            )
                            first_audio = False
                            runtime.metrics.audio(len(encoded))
                            yield bailian.sse(
                                bailian.sentence_synthesis(
                                    tts.request_id,
                                    event.index,
                                    words=[],
                                    audio_b64=base64.b64encode(encoded).decode("ascii"),
                                )
                            )
                        elif isinstance(event, SentenceDone):
                            characters = event.characters
                            yield bailian.sse(
                                bailian.sentence_end(
                                    tts.request_id,
                                    event.index,
                                    event.text,
                                    event.words,
                                    event.characters,
                                )
                            )

                if not disconnected and collect_audio and all_pcm:
                    try:
                        full_audio = await encode_audio_async(
                            bytes(all_pcm),
                            source_rate,
                            tts.format,
                            sample_rate=tts.sample_rate,
                            volume=tts.volume,
                            pitch=tts.pitch,
                            bit_rate=tts.bit_rate,
                        )
                        stored = await runtime.store.save(
                            full_audio, tts.format, tts.request_id
                        )
                    except Exception:  # noqa: BLE001 - audio already streamed
                        logger.exception("storing streamed audio failed")
    except APIError as exc:
        runtime.metrics.error(exc.code, api="sse")
        runtime.metrics.request("sse", "error")
        yield bailian.sse(bailian.error_body(tts.request_id, exc))
        return
    except Exception:  # noqa: BLE001 - headers are already sent
        logger.exception("SSE stream failed")
        error = internal_error()
        runtime.metrics.error(error.code, api="sse")
        runtime.metrics.request("sse", "error")
        yield bailian.sse(bailian.error_body(tts.request_id, error))
        return
    finally:
        await pipeline.aclose()
        await runtime.lifecycle.leave()

    if disconnected:
        return

    runtime.metrics.request("sse", "ok")
    runtime.metrics.request_duration(
        time.monotonic() - started,
        api="sse",
        model=tts.model,
    )
    yield bailian.sse(
        bailian.http_stream_finish(
            tts.request_id,
            stored=stored,
            characters=characters,
        )
    )
