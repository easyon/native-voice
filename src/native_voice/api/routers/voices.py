from __future__ import annotations

import io
import secrets

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ...audio.resample import resample
from ...security.auth import AuthError, Principal
from ...service.voice_store import PROMPT_PREFIX

router = APIRouter()


def _principal(request: Request) -> Principal:
    try:
        return request.app.state.authenticator.authenticate_headers(request.headers)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


def _require_tts(principal: Principal) -> None:
    if not principal.has("tts"):
        raise HTTPException(status_code=403, detail="API key needs the tts scope")


async def _transcribe(state, audio16: np.ndarray) -> str:
    session = state.backends.asr.create_session()
    try:
        partial = ""
        step = 16000
        for i in range(0, len(audio16), step):
            partial = await session.feed(audio16[i:i + step])
        final = await session.finalize()
        return (final or partial).strip()
    finally:
        try:
            await session.close()
        except Exception:
            pass


@router.get("/v1/voices")
async def voices(request: Request) -> dict:
    return {"voices": request.app.state.backends.tts.voices()}


@router.post("/v1/voices/clone")
async def clone_voice(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008 - FastAPI dependency marker
    name: str = Form(""),
    text: str = Form(""),
    consent: bool = Form(False),
):
    state = request.app.state
    principal = _principal(request)
    _require_tts(principal)
    if not consent:
        raise HTTPException(status_code=400, detail="consent is required")

    cfg = state.config
    limit = int(cfg.voice_store.max_upload_mb * 1024 * 1024)
    raw = await file.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(status_code=413, detail=f"audio exceeds {cfg.voice_store.max_upload_mb} MB")
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"cannot decode audio: {exc}") from exc
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr != 24000:
        audio = resample(audio, int(sr), 24000)

    # Trim leading/trailing near-silence, then normalize.
    idx = np.flatnonzero(np.abs(audio) > 0.01)
    if idx.size == 0:
        raise HTTPException(status_code=400, detail="reference audio is silent")
    start = max(0, int(idx[0]) - int(0.05 * 24000))
    end = min(len(audio), int(idx[-1]) + int(0.05 * 24000))
    audio = audio[start:end]
    duration = len(audio) / 24000.0
    if duration < cfg.voice_store.min_seconds or duration > cfg.voice_store.max_seconds:
        raise HTTPException(
            status_code=400,
            detail=(
                f"reference audio must be {cfg.voice_store.min_seconds:g}-"
                f"{cfg.voice_store.max_seconds:g}s (got {duration:.2f}s)"
            ),
        )
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio * min(1.0, 0.95 / peak)

    transcript = (text or "").strip()[:500]
    if not transcript:
        try:
            transcript = (await _transcribe(state, resample(audio, 24000, 16000)))[:500]
        except Exception:
            transcript = ""
    if not transcript:
        raise HTTPException(status_code=400, detail="cannot recognize the reference text; please provide it")

    display = (name or "").strip()[:40] or "我的音色"
    voice_id = "clone_" + secrets.token_hex(6)
    prompt_text = PROMPT_PREFIX + transcript
    description = f"克隆 · {display}"
    record = state.voice_store.add(
        voice_id, display, transcript, prompt_text, description, audio, 24000
    )
    try:
        await state.backends.tts.add_voice(voice_id, record["path"], prompt_text, description)
    except NotImplementedError as exc:
        state.voice_store.delete(voice_id)
        raise HTTPException(status_code=501, detail="this backend does not support voice cloning") from exc
    except Exception as exc:
        state.voice_store.delete(voice_id)
        raise HTTPException(status_code=503, detail=f"TTS engine voice add failed: {exc}") from exc

    return {
        "voice": {"name": voice_id, "display_name": display, "description": description},
        "transcript": transcript,
        "duration": round(duration, 2),
    }


@router.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str, request: Request) -> dict:
    state = request.app.state
    principal = _principal(request)
    _require_tts(principal)
    if state.voice_store.get(voice_id) is None:
        raise HTTPException(status_code=404, detail="voice not found")
    try:
        await state.backends.tts.remove_voice(voice_id)
    except NotImplementedError:
        pass
    except Exception:
        pass
    state.voice_store.delete(voice_id)
    return {"ok": True, "voice": voice_id}
