"""Server -> client conversation events."""
from __future__ import annotations


def session_ready(session_id: str, voice: str, sample_rate: int, system_prompt: str = "") -> dict:
    return {
        "type": "session.ready",
        "session_id": session_id,
        "voice": voice,
        "sample_rate": sample_rate,
        "system_prompt": system_prompt,
    }


def asr_partial(text: str, sentence_id: int, begin_ms: int) -> dict:
    return {"type": "asr.partial", "text": text, "sentence_id": sentence_id, "begin_ms": begin_ms}


def asr_final(text: str, sentence_id: int, begin_ms: int, end_ms: int) -> dict:
    return {
        "type": "asr.final",
        "text": text,
        "sentence_id": sentence_id,
        "begin_ms": begin_ms,
        "end_ms": end_ms,
    }


def text_delta(turn_id: str, text: str) -> dict:
    return {"type": "assistant.text.delta", "turn_id": turn_id, "text": text}


def text_done(turn_id: str, text: str) -> dict:
    return {"type": "assistant.text.done", "turn_id": turn_id, "text": text}


def audio_start(turn_id: str, index: int, sample_rate: int) -> dict:
    return {
        "type": "tts.audio.start",
        "turn_id": turn_id,
        "sentence_index": index,
        "sample_rate": sample_rate,
        "format": "pcm",
    }


def audio_done(turn_id: str, index: int) -> dict:
    return {"type": "tts.audio.done", "turn_id": turn_id, "sentence_index": index}


def all_audio_done(turn_id: str) -> dict:
    return {"type": "assistant.audio.all_done", "turn_id": turn_id}


def backchannel_start(backchannel_id: str, sample_rate: int, volume: float) -> dict:
    return {
        "type": "assistant.backchannel.start",
        "backchannel_id": backchannel_id,
        "sample_rate": sample_rate,
        "format": "pcm",
        "volume": volume,
    }


def backchannel_done(backchannel_id: str) -> dict:
    return {"type": "assistant.backchannel.done", "backchannel_id": backchannel_id}


def interrupted(turn_id: str, reason: str) -> dict:
    return {"type": "assistant.interrupted", "turn_id": turn_id, "reason": reason}


def cancelled(turn_id: str, reason: str) -> dict:
    return {"type": "assistant.cancelled", "turn_id": turn_id, "reason": reason}


def turn_metrics(turn_id: str, **values) -> dict:
    return {"type": "metrics.turn", "turn_id": turn_id, **values}


def playback_metrics(turn_id: str, **values) -> dict:
    return {"type": "metrics.playback", "turn_id": turn_id, **values}


def error(code: str, message: str) -> dict:
    return {"type": "error", "code": code, "message": message}


def session_closed(reason: str) -> dict:
    return {"type": "session.closed", "reason": reason}
