from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from cosyvoice_common.errors import invalid_parameter
from cosyvoice_common.text import approx_words, count_characters

from ..engines.base import AudioChunk
from ..schemas import SynthesisRequest
from .orchestrator import SynthesisOrchestrator


@dataclass(slots=True)
class SentenceStart:
    index: int
    text: str


@dataclass(slots=True)
class SentenceAudio:
    index: int
    text: str
    chunk: AudioChunk


@dataclass(slots=True)
class SentenceDone:
    index: int
    text: str
    pcm: bytes
    sample_rate: int
    words: list[dict[str, Any]]
    characters: int


SentenceEvent = SentenceStart | SentenceAudio | SentenceDone


async def stream_sentences(
    orchestrator: SynthesisOrchestrator,
    request: SynthesisRequest,
) -> AsyncIterator[SentenceEvent]:
    """Sentence-level synthesis pipeline shared by SSE and WebSocket adapters."""
    sentences = orchestrator.sentences(request)
    if not sentences:
        raise invalid_parameter("text must not be empty")

    total_characters = 0
    for index, sentence in enumerate(sentences):
        yield SentenceStart(index=index, text=sentence)
        pcm = bytearray()
        source_rate = request.sample_rate
        stream = orchestrator.stream_sentence(request, sentence)
        try:
            async for chunk in stream:
                source_rate = chunk.sample_rate
                pcm.extend(chunk.pcm)
                yield SentenceAudio(index=index, text=sentence, chunk=chunk)
        finally:
            await stream.aclose()

        words: list[dict[str, Any]] = []
        if request.word_timestamp_enabled and source_rate > 0:
            duration_ms = int(len(pcm) / (source_rate * 2) * 1000)
            words = approx_words(sentence, duration_ms)
        total_characters += count_characters(sentence)
        yield SentenceDone(
            index=index,
            text=sentence,
            pcm=bytes(pcm),
            sample_rate=source_rate,
            words=words,
            characters=total_characters,
        )
