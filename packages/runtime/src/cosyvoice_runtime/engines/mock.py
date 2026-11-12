from __future__ import annotations

import array
import asyncio
import math
from collections.abc import AsyncIterator

from ..schemas import SynthesisRequest
from ..settings import MockEngineConfig
from .base import AudioChunk, TTSEngine


class MockTTSEngine(TTSEngine):
    """Deterministic tone generator for protocol tests and local development."""

    def __init__(self, config: MockEngineConfig) -> None:
        self.config = config

    async def ready(self) -> bool:
        return True

    async def synthesize(self, request: SynthesisRequest) -> AsyncIterator[AudioChunk]:
        sample_rate = request.sample_rate
        rate = max(0.5, min(2.0, float(request.rate)))
        text_len = max(1, len(request.text))
        duration_s = max(0.35, min(10.0, text_len * 0.06)) / rate
        total_frames = max(1, int(sample_rate * duration_s))
        chunk_frames = max(1, int(sample_rate * max(10, self.config.chunk_ms) / 1000))
        amplitude = max(0.0, min(0.9, float(self.config.amplitude))) * 32767.0
        frequency = 220.0 * max(0.5, min(2.0, float(request.pitch)))
        fade_frames = min(int(sample_rate * 0.02), max(1, total_frames // 4))

        offset = 0
        while offset < total_frames:
            frames = min(chunk_frames, total_frames - offset)
            samples = array.array("h")
            for i in range(frames):
                absolute = offset + i
                envelope = 1.0
                if absolute < fade_frames:
                    envelope = absolute / fade_frames
                elif absolute > total_frames - fade_frames:
                    envelope = max(0.0, (total_frames - absolute) / fade_frames)
                value = amplitude * envelope * math.sin(
                    2.0 * math.pi * frequency * absolute / sample_rate
                )
                samples.append(max(-32768, min(32767, int(value))))
            offset += frames
            yield AudioChunk(
                pcm=samples.tobytes(),
                sample_rate=sample_rate,
                is_final=offset >= total_frames,
            )
            if self.config.realtime:
                await asyncio.sleep(frames / sample_rate)
            else:
                await asyncio.sleep(0)
