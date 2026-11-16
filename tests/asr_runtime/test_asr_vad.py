from __future__ import annotations

from funasr_runtime.core.vad import (
    EnergyVadStream,
    MockVadStream,
    PassthroughVadStream,
    SpeechEnd,
    SpeechStart,
)

from tests.asr_runtime.conftest import silence_pcm, tone_pcm


async def _feed_all(vad, pcm: bytes, *, chunk_ms: int = 100):
    events = []
    chunk = chunk_ms * 32
    for offset in range(0, len(pcm), chunk):
        events.extend(await vad.feed(pcm[offset : offset + chunk]))
    return events


async def test_mock_vad_segments() -> None:
    vad = MockVadStream(segment_ms=500)
    events = await _feed_all(vad, tone_pcm(1_000))
    starts = [event.time_ms for event in events if isinstance(event, SpeechStart)]
    ends = [event.time_ms for event in events if isinstance(event, SpeechEnd)]
    assert starts == [0, 500, 1_000]
    assert ends == [500, 1_000]
    final = await vad.finish()
    assert final == [SpeechEnd(1_000, "flush")]


async def test_passthrough_vad() -> None:
    vad = PassthroughVadStream()
    events = await _feed_all(vad, tone_pcm(300))
    assert events == [SpeechStart(0)]
    assert await vad.finish() == [SpeechEnd(300, "flush")]


async def test_energy_vad_detects_speech_and_silence() -> None:
    vad = EnergyVadStream(
        threshold_db=-35,
        min_speech_ms=100,
        silence_ms=300,
        pre_roll_ms=100,
    )
    pcm = silence_pcm(200) + tone_pcm(500) + silence_pcm(600)
    events = await _feed_all(vad, pcm)
    starts = [event for event in events if isinstance(event, SpeechStart)]
    ends = [event for event in events if isinstance(event, SpeechEnd)]
    assert len(starts) == 1
    assert len(ends) == 1
    assert 0 <= starts[0].time_ms <= 250
    assert 600 <= ends[0].time_ms <= 900


async def test_energy_vad_splits_long_segments() -> None:
    vad = EnergyVadStream(
        threshold_db=-35,
        min_speech_ms=100,
        silence_ms=1_000,
        max_segment_ms=600,
        pre_roll_ms=0,
    )
    events = await _feed_all(vad, tone_pcm(1_800))
    ends = [event for event in events if isinstance(event, SpeechEnd)]
    starts = [event for event in events if isinstance(event, SpeechStart)]
    assert len(ends) >= 2
    assert len(starts) >= 2


async def test_energy_vad_flushes_on_finish() -> None:
    vad = EnergyVadStream(threshold_db=-35, min_speech_ms=100, silence_ms=5_000)
    await vad.feed(tone_pcm(400))
    events = await vad.finish()
    assert any(isinstance(event, SpeechEnd) for event in events)
