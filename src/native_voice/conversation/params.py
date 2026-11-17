from __future__ import annotations

from dataclasses import dataclass

from ..config.schema import ConversationConfig


@dataclass
class VadParams:
    input_rate: int = 16000
    frame_ms: int = 20
    endpoint_silence_ms: int = 500
    noise_override_ms: int = 500
    min_speech_ms: int = 240
    max_utterance_s: float = 20.0
    preroll_ms: int = 450
    silence_db_abs: float = -46.0
    barge_db: float = -30.0
    barge_sustain_ms: int = 400
    echo_tail_s: float = 0.45
    vad_warmup_frames: int = 40

    @classmethod
    def from_config(cls, cfg: ConversationConfig) -> VadParams:
        return cls(
            input_rate=cfg.input_rate,
            endpoint_silence_ms=cfg.endpoint_silence_ms,
            noise_override_ms=cfg.noise_override_ms,
            min_speech_ms=cfg.min_speech_ms,
            max_utterance_s=cfg.max_utterance_s,
            preroll_ms=cfg.preroll_ms,
            barge_db=cfg.barge_db,
            barge_sustain_ms=cfg.barge_sustain_ms,
        )
