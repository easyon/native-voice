from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ASRModelConfig:
    model: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
    device: str = "cuda:0"
    chunk_ms: int = 600
    rate: int = 16000
