from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...config.schema import VoiceProfile


@dataclass
class TTSModelConfig:
    model_dir: Path
    prompt_wav: Path
    prompt_text: str
    voices: dict[str, VoiceProfile] = field(default_factory=dict)
    hop: int = 15
    max_cache_len: int = 2048
    use_cuda_graph: bool = True
    warmup: bool = True
    verbose: bool = False
    third_party_dir: Path = Path("third_party/CosyVoice-main")
