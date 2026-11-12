from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from .settings import ModelConfig, VoiceConfig


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class HotFix(_Base):
    replace: list[dict[str, str]] = Field(default_factory=list)
    pronunciation: list[dict[str, str]] = Field(default_factory=list)


class SpeechInput(_Base):
    text: str = ""
    voice: str | None = None
    format: str = "wav"
    sample_rate: int = 24_000
    volume: int = 50
    rate: float = 1.0
    pitch: float = 1.0
    bit_rate: int | None = None
    enable_ssml: bool = False
    word_timestamp_enabled: bool = False
    seed: int = 0
    language_hints: list[str] = Field(default_factory=list)
    instruction: str | None = None
    enable_aigc_tag: bool = False
    enable_markdown_filter: bool = False
    hot_fix: HotFix | None = None
    text_type: str | None = None


class SpeechSynthesizerRequest(_Base):
    model: str
    input: SpeechInput


@dataclass(slots=True)
class SynthesisRequest:
    text: str
    model: str
    voice: str
    model_config: ModelConfig
    voice_config: VoiceConfig
    format: str
    sample_rate: int
    volume: int
    rate: float
    pitch: float
    bit_rate: int | None
    instruction: str | None
    seed: int
    language_hints: list[str]
    word_timestamp_enabled: bool
    request_id: str
    streaming: bool = False
    enable_ssml: bool = False
    api: str = "http"
