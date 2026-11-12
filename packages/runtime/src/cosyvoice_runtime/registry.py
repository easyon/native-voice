from __future__ import annotations

import threading

from cosyvoice_common.errors import invalid_parameter, voice_not_found

from .settings import ModelConfig, RuntimeConfig, VoiceConfig


class Registry:
    """Configured model and voice lookup.

    There is intentionally no Bailian alias mapping: the model name and voice ID
    used by the business layer are exactly the keys in runtime.yaml.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._lock = threading.RLock()

    @property
    def config(self) -> RuntimeConfig:
        with self._lock:
            return self._config

    def reload(self, config: RuntimeConfig) -> None:
        with self._lock:
            self._config = config

    def get_model(self, name: str) -> ModelConfig:
        with self._lock:
            model = self._config.models.get(name)
        if model is None:
            known = ", ".join(sorted(self._config.models)) or "<empty>"
            raise invalid_parameter(
                f"unknown model: {name}; configured models: {known}"
            )
        return model

    def get_voice(self, voice_id: str) -> VoiceConfig:
        with self._lock:
            voice = self._config.voices.get(voice_id)
        if voice is None:
            known = ", ".join(sorted(self._config.voices)) or "<empty>"
            raise voice_not_found(
                f"voice not found: {voice_id}; configured voices: {known}"
            )
        return voice

    def resolve(self, model_name: str, voice_id: str | None) -> tuple[ModelConfig, VoiceConfig, str]:
        model = self.get_model(model_name)
        resolved_voice = voice_id or model.default_voice
        if not resolved_voice:
            raise invalid_parameter(
                f"voice is required for model {model_name!r} "
                "(no default_voice configured)"
            )
        return model, self.get_voice(resolved_voice), resolved_voice
