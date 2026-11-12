from __future__ import annotations

import logging

from cosyvoice_common.audio import validate_format, validate_sample_rate
from cosyvoice_common.errors import (
    invalid_parameter,
    service_unavailable,
    text_too_long,
    unsupported_parameter,
)
from cosyvoice_common.text import preprocess_text

from .core.lifecycle import Lifecycle
from .core.limits import TenantLimiter
from .core.orchestrator import SynthesisOrchestrator
from .engine_factory import create_engine
from .engines.base import TTSEngine
from .metrics import Metrics
from .registry import Registry
from .schemas import SpeechInput, SynthesisRequest
from .settings import RuntimeConfig
from .storage import AudioStore, create_store

logger = logging.getLogger("cosyvoice_runtime.runtime")


class Runtime:
    """Composition root for the gateway.

    Protocol adapters depend on this object; engines, stores and orchestrators
    are replaceable through their interfaces/factories.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.registry = Registry(config)
        self.metrics = Metrics(config.service_name, config.version)
        self.lifecycle = Lifecycle(config.limits.shutdown_timeout_s)
        self.limiter = TenantLimiter(config.limits)
        self.engine: TTSEngine = create_engine(config)
        self.store: AudioStore = create_store(config.storage)
        self.orchestrator = SynthesisOrchestrator(
            engine=self.engine,
            limits=config.limits,
            metrics=self.metrics,
        )
        self._ready = False
        self._ws_connections = 0

    async def startup(self) -> None:
        await self.store.startup()
        await self.engine.startup()
        await self._warmup()
        self._ready = True
        self.metrics.set_ready(True)
        logger.info(
            "runtime started: engine=%s models=%s voices=%s",
            self.config.engine.type,
            ", ".join(sorted(self.config.models)) or "<none>",
            ", ".join(sorted(self.config.voices)) or "<none>",
        )

    async def shutdown(self) -> None:
        if not self._ready and self.lifecycle.draining:
            return
        self._ready = False
        self.metrics.set_ready(False)
        await self.lifecycle.begin_shutdown()

        # Stop accepting new work immediately, then let in-flight requests and
        # engine generators finish. This is what lets Kubernetes terminate a
        # pod without cutting a live call mid-sentence.
        if await self.lifecycle.drain():
            logger.info("all in-flight requests drained")
        if not await self.orchestrator.wait_idle(self.config.limits.shutdown_timeout_s):
            logger.warning("engine streams did not drain before shutdown timeout")
        await self.engine.shutdown()
        await self.store.shutdown()
        logger.info("runtime stopped")

    async def ready(self) -> bool:
        if not self._ready or self.lifecycle.draining:
            return False
        ready = await self.engine.ready()
        self.metrics.set_ready(ready)
        return ready

    async def websocket_connected(self) -> None:
        maximum = int(self.config.server.max_ws_connections)
        if maximum > 0 and self._ws_connections >= maximum:
            raise service_unavailable("websocket connection limit reached")
        self._ws_connections += 1
        self.metrics.ws_connected()

    async def websocket_disconnected(self) -> None:
        self._ws_connections = max(0, self._ws_connections - 1)
        self.metrics.ws_disconnected()

    async def _warmup(self) -> None:
        # Adapters may warm their own connection pool / model. The in-process
        # CosyVoice adapter additionally needs one real request to compile the
        # model and validate the configured voice.
        await self.engine.warmup()
        if self.config.engine.type != "cosyvoice":
            return
        if not self.config.models:
            return
        model_name = next(iter(self.config.models))
        model = self.config.models[model_name]
        voice_id = model.default_voice or next(iter(self.config.voices), None)
        if not voice_id:
            logger.warning("warmup skipped: no voice configured")
            return
        try:
            request = self.build_request(
                model_name,
                SpeechInput(
                    text=self.config.engine.cosyvoice.warmup_text,
                    voice=voice_id,
                ),
                "warmup",
                streaming=False,
                api="warmup",
            )
            async for _ in self.engine.synthesize(request):
                pass
            logger.info("warmup completed: model=%s voice=%s", model_name, voice_id)
        except Exception:  # noqa: BLE001 - startup must fail if the model is unusable
            logger.exception("warmup failed; refusing to become ready")
            raise

    def build_request(
        self,
        model_name: str,
        options: SpeechInput,
        request_id: str,
        *,
        streaming: bool,
        allow_empty_text: bool = False,
        api: str = "http",
    ) -> SynthesisRequest:
        model, voice, voice_id = self.registry.resolve(model_name, options.voice)
        fmt = validate_format(options.format)
        sample_rate = validate_sample_rate(options.sample_rate)
        if not 0 <= options.volume <= 100:
            raise invalid_parameter("volume must be between 0 and 100")
        if not 0.5 <= options.rate <= 2.0:
            raise invalid_parameter("rate must be between 0.5 and 2.0")
        if not 0.5 <= options.pitch <= 2.0:
            raise invalid_parameter("pitch must be between 0.5 and 2.0")
        if streaming and abs(options.pitch - 1.0) > 1e-6:
            raise unsupported_parameter("pitch is not supported for streaming synthesis")
        if streaming and fmt == "opus":
            raise unsupported_parameter(
                "opus is not supported for streaming synthesis; use pcm, wav or mp3"
            )
        if options.text_type and options.text_type.lower() not in {
            "plaintext",
            "plain_text",
        }:
            raise unsupported_parameter(
                f"text_type {options.text_type!r} is not supported; use PlainText"
            )
        if options.enable_aigc_tag:
            raise unsupported_parameter("enable_aigc_tag is not implemented")

        if options.text.strip():
            text = preprocess_text(
                options.text,
                options,
                ssml_strict=self.config.limits.ssml_strict,
            )
        elif allow_empty_text:
            text = ""
        else:
            raise invalid_parameter("text must not be empty")
        if len(text) > self.config.limits.max_text_chars:
            raise text_too_long(
                f"text length {len(text)} exceeds {self.config.limits.max_text_chars}"
            )

        instruction = options.instruction
        if instruction is None:
            instruction = voice.instruct or None

        return SynthesisRequest(
            text=text,
            model=model_name,
            voice=voice_id,
            model_config=model,
            voice_config=voice,
            format=fmt,
            sample_rate=sample_rate,
            volume=options.volume,
            rate=options.rate,
            pitch=options.pitch,
            bit_rate=options.bit_rate,
            instruction=instruction,
            seed=options.seed,
            language_hints=list(options.language_hints),
            word_timestamp_enabled=options.word_timestamp_enabled,
            request_id=request_id,
            streaming=streaming,
            enable_ssml=options.enable_ssml,
            api=api,
        )
