from __future__ import annotations

import logging

from cosyvoice_common.errors import invalid_parameter, service_unavailable

from .core.lifecycle import Lifecycle
from .core.limits import TenantLimiter
from .engine_factory import create_engine
from .engines.base import ASREngine
from .gpu_metrics import GpuSampler
from .metrics import Metrics
from .protocol.bailian import AsrParameters
from .schemas import StreamParams
from .settings import ModelConfig, RuntimeConfig

logger = logging.getLogger("funasr_runtime.runtime")


class Runtime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.metrics = Metrics(config.service_name, config.version)
        self.gpu_sampler = GpuSampler(
            self.metrics,
            interval_s=config.metrics.gpu_sampling_interval_s,
            device=config.metrics.gpu_device,
        )
        self.lifecycle = Lifecycle(config.limits.shutdown_timeout_s)
        self.limiter = TenantLimiter(config.limits)
        self.engine: ASREngine = create_engine(config)
        self._ready = False
        self._ws_connections = 0

    async def startup(self) -> None:
        await self.engine.startup()
        await self.engine.warmup()
        self._ready = True
        self.metrics.set_ready(True)
        self.gpu_sampler.start()
        logger.info(
            "runtime started: engine=%s models=%s",
            self.config.engine.type,
            ", ".join(sorted(self.config.models)),
        )

    async def shutdown(self) -> None:
        if not self._ready and self.lifecycle.draining:
            return
        self._ready = False
        self.metrics.set_ready(False)
        await self.lifecycle.begin_shutdown()
        if await self.lifecycle.drain():
            logger.info("all in-flight requests drained")
        self.gpu_sampler.stop()
        await self.engine.shutdown()
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

    def resolve_model(self, name: str) -> tuple[str, ModelConfig]:
        key = self.config.aliases.get(name, name)
        model = self.config.models.get(key)
        if model is None:
            raise invalid_parameter(f"unknown model: {name!r}")
        return key, model

    def resolve_offline_model(self, name: str | None) -> tuple[str, ModelConfig]:
        return self.resolve_model(name or self.config.offline.default_model)

    def build_params(
        self,
        model_name: str,
        model: ModelConfig,
        parameters: AsrParameters,
        context: str = "",
    ) -> StreamParams:
        language = (
            parameters.language_hints[0]
            if parameters.language_hints
            else model.language
        )
        hotwords = parameters.hotwords or model.hotwords or context
        itn = model.itn if parameters.itn is None else parameters.itn
        speech_noise_threshold = 0.5 + 0.5 * float(
            parameters.speech_noise_threshold
        )

        if model.engine == "funasr":
            engine_config = self.config.engine.funasr
            partial_interval_ms = engine_config.partial_interval_ms
            partial_window_s = engine_config.partial_window_s
        elif model.engine == "llamacpp":
            engine_config = self.config.engine.llamacpp
            partial_interval_ms = engine_config.partial_interval_ms
            partial_window_s = 8.0
        else:
            partial_interval_ms = self.config.engine.mock.partial_interval_ms
            partial_window_s = 8.0

        return StreamParams(
            sample_rate=parameters.sample_rate,
            language=language,
            hotwords=hotwords,
            itn=itn,
            max_sentence_silence_ms=parameters.max_sentence_silence,
            speech_noise_threshold=speech_noise_threshold,
            partial_interval_ms=partial_interval_ms,
            partial_window_s=partial_window_s,
            max_segment_ms=self.config.limits.max_segment_ms,
            extra={"model_name": model_name},
        )
