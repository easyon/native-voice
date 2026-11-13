from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import hashlib
import importlib.util
import inspect
import logging
import os
import random
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from cosyvoice_common.errors import (
    APIError,
    internal_error,
    invalid_parameter,
    service_unavailable,
)

from ..schemas import SynthesisRequest
from ..settings import CosyVoiceEngineConfig, ModelConfig, VoiceConfig
from .base import AudioChunk, TTSEngine

logger = logging.getLogger("cosyvoice_runtime.engine")


class CosyVoiceEngine(TTSEngine):
    """In-process adapter for the official CosyVoice Python API.

    This is the single-node / functional integration path. For production, the
    same interface can be backed by Triton/gRPC without changing the gateway.
    One process should own one GPU; use one replica per GPU and put a CPU
    gateway in front for multi-GPU / 100+ concurrency deployments.
    """

    def __init__(
        self,
        config: CosyVoiceEngineConfig,
        models: dict[str, ModelConfig],
        voices: dict[str, VoiceConfig] | None = None,
    ) -> None:
        self.config = config
        self.models = models
        self.voices = voices or {}
        self._models: dict[str, Any] = {}
        self._models_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max(1, config.max_concurrent))
        # (engine-scoped artifact key, prompt fingerprint) -> registered spk id
        self._prompt_ids: dict[tuple[str, str], str] = {}

    async def startup(self) -> None:
        if not self.config.preload:
            return
        # Preload per (artifact, engine) pair so a deployment that mixes
        # CosyVoice2 and CosyVoice3 artifacts loads each with the right class.
        targets = {
            (cfg.artifact, cfg.engine)
            for cfg in self.models.values()
            if cfg.artifact
        }
        for artifact, engine_type in sorted(targets):
            await self._get_model(artifact, engine_type)

    async def shutdown(self) -> None:
        self._models.clear()
        self._prompt_ids.clear()
        await asyncio.to_thread(self._empty_cuda_cache)

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best effort during shutdown
            pass

    async def ready(self) -> bool:
        if self.config.preload:
            targets = {
                f"{cfg.engine}:{cfg.artifact}"
                for cfg in self.models.values()
                if cfg.artifact
            }
            return bool(targets) and all(t in self._models for t in targets)
        for cfg in self.models.values():
            if cfg.artifact and not await asyncio.to_thread(os.path.exists, cfg.artifact):
                return False
        return True

    async def _get_model(self, artifact: str, engine_type: str) -> Any:
        key = f"{engine_type}:{artifact}"
        if key in self._models:
            return self._models[key]
        if not artifact:
            raise service_unavailable("model artifact is not configured")
        if not await asyncio.to_thread(os.path.exists, artifact):
            raise service_unavailable(f"model artifact not found: {artifact}")
        async with self._models_lock:
            if key in self._models:
                return self._models[key]
            model = await asyncio.to_thread(self._load_model_sync, artifact, engine_type)
            self._models[key] = model
            return model

    @staticmethod
    def _module_available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    def _check_acceleration(self, artifact: str) -> None:
        """Fail fast when an optional accelerator cannot be loaded.

        Upstream imports vllm/tensorrt lazily and fails with a bare
        ImportError, and it writes the exported vLLM model and the TensorRT
        plan inside the artifact directory. Both are easy to hit in
        production, so check them before spending a minute loading weights.
        """
        if self.config.load_vllm:
            if not self._module_available("vllm"):
                raise service_unavailable(
                    "engine.cosyvoice.load_vllm=true, but vllm is not installed; "
                    "install requirements-accel.lock or disable load_vllm"
                )
            vllm_dir = os.path.join(artifact, "vllm")
            if not os.path.isdir(vllm_dir) and not os.access(artifact, os.W_OK):
                raise service_unavailable(
                    "load_vllm exports the LLM into <artifact>/vllm on first "
                    f"start, but {artifact} is not writable; mount it writable "
                    "or pre-export the vllm directory"
                )
        if self.config.load_trt:
            if not self._module_available("tensorrt"):
                raise service_unavailable(
                    "engine.cosyvoice.load_trt=true, but tensorrt is not "
                    "installed; install requirements-accel.lock or disable "
                    "load_trt"
                )
            plan = any(Path(artifact).glob("flow.decoder.estimator.fp32.*.plan"))
            onnx = os.path.exists(
                os.path.join(artifact, "flow.decoder.estimator.fp32.onnx")
            )
            if not plan and not onnx:
                raise service_unavailable(
                    "load_trt=true needs flow.decoder.estimator.fp32.onnx to "
                    "build the engine, or a prebuilt "
                    "flow.decoder.estimator.fp32.*.plan"
                )
            if not plan and not os.access(artifact, os.W_OK):
                raise service_unavailable(
                    f"load_trt builds the TensorRT plan inside {artifact}, but "
                    "the directory is not writable; prebuild the plan or mount "
                    "it writable"
                )

    def _load_model_sync(self, artifact: str, engine_type: str) -> Any:
        if not artifact or not os.path.exists(artifact):
            raise service_unavailable(f"model artifact not found: {artifact}")
        self._check_acceleration(artifact)
        # Newer torchaudio routes all I/O through TorchCodec; decode prompt
        # wavs with soundfile so the engine does not depend on that pairing.
        # Upstream also imports modelscope at module import time and only uses
        # it for repo-id checkpoints, which this runtime never passes.
        from .cosyvoice_compat import (
            install_modelscope_stub,
            install_vllm_engine_overrides,
            install_wav_loader,
            install_wetext_assets,
        )

        install_modelscope_stub()
        try:
            from cosyvoice.cli import cosyvoice as cv
        except ImportError as exc:  # pragma: no cover - depends on deployment
            raise service_unavailable(
                "CosyVoice package is not installed; install the cosyvoice extra "
                "or set engine.type=mock"
            ) from exc

        install_wav_loader()
        install_wetext_assets()
        if self.config.load_vllm and not install_vllm_engine_overrides(
            self.config.vllm.overrides()
        ):
            logger.warning(
                "could not apply vLLM engine sizing; upstream hardcodes "
                "gpu_memory_utilization=0.2, which may not start on this GPU"
            )
        if self.config.tf32:
            self._enable_tf32()

        class_name = {
            "cosyvoice3": "CosyVoice3",
            "cosyvoice2": "CosyVoice2",
            "cosyvoice": "CosyVoice",
        }.get(engine_type, "CosyVoice3")
        cls = getattr(cv, class_name, None) or getattr(cv, "CosyVoice", None)
        if cls is None:
            raise service_unavailable(f"no CosyVoice class found for engine {engine_type}")

        kwargs: dict[str, Any] = {}
        try:
            params = inspect.signature(cls).parameters
        except (TypeError, ValueError):
            params = {}
        for name in (
            "load_jit",
            "load_trt",
            "load_vllm",
            "fp16",
            "trt_concurrent",
        ):
            if name in params:
                kwargs[name] = getattr(self.config, name)
        logger.info("loading CosyVoice model: %s (%s)", artifact, class_name)
        model = cls(artifact, **kwargs)
        if self.config.load_vllm or self.config.load_trt:
            logger.info(
                "accelerations active: vllm=%s trt=%s trt_concurrent=%d",
                self.config.load_vllm,
                self.config.load_trt,
                self.config.trt_concurrent,
            )
        self._register_prompt_voices(model, artifact, engine_type)
        return model

    @staticmethod
    def _enable_tf32() -> None:
        try:
            import torch
        except ImportError:
            return
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            logger.info("TF32 matmul enabled")
        except Exception:  # noqa: BLE001 - older torch / non-CUDA builds
            logger.warning("could not enable TF32", exc_info=True)

    @staticmethod
    def _prompt_fingerprint(voice: VoiceConfig) -> str:
        raw = f"{voice.prompt_wav}\0{voice.prompt_text}".encode()
        return "runtime-" + hashlib.sha1(raw).hexdigest()[:16]

    @staticmethod
    def _uses_zero_shot(voice: VoiceConfig) -> bool:
        if voice.mode == "zero_shot":
            return True
        return voice.mode == "auto" and bool(voice.prompt_wav and voice.prompt_text)

    @staticmethod
    def _normalize_prompt_text(model: Any, text: str) -> str:
        frontend = getattr(model, "frontend", None)
        normalize = getattr(frontend, "text_normalize", None)
        if normalize is None:
            return text
        try:
            return str(normalize(text, split=False))
        except Exception:  # noqa: BLE001 - optional text frontend
            logger.warning("prompt text normalization failed; using raw text", exc_info=True)
            return text

    def _register_prompt_voices(
        self, model: Any, artifact: str, engine_type: str
    ) -> None:
        """Pre-compute zero-shot prompt features once per configured voice.

        ``add_zero_shot_spk`` stores speech tokens, prompt features and the
        speaker embedding in the frontend so later requests only encode the
        text. It is a pure latency optimization: any failure leaves the normal
        (per-request extraction) path in place.
        """
        if not self.config.cache_prompt_features:
            return
        add = getattr(model, "add_zero_shot_spk", None)
        if add is None:
            logger.info(
                "%s does not expose add_zero_shot_spk; zero-shot prompt "
                "features are recomputed per request",
                engine_type,
            )
            return

        scope = f"{engine_type}:{artifact}"
        registered = 0
        for name, voice in self.voices.items():
            if not self._uses_zero_shot(voice):
                continue
            if not voice.prompt_wav or not voice.prompt_text:
                continue
            fingerprint = self._prompt_fingerprint(voice)
            if (scope, fingerprint) in self._prompt_ids:
                continue
            if not os.path.exists(voice.prompt_wav):
                logger.warning(
                    "voice %s: prompt_wav not found, prompt features will be "
                    "extracted per request: %s",
                    name,
                    voice.prompt_wav,
                )
                continue
            try:
                prompt_text = self._normalize_prompt_text(model, voice.prompt_text)
                add(prompt_text, voice.prompt_wav, fingerprint)
            except Exception:  # noqa: BLE001 - optimization only
                logger.warning(
                    "voice %s: failed to cache zero-shot prompt features",
                    name,
                    exc_info=True,
                )
                continue
            self._prompt_ids[(scope, fingerprint)] = fingerprint
            registered += 1
        if registered:
            logger.info(
                "cached zero-shot prompt features for %d voice(s) on %s",
                registered,
                artifact,
            )

    def _call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn(*args, **kwargs)
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not has_var_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return fn(*args, **kwargs)

    def _build_generator(self, model: Any, request: SynthesisRequest) -> Any:
        voice = request.voice_config
        mode = voice.mode
        if mode == "auto":
            if voice.prompt_wav and voice.prompt_text:
                mode = "zero_shot"
            elif voice.prompt_wav:
                mode = "instruct2"
            else:
                mode = "sft"

        instruct = request.instruction
        if instruct is None:
            instruct = voice.instruct or None
        stream = True
        speed = max(0.5, min(2.0, float(request.rate)))

        if mode == "zero_shot":
            if not voice.prompt_wav or not voice.prompt_text:
                raise invalid_parameter(
                    "zero_shot voice requires prompt_wav and prompt_text"
                )
            zero_shot_spk_id = ""
            if self.config.cache_prompt_features:
                scope = f"{request.model_config.engine}:{request.model_config.artifact}"
                zero_shot_spk_id = self._prompt_ids.get(
                    (scope, self._prompt_fingerprint(voice)), ""
                )
            return self._call(
                model.inference_zero_shot,
                request.text,
                voice.prompt_text,
                voice.prompt_wav,
                zero_shot_spk_id=zero_shot_spk_id,
                stream=stream,
                speed=speed,
            )

        if mode == "instruct2":
            if not voice.prompt_wav:
                raise invalid_parameter("instruct2 voice requires prompt_wav")
            if not instruct:
                raise invalid_parameter("instruct2 voice requires instruction text")
            return self._call(
                model.inference_instruct2,
                request.text,
                instruct,
                voice.prompt_wav,
                stream=stream,
                speed=speed,
            )

        if not voice.spk_id:
            raise invalid_parameter("sft voice requires spk_id")
        return self._call(
            model.inference_sft,
            request.text,
            voice.spk_id,
            instruct_text=instruct or "",
            stream=stream,
            speed=speed,
        )

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed % (2**32))
        except Exception:  # noqa: BLE001 - optional dependency
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:  # noqa: BLE001 - optional dependency
            pass

    @staticmethod
    def _inference_mode() -> Iterator[None]:
        try:
            import torch

            return torch.inference_mode()
        except Exception:  # noqa: BLE001 - torch is optional for the gateway
            return nullcontext()

    @staticmethod
    def _to_pcm(output: Any) -> bytes:
        speech = output.get("tts_speech") if isinstance(output, dict) else output
        if speech is None:
            return b""
        if hasattr(speech, "detach"):
            speech = speech.detach().cpu().float().numpy()
        import numpy as np

        array = np.asarray(speech, dtype=np.float32).reshape(-1)
        np.clip(array, -1.0, 1.0, out=array)
        return (array * 32767.0).astype("<i2").tobytes()

    @staticmethod
    def _worker_error(exc: BaseException) -> APIError:
        if isinstance(exc, APIError):
            return exc
        name = exc.__class__.__name__.lower()
        text = str(exc).lower()
        if "outofmemory" in name or "out of memory" in text:
            return service_unavailable("engine resource exhausted")
        return internal_error("model inference failed")

    async def synthesize(self, request: SynthesisRequest) -> AsyncIterator[AudioChunk]:
        async with self._semaphore:
            model = await self._get_model(
                request.model_config.artifact,
                request.model_config.engine,
            )
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[AudioChunk | APIError | None] = asyncio.Queue(
                maxsize=max(1, int(self.config.queue_size))
            )
            cancel = threading.Event()
            try:
                sample_rate = int(
                    getattr(model, "sample_rate", self.config.sample_rate)
                    or self.config.sample_rate
                )
            except (TypeError, ValueError):
                sample_rate = int(self.config.sample_rate)

            def enqueue(item: AudioChunk | APIError | None, timeout: float = 5.0) -> bool:
                if loop.is_closed():
                    return False
                try:
                    future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                    future.result(timeout=timeout)
                    return True
                except (concurrent.futures.TimeoutError, RuntimeError):
                    cancel.set()
                    return False

            def schedule_put(item: AudioChunk | APIError | None) -> None:
                """Deliver an item even if the bounded queue is temporarily full.

                This prevents a slow client from leaving the consumer blocked on
                ``queue.get()`` after the worker has exited.
                """
                if loop.is_closed():
                    return

                async def put() -> None:
                    try:
                        await asyncio.wait_for(queue.put(item), timeout=30.0)
                    except (asyncio.TimeoutError, RuntimeError):
                        pass

                asyncio.run_coroutine_threadsafe(put(), loop)

            def worker() -> None:
                generator = None
                try:
                    self._set_seed(request.seed)
                    with self._inference_mode():
                        generator = self._build_generator(model, request)
                        for output in generator:
                            if cancel.is_set():
                                break
                            pcm = self._to_pcm(output)
                            if pcm and not enqueue(
                                AudioChunk(pcm=pcm, sample_rate=sample_rate)
                            ):
                                break
                except BaseException as exc:  # propagate to async consumer
                    if not cancel.is_set():
                        logger.exception(
                            "cosyvoice worker failed: request_id=%s",
                            request.request_id,
                        )
                        if not enqueue(self._worker_error(exc)):
                            schedule_put(self._worker_error(exc))
                finally:
                    close = getattr(generator, "close", None)
                    if close is not None:
                        try:
                            close()
                        except Exception:  # noqa: BLE001 - best effort
                            pass
                    schedule_put(None)

            context = contextvars.copy_context()
            thread = threading.Thread(
                target=context.run,
                args=(worker,),
                name=f"cosyvoice-{request.request_id[:8]}",
                daemon=True,
            )
            thread.start()
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, APIError):
                        raise item
                    yield item
            finally:
                cancel.set()
                await asyncio.to_thread(
                    thread.join,
                    max(0.0, float(self.config.cancel_grace_s)),
                )
                if thread.is_alive():
                    logger.warning(
                        "cosyvoice worker did not stop within %.1fs: request_id=%s",
                        self.config.cancel_grace_s,
                        request.request_id,
                    )
