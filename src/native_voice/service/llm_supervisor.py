"""Start/stop the local llama.cpp server for the conversation API."""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import httpx

from ..config.schema import ServiceConfig

log = logging.getLogger(__name__)


class LLMSupervisor:
    def __init__(self, cfg: ServiceConfig):
        self.cfg = cfg
        self.log_file = None
        self.proc: subprocess.Popen | None = None

    @property
    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.cfg.llm.port}/health"

    async def _healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                return (await client.get(self.health_url)).status_code == 200
        except Exception:
            return False

    async def start(self, timeout: float = 120.0) -> None:
        if await self._healthy():
            log.info("reusing llama-server on port %s", self.cfg.llm.port)
            return
        if self.cfg.llm.mode != "spawn":
            raise RuntimeError(f"llama-server is not reachable at {self.health_url}")
        data_dir = Path(self.cfg.root) / ".data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(data_dir / "llama-server.log", "w", encoding="utf-8")
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            [
                str(self.cfg.llm.server),
                "-m",
                str(self.cfg.llm.model),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.cfg.llm.port),
                "-c",
                str(self.cfg.llm.ctx),
                "-b",
                str(self.cfg.llm.batch_size),
                "-ub",
                str(self.cfg.llm.ubatch_size),
                "-ngl",
                str(self.cfg.llm.n_gpu_layers),
                "-t",
                str(self.cfg.llm.threads),
                "--threads-batch",
                str(self.cfg.llm.batch_threads),
                "-np",
                str(self.cfg.llm.slots),
                # Use the chat template embedded in the GGUF. Required for
                # chat_template_kwargs (Qwen3 thinking switch) and for parsing
                # tool calls into OpenAI-compatible tool_calls.
                "--jinja",
                *(
                    ["-ctk", self.cfg.llm.kv_cache_type, "-ctv", self.cfg.llm.kv_cache_type, "-fa", "on"]
                    if self.cfg.llm.kv_cache_type != "f16"
                    else []
                ),
            ],
            cwd=str(Path(self.cfg.llm.server).parent),
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            creationflags=creation,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self._healthy():
                log.info("llama-server ready on port %s", self.cfg.llm.port)
                return
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited (code {self.proc.returncode}); "
                    "see .data/llama-server.log"
                )
            time.sleep(0.5)
        raise RuntimeError("llama-server startup timeout")

    async def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except Exception:
                self.proc.kill()
        self.proc = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
