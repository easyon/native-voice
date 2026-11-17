"""Dev/single-box supervisor for the two GPU engine processes.

Production uses Docker/systemd and `engine.mode=external`; this class exists so
`--backend worker` works on a developer machine with one command.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from ..config.schema import ProviderSpec, ServiceConfig

log = logging.getLogger(__name__)


def _interpreter(path: str) -> str:
    """Resolve a venv interpreter, mapping Windows/Linux layout when needed.

    The checked-in config uses ``Scripts/python.exe`` (Windows dev). Production
    Linux images may reuse the same config; a missing Windows path maps to the
    sibling ``bin/python`` (and vice versa) before we report an error.
    """
    p = Path(path)
    if p.exists():
        return str(p)
    if p.name == "python.exe" and p.parent.name == "Scripts":
        alt = p.parent.parent / "bin" / "python"
    elif p.name == "python" and p.parent.name == "bin":
        alt = p.parent.parent / "Scripts" / "python.exe"
    else:
        return str(p)
    return str(alt) if alt.exists() else str(p)


class EngineSupervisor:
    def __init__(self, cfg: ServiceConfig, config_path: str | Path):
        self.cfg = cfg
        self.config_path = str(Path(config_path).resolve())
        self.procs: list[tuple[subprocess.Popen, object]] = []

    async def start(self) -> None:
        for kind, port, http in (
            ("asr", self.cfg.engine.asr_port, self.cfg.engine.asr_http),
            ("tts", self.cfg.engine.tts_port, self.cfg.engine.tts_http),
        ):
            if await self._ready(http):
                log.info("%s engine already running at %s", kind, http)
                continue
            self._spawn(kind, port)
        await self._wait_ready()

    def _provider(self, kind: str) -> tuple[str, ProviderSpec | None]:
        name = self.cfg.engine.asr_provider if kind == "asr" else self.cfg.engine.tts_provider
        return name, (self.cfg.providers.get(kind) or {}).get(name)

    def _spawn(self, kind: str, port: int) -> None:
        name, spec = self._provider(kind)
        default = "paraformer" if kind == "asr" else "cosyvoice3"
        if spec is None and name != default:
            raise RuntimeError(
                f"{kind} provider {name!r} is not defined under providers.{kind} in the config"
            )
        python = _interpreter(getattr(spec, "python", "") or sys.executable)
        module = getattr(spec, "module", "") or "native_voice.engine.cli"
        if python != sys.executable and not Path(python).exists():
            raise RuntimeError(
                f"{kind} provider {name!r} points at missing interpreter {python}; "
                "run the provider setup script first"
            )
        data_dir = Path(self.cfg.root) / ".data"
        data_dir.mkdir(parents=True, exist_ok=True)
        log_path = data_dir / f"engine-{kind}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        env = os.environ.copy()
        src = str(Path(self.cfg.root) / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            [
                python,
                "-m",
                module,
                kind,
                "--config",
                self.config_path,
                "--port",
                str(port),
                "--provider",
                name,
            ],
            cwd=str(self.cfg.root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creation,
        )
        self.procs.append((proc, log_file))
        log.info(
            "spawned %s engine provider=%s python=%s pid=%s log=%s",
            kind, name, python, proc.pid, log_path,
        )

    async def _ready(self, http_url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.get(http_url.rstrip("/") + "/readyz")
                return resp.status_code == 200 and bool(resp.json().get("ready"))
        except Exception:
            return False

    async def _wait_ready(self, timeout: float = 180.0) -> None:
        deadline = time.monotonic() + timeout
        for kind, http in (
            ("asr", self.cfg.engine.asr_http),
            ("tts", self.cfg.engine.tts_http),
        ):
            while time.monotonic() < deadline:
                if await self._ready(http):
                    log.info("%s engine ready", kind)
                    break
                for proc, _ in self.procs:
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"{kind} engine exited with code {proc.returncode}; see .data/engine-{kind}.log"
                        )
                await asyncio.sleep(1.0)
            else:
                raise RuntimeError(f"{kind} engine did not become ready within {timeout:.0f}s")

    async def stop(self) -> None:
        for proc, log_file in self.procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
            try:
                log_file.close()
            except Exception:
                pass
        self.procs.clear()
