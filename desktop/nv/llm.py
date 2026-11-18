"""Streaming chat client for the local llama.cpp server (CPU)."""
from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import requests

from .config import Config


def _suffix_prefix_len(text: str, token: str) -> int:
    """Length of the longest suffix of `text` that is a prefix of `token`."""
    for n in range(min(len(text), len(token) - 1), 0, -1):
        if text.endswith(token[:n]):
            return n
    return 0


# Assembled from parts: some checkouts rewrite the literal tag on the way in.
_TAG_OPEN = "<" + "think" + ">"
_TAG_CLOSE = "<" + "/" + "think" + ">"


class ThinkFilter:
    """Strip Qwen3 reasoning blocks that may be split across deltas."""

    def __init__(self):
        self.pending = ""
        self.inside = False

    def feed(self, text: str) -> str:
        self.pending += text
        out: list[str] = []
        while True:
            if self.inside:
                idx = self.pending.find(_TAG_CLOSE)
                if idx >= 0:
                    self.pending = self.pending[idx + len(_TAG_CLOSE):]
                    self.inside = False
                    continue
                keep = _suffix_prefix_len(self.pending, _TAG_CLOSE)
                self.pending = self.pending[-keep:] if keep else ""
                break

            idx = self.pending.find(_TAG_OPEN)
            if idx >= 0:
                out.append(self.pending[:idx])
                self.pending = self.pending[idx + len(_TAG_OPEN):]
                self.inside = True
                continue

            # The opening tag may be dropped upstream; a bare closing tag then
            # marks the end of reasoning that must not be spoken.
            idx = self.pending.find(_TAG_CLOSE)
            if idx >= 0:
                self.pending = self.pending[idx + len(_TAG_CLOSE):]
                continue

            keep = max(
                _suffix_prefix_len(self.pending, _TAG_OPEN),
                _suffix_prefix_len(self.pending, _TAG_CLOSE),
            )
            if keep:
                out.append(self.pending[:-keep])
                self.pending = self.pending[-keep:]
            else:
                out.append(self.pending)
                self.pending = ""
            break
        return "".join(out)

    def flush(self) -> str:
        text, self.pending = self.pending, ""
        if self.inside:
            return ""
        if text in (_TAG_OPEN[: len(text)], _TAG_CLOSE[: len(text)]):
            return ""
        return text


class LlamaClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base = f"http://127.0.0.1:{cfg.llama_port}"
        self._proc: subprocess.Popen | None = None
        self._log_path = Path(cfg.log_dir) / "llama-server.log"

    # -------------------------------------------------------------- lifecycle
    def _healthy(self) -> bool:
        try:
            return requests.get(self.base + "/health", timeout=1.0).status_code == 200
        except Exception:
            return False

    def start(self, timeout: float = 90.0) -> None:
        if self._healthy():
            print(f"[LLM] 复用已运行的 llama-server :{self.cfg.llama_port}", flush=True)
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        args = [
            str(self.cfg.llama_server),
            "-m", str(self.cfg.llm_model),
            "--host", "127.0.0.1",
            "--port", str(self.cfg.llama_port),
            "-c", str(self.cfg.llama_ctx),
            "-t", str(self.cfg.llama_threads),
            "--threads-batch", str(self.cfg.llama_batch_threads),
            "-np", "1",
            # 使用 GGUF 内嵌 chat template（enable_thinking 等参数依赖它）。
            "--jinja",
        ]
        print(f"[LLM] 启动 {self.cfg.llm_model.name} ...", flush=True)
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._log = open(self._log_path, "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            args,
            cwd=str(Path(self.cfg.llama_server).parent),
            stdout=self._log,
            stderr=subprocess.STDOUT,
            creationflags=creation,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._healthy():
                print(f"[LLM] 就绪 ({time.time() - (deadline - timeout):.1f}s)", flush=True)
                return
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server 退出，日志: {self._log_path}")
            time.sleep(0.25)
        raise RuntimeError(f"llama-server 启动超时，日志: {self._log_path}")

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None

    def warmup(self, system_prompt: str) -> None:
        """Prime llama-server's prompt cache with the system prefix.

        The first real turn then only prefills the short user text instead of the
        whole system prompt, cutting ~0.5-0.8 s off the first token.
        """
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "你好"},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.time()
        try:
            r = requests.post(self.base + "/v1/chat/completions", json=payload, timeout=60)
            if r.status_code == 200:
                print(f"[LLM] prompt cache 预热完成 ({time.time() - t0:.1f}s)", flush=True)
        except Exception as exc:
            print(f"[LLM] 预热失败: {exc!r}", flush=True)

    # ------------------------------------------------------------------- chat
    def chat_stream(self, messages: list[dict], cancel=None, max_tokens: int | None = None) -> Iterator[str]:
        payload = {
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        with requests.post(
            self.base + "/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(10, 120),
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
            filt = ThinkFilter()
            for raw in resp.iter_lines():
                if cancel is not None and cancel.is_set():
                    break
                if not raw:
                    continue
                if raw.startswith(b"data: "):
                    data = raw[6:]
                elif raw.startswith(b"data:"):
                    data = raw[5:]
                else:
                    continue
                data = data.strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data.decode("utf-8"))
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    text = filt.feed(piece)
                    if text:
                        yield text
            tail = filt.flush()
            if tail and (cancel is None or not cancel.is_set()):
                yield tail
