"""Streaming Paraformer (FunASR online model): 600 ms chunks with rolling cache.

SenseVoiceSmall is an offline model: it needs the whole utterance and therefore
cannot show partial text.  This class wraps the online Paraformer so the call UI
can display `[识别] ...` while the user is still speaking.

The model returns incremental text for each 600 ms chunk; `text` is the
concatenation of the pieces seen so far.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config

_ENC_LOOK_BACK = 4
_DEC_LOOK_BACK = 1


class StreamingASR:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rate = cfg.input_rate
        # FunASR chunk units are 60 ms: [0, n, n/2] -> n*60 ms of current audio
        # and n/2*60 ms of look-ahead. The online Paraformer stays accurate down
        # to 480 ms; 300 ms or less causes insertions/duplications.
        self.chunk_ms = max(240, int(round(cfg.asr_stream_chunk_ms / 60.0)) * 60)
        n = self.chunk_ms // 60
        self.chunk_size = [0, n, max(1, n // 2)]
        self.chunk = n * 960                 # 960 samples per 60 ms at 16 kHz
        self.model = None
        self.cache: dict = {}
        self.buf = np.zeros(0, dtype=np.float32)
        self.parts: list[str] = []
        self.on_partial = None                # callback(text: str)

    # ------------------------------------------------------------------ load
    def load(self):
        from funasr import AutoModel

        model = self.cfg.asr_stream_model
        local = Path.home() / ".cache" / "modelscope" / "hub" / "models" / model
        if local.exists():
            model = str(local)
        device = self.cfg.asr_device
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"
        print(f"[ASR] 加载流式 Paraformer ({device}, {self.chunk_ms}ms 分块) ...", flush=True)
        if self.chunk_ms < 480:
            print(f"[ASR] 警告: {self.chunk_ms}ms 分块低于模型稳定范围，准确率可能下降", flush=True)
        self.model = AutoModel(
            model=model,
            device=device,
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
        )
        try:
            self._generate(np.zeros(self.chunk, dtype=np.float32), final=False, cache={})
        except Exception:
            pass
        print("[ASR] 流式识别就绪", flush=True)

    # ------------------------------------------------------------------ state
    @property
    def text(self) -> str:
        return "".join(self.parts)

    def _generate(self, audio: np.ndarray, final: bool, cache: dict) -> str:
        result = self.model.generate(
            input=np.asarray(audio, dtype=np.float32),
            cache=cache,
            is_final=final,
            chunk_size=self.chunk_size,
            encoder_chunk_look_back=_ENC_LOOK_BACK,
            decoder_chunk_look_back=_DEC_LOOK_BACK,
        )
        if not result:
            return ""
        return str(result[0].get("text", ""))

    def reset(self, seed: np.ndarray | None = None):
        """Start a new utterance, optionally seeded with VAD pre-roll audio."""
        self.cache = {}
        self.parts = []
        self.buf = np.zeros(0, dtype=np.float32)
        if seed is not None and len(seed):
            self.buf = np.asarray(seed, dtype=np.float32).reshape(-1)
            self._drain()

    def _drain(self):
        while len(self.buf) >= self.chunk:
            piece = self.buf[:self.chunk]
            self.buf = self.buf[self.chunk:]
            self._append(self._generate(piece, final=False, cache=self.cache))

    def feed(self, audio: np.ndarray) -> str:
        """Feed 20 ms-scale audio; returns the updated partial text."""
        self.buf = np.concatenate([self.buf, np.asarray(audio, dtype=np.float32).reshape(-1)])
        self._drain()
        return self.text

    def finalize(self) -> str:
        """Flush the tail and return the final streaming transcript."""
        if len(self.buf) > 0:
            self._append(self._generate(self.buf, final=True, cache=self.cache))
            self.buf = np.zeros(0, dtype=np.float32)
        else:
            # Exact multiple of the chunk size: send a tiny final silence so the
            # decoder gets an is_final=True step.
            silence = np.zeros(int(0.2 * self.rate), dtype=np.float32)
            self._append(self._generate(silence, final=True, cache=self.cache))
        text = self.text
        self.cache = {}
        return text

    def _append(self, piece: str):
        if piece:
            self.parts.append(piece)
            if self.on_partial is not None:
                self.on_partial(self.text)
