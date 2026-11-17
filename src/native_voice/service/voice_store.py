"""Persistent store for cloned voices.

Metadata lives in ``.data/voices/voices.json``; reference audio is saved as
24 kHz mono PCM16 wav next to it. The TTS engine loads this store at startup and
the gateway writes it; both processes must share the configured directory
(a mounted volume in Docker).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"


class VoiceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "voices.json"
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self.meta_path.exists():
            return {"voices": []}
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if not isinstance(data.get("voices"), list):
                return {"voices": []}
            return data
        except Exception:
            return {"voices": []}

    def _save(self, data: dict) -> None:
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.meta_path)

    def list(self) -> list[dict]:
        with self._lock:
            data = self._load()
        out = []
        for item in data["voices"]:
            wav = self.root / item.get("wav", "")
            if not wav.exists():
                continue
            out.append({**item, "path": str(wav.resolve())})
        return out

    def get(self, voice_id: str) -> dict | None:
        for item in self.list():
            if item.get("id") == voice_id:
                return item
        return None

    def add(
        self,
        voice_id: str,
        display_name: str,
        transcript: str,
        prompt_text: str,
        description: str,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> dict:
        import soundfile as sf

        wav_name = f"{voice_id}.wav"
        wav_path = self.root / wav_name
        sf.write(str(wav_path), np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")
        record = {
            "id": voice_id,
            "name": display_name,
            "transcript": transcript,
            "prompt_text": prompt_text,
            "description": description or f"克隆 · {display_name}",
            "wav": wav_name,
            "created_at": time.time(),
        }
        with self._lock:
            data = self._load()
            data["voices"] = [v for v in data["voices"] if v.get("id") != voice_id]
            data["voices"].append(record)
            self._save(data)
        return {**record, "path": str(wav_path.resolve())}

    def delete(self, voice_id: str) -> bool:
        with self._lock:
            data = self._load()
            kept = [v for v in data["voices"] if v.get("id") != voice_id]
            if len(kept) == len(data["voices"]):
                return False
            removed = [v for v in data["voices"] if v.get("id") == voice_id]
            data["voices"] = kept
            self._save(data)
        for item in removed:
            wav = (self.root / item.get("wav", "")).resolve()
            try:
                if wav.is_relative_to(self.root.resolve()):
                    wav.unlink(missing_ok=True)
            except Exception:
                pass
        return True
