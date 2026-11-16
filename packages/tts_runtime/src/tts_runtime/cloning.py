from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import wave
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path

from cosyvoice_common.audio import ffmpeg_path
from cosyvoice_common.errors import (
    APIError,
    invalid_parameter,
    unsupported_format,
    unsupported_parameter,
)

from .settings import CloningConfig, VoiceConfig

logger = logging.getLogger("tts_runtime.cloning")

# Deliberately narrow: the upload is decoded by ffmpeg, not executed.
_ALLOWED_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".webm",
    ".flac",
}
_SAFE_ID = re.compile(r"^clone-[a-z0-9][a-z0-9-]{1,48}$")
_SAFE_NAME = re.compile(r"[^\w一-鿿 -]+", re.UNICODE)


class _WavUnsupported(Exception):
    """The WAV is valid but needs ffmpeg (bit depth, channels or rate)."""


@dataclass(slots=True)
class ClonedVoice:
    id: str
    name: str
    transcript: str
    created_at: float
    duration_s: float
    sample_rate: int
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ClonedVoiceStore:
    """Filesystem-backed zero-shot voices.

    One directory per voice:

        <dir>/<voice-id>/prompt.wav
        <dir>/<voice-id>/meta.json

    The runtime registers the returned ``VoiceConfig`` in the registry, so a
    cloned voice behaves exactly like a zero-shot voice from the YAML config.
    """

    def __init__(self, config: CloningConfig) -> None:
        self.config = config
        self._dir = Path(config.dir)
        self._voices: dict[str, tuple[ClonedVoice, VoiceConfig]] = {}
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        if not self.config.enabled:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("cannot create cloning dir %s: %s", self._dir, exc)
            return
        loaded = 0
        for meta_path in sorted(self._dir.glob("*/meta.json")):
            voice_id = meta_path.parent.name
            if not _SAFE_ID.match(voice_id):
                logger.warning("ignoring cloned voice with unsafe id: %s", voice_id)
                continue
            try:
                info, voice = self._read_voice(meta_path)
            except Exception:  # noqa: BLE001 - one bad directory must not block startup
                logger.warning("ignoring invalid cloned voice %s", meta_path.parent, exc_info=True)
                continue
            self._voices[voice_id] = (info, voice)
            loaded += 1
        if loaded:
            logger.info("loaded %d cloned voice(s) from %s", loaded, self._dir)

    def _read_voice(self, meta_path: Path) -> tuple[ClonedVoice, VoiceConfig]:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        voice_id = meta_path.parent.name
        prompt_wav = meta_path.parent / "prompt.wav"
        if not prompt_wav.is_file():
            raise ValueError("prompt.wav is missing")
        prompt_text = str(raw["prompt_text"])
        if not prompt_text:
            raise ValueError("prompt_text is empty")
        info = ClonedVoice(
            id=voice_id,
            name=str(raw.get("name") or voice_id),
            transcript=str(raw.get("transcript") or ""),
            created_at=float(raw.get("created_at") or 0.0),
            duration_s=float(raw.get("duration_s") or 0.0),
            sample_rate=int(raw.get("sample_rate") or 0),
            source=str(raw.get("source") or "upload"),
        )
        voice = VoiceConfig(
            mode="zero_shot",
            prompt_wav=str(prompt_wav),
            prompt_text=prompt_text,
        )
        return info, voice

    def list(self) -> list[ClonedVoice]:
        return sorted(
            (info for info, _ in self._voices.values()),
            key=lambda item: item.created_at,
            reverse=True,
        )

    def get(self, voice_id: str) -> ClonedVoice | None:
        item = self._voices.get(voice_id)
        return item[0] if item else None

    def voice_config(self, voice_id: str) -> VoiceConfig | None:
        item = self._voices.get(voice_id)
        return item[1] if item else None

    def voice_configs(self) -> dict[str, VoiceConfig]:
        return {voice_id: voice for voice_id, (_, voice) in self._voices.items()}

    async def create(
        self,
        data: bytes,
        filename: str,
        transcript: str,
        name: str = "",
        source: str = "upload",
    ) -> ClonedVoice:
        if not self.config.enabled:
            raise APIError("AccessDenied", "voice cloning is disabled", 403)
        if not data:
            raise invalid_parameter("audio file is empty")
        if len(data) > self.config.max_upload_bytes:
            raise APIError(
                "RequestTooLarge",
                f"audio file exceeds {self.config.max_upload_bytes} bytes",
                413,
            )

        transcript = (transcript or "").strip()
        if not transcript:
            raise invalid_parameter(
                "transcript is required: enter the exact words spoken in the "
                "reference recording"
            )
        if len(transcript) > self.config.max_transcript_chars:
            raise invalid_parameter(
                f"transcript exceeds {self.config.max_transcript_chars} characters"
            )

        suffix = Path(filename or "").suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise unsupported_format(
                "unsupported audio type; upload wav, mp3, m4a, aac, ogg, opus, "
                "webm or flac"
            )
        if source not in {"upload", "record"}:
            source = "upload"

        async with self._lock:
            voice_id = self._new_id(name)
            info, voice = await asyncio.to_thread(
                self._write_voice, voice_id, data, suffix, transcript, name, source
            )
            self._voices[voice_id] = (info, voice)
            logger.info(
                "cloned voice created: id=%s name=%s duration=%.2fs source=%s",
                info.id,
                info.name,
                info.duration_s,
                info.source,
            )
            return info

    def _write_voice(
        self,
        voice_id: str,
        data: bytes,
        suffix: str,
        transcript: str,
        name: str,
        source: str,
    ) -> tuple[ClonedVoice, VoiceConfig]:
        target = self._dir / voice_id
        target.mkdir(parents=True, exist_ok=False)
        source_path = target / f"upload{suffix}"
        prompt_wav = target / "prompt.wav"
        try:
            source_path.write_bytes(data)
            sample_rate, duration = self._normalize_audio(source_path, prompt_wav, suffix)
            if duration < self.config.min_duration_s:
                raise invalid_parameter(
                    f"reference audio is too short ({duration:.2f}s); "
                    f"need at least {self.config.min_duration_s:.1f}s"
                )
            if duration > self.config.max_duration_s:
                raise invalid_parameter(
                    f"reference audio is too long ({duration:.2f}s); "
                    f"maximum is {self.config.max_duration_s:.1f}s"
                )

            prompt_text = self._format_prompt_text(transcript)
            clean_name = _SAFE_NAME.sub("", (name or "").strip())[:64].strip()
            if not clean_name:
                clean_name = f"clone {time.strftime('%m-%d %H:%M')}"

            info = ClonedVoice(
                id=voice_id,
                name=clean_name,
                transcript=transcript,
                created_at=time.time(),
                duration_s=round(duration, 3),
                sample_rate=sample_rate,
                source=source,
            )
            meta = {
                **info.to_dict(),
                "prompt_text": prompt_text,
            }
            tmp = target / "meta.json.tmp"
            tmp.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, target / "meta.json")
            voice = VoiceConfig(
                mode="zero_shot",
                prompt_wav=str(prompt_wav),
                prompt_text=prompt_text,
            )
            return info, voice
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        finally:
            try:
                source_path.unlink()
            except OSError:
                pass

    def _normalize_audio(
        self, source: Path, target: Path, suffix: str
    ) -> tuple[int, float]:
        if suffix == ".wav":
            try:
                return self._normalize_wav(source, target)
            except _WavUnsupported:
                pass
        exe = ffmpeg_path()
        if not exe:
            raise unsupported_parameter(
                "ffmpeg is required to decode this audio type; upload a "
                "16-bit PCM WAV or install ffmpeg"
            )
        cmd = [
            exe,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            # Bound the decode work even if a small container hides hours of
            # audio; the duration check below is the real limit.
            "-t",
            "60",
            "-ac",
            "1",
            "-ar",
            str(int(self.config.prompt_sample_rate)),
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=60, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise invalid_parameter("audio decoding timed out") from exc
        if result.returncode != 0 or not target.is_file():
            detail = result.stderr.decode("utf-8", "replace").strip()[-300:]
            raise invalid_parameter(f"could not decode audio: {detail or 'ffmpeg failed'}")
        try:
            channels, rate, frames = _wav_shape(target)
        except (wave.Error, EOFError) as exc:
            raise invalid_parameter("decoded audio has no playable audio stream") from exc
        if channels != 1:
            raise invalid_parameter("decoded audio is not mono")
        return rate, frames / float(rate)

    def _normalize_wav(self, source: Path, target: Path) -> tuple[int, float]:
        try:
            with wave.open(str(source), "rb") as reader:
                channels = reader.getnchannels()
                width = reader.getsampwidth()
                rate = reader.getframerate()
                frames = reader.getnframes()
                pcm = reader.readframes(frames)
        except (wave.Error, EOFError) as exc:
            raise _WavUnsupported() from exc
        if width != 2 or channels not in (1, 2) or rate < 16_000:
            raise _WavUnsupported()
        if channels == 2:
            samples = array("h")
            samples.frombytes(pcm)
            if sys.byteorder == "big":
                samples.byteswap()
            mono = array(
                "h",
                (
                    (samples[index] + samples[index + 1]) // 2
                    for index in range(0, len(samples), 2)
                ),
            )
            if sys.byteorder == "big":
                mono.byteswap()
            pcm = mono.tobytes()
            frames = len(mono)
        with wave.open(str(target), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(pcm)
        return rate, frames / float(rate)

    @staticmethod
    def _format_prompt_text(transcript: str) -> str:
        # CosyVoice3 splits the prompt on this marker; text after it is the
        # transcript of the reference recording.
        if "<|endofprompt|>" in transcript:
            return transcript
        return "<|endofprompt|>" + transcript

    def _new_id(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", (name or "voice").lower()).strip("-")[:24]
        slug = slug or "voice"
        for _ in range(10):
            voice_id = f"clone-{slug}-{uuid.uuid4().hex[:8]}"
            if voice_id not in self._voices and not (self._dir / voice_id).exists():
                return voice_id
        raise APIError("InternalError", "could not allocate a voice id", 500)

    async def delete(self, voice_id: str) -> bool:
        if not _SAFE_ID.match(voice_id):
            return False
        async with self._lock:
            item = self._voices.pop(voice_id, None)
            if item is None:
                return False
            directory = self._dir / voice_id
            await asyncio.to_thread(shutil.rmtree, directory, True)
            logger.info("cloned voice deleted: id=%s", voice_id)
            return True


def _wav_shape(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as reader:
        return reader.getnchannels(), reader.getframerate(), reader.getnframes()
