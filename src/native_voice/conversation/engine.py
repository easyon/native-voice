"""Async full-duplex conversation session for the browser API.

The engine keeps the tuned local behaviour: streaming partials, endpointing,
first-sentence TTS priority, barge-in, and merging a quick follow-up that
arrives before any assistant audio was sent. It talks to the same backends as
the Bailian-compatible API (local models or remote GPU engines).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

from ..api.protocol.bailian.writer import ConnectionWriter
from ..audio.pcm import float32_to_pcm16, pcm16_to_float32
from ..backends.base import Backends
from ..config.schema import ServiceConfig
from ..observability.metrics import METRICS
from . import events
from .ack import is_backchannel_text
from .params import VadParams
from .speech_gate import SpeechGate
from .splitter import SentenceSplitter
from .vad import EndpointVad, VadEvent

log = logging.getLogger(__name__)


@dataclass
class Turn:
    turn_id: str
    user_text: str
    t_ref: float
    t_endpoint: float
    t_text: float
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    tasks: list[asyncio.Task] = field(default_factory=list)
    text: str = ""
    index: int = 0
    t_llm: float = 0.0
    t_tts: float = 0.0
    metrics_sent: bool = False
    llm_done: bool = False
    audio_done: bool = False
    playback_started: bool = False
    playback_started_at: float = 0.0
    merged: bool = False
    history_saved: bool = False


class ConversationEngine:
    def __init__(self, cfg: ServiceConfig, backends: Backends, llm, writer: ConnectionWriter):
        self.cfg = cfg
        self.backends = backends
        self.llm = llm
        self.writer = writer
        self.speech_gate = SpeechGate(
            cfg.conversation.speech_gate_model,
            enabled=cfg.conversation.speech_gate,
            start_threshold=cfg.conversation.speech_start_threshold,
            silence_threshold=cfg.conversation.speech_silence_threshold,
        )
        self.vad = EndpointVad(VadParams.from_config(cfg.conversation), self.speech_gate)
        self.asr = backends.asr.create_session()
        self.history: list[dict] = []
        self.turn: Turn | None = None
        self.partial = ""
        self.last_partial = ""
        self.begin_ms = 0
        self.stream_ms = 0.0
        self.sentence_id = 1
        self.merge_pending = False
        self.merge_text = ""
        self.voice = cfg.conversation.voice
        self.system_prompt = cfg.llm.system_prompt
        self._closed = False
        self._bc_task: asyncio.Task | None = None
        self._bc_cache: dict[tuple[str, str], bytes] = {}
        self._bc_cancel = threading.Event()
        self._bc_utterance = 0
        self._bc_count = 0
        # Audio timeline (stream_ms), not wall clock: network bursts during a
        # test/replay must still count as "the user has been speaking".
        self._bc_started_ms = 0.0
        self._bc_last_ms = 0.0
        # Whether this user utterance began while the assistant was talking.
        self._user_over_turn = False
        # Audio-timeline position where the current user utterance was confirmed
        # by the VAD; used to fall back to a barge-in when ASR partials lag.
        self._speech_started_ms = 0.0

    async def start(self) -> None:
        await self.writer.send_json(
            events.session_ready(
                str(uuid.uuid4()),
                self.voice,
                self.cfg.conversation.output_rate,
                self.system_prompt,
            )
        )

    def set_system_prompt(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            text = self.cfg.llm.system_prompt
        self.system_prompt = text[:2000]

    def set_endpoint_ms(self, ms: int) -> None:
        ms = max(150, min(2000, int(ms)))
        self.vad.cfg.endpoint_silence_ms = ms
        self.cfg.conversation.endpoint_silence_ms = ms

    async def close(self) -> None:
        self._closed = True
        self._bc_utterance += 1
        self._bc_cancel.set()
        if self._bc_task is not None and not self._bc_task.done():
            self._bc_task.cancel()
        await self._cancel_turn("session_closed", emit=False)
        try:
            await self.asr.close()
        except Exception:
            pass

    # ------------------------------------------------------------- audio in
    async def feed_audio(self, pcm16: bytes) -> None:
        audio = pcm16_to_float32(pcm16)
        if audio.size == 0:
            return
        self.stream_ms += audio.size / self.cfg.conversation.input_rate * 1000.0
        events_in = self.vad.feed(audio)
        # Feed continuously so the model has the utterance onset even though the
        # VAD only declares speech after its 800 ms warmup / sustain window.
        text = await self.asr.feed(audio)
        for ev in events_in:
            if ev.kind == "start":
                await self._on_start(ev)
            elif ev.kind == "end":
                await self._on_end(ev)
        if self.vad.active:
            # Barge-in waits for real transcribed content, not just VAD energy
            # or an acknowledgement such as "嗯"/"对".
            await self._maybe_barge_in(text)
            await self._emit_partial(text)
            self._maybe_backchannel()

    async def _on_start(self, ev: VadEvent) -> None:
        self._bc_utterance += 1
        self._bc_cancel.set()
        self._bc_count = 0
        self._bc_started_ms = self.stream_ms
        self._bc_last_ms = 0.0
        self._user_over_turn = self.turn is not None
        self._speech_started_ms = self.stream_ms
        # Do not cancel the assistant here: VAD energy alone can be noise, a
        # cough, or a backchannel.  _maybe_barge_in() waits for an ASR partial
        # with content, then falls back to sustained speech if ASR lags.
        self.partial = ""
        self.last_partial = ""
        self.begin_ms = int(max(0.0, self.stream_ms - ev.duration * 1000.0))

    async def _on_end(self, ev: VadEvent) -> None:
        self._bc_utterance += 1
        self._bc_cancel.set()
        self._bc_started_ms = 0.0
        self._speech_started_ms = 0.0
        t_endpoint = time.monotonic()
        try:
            final = await self.asr.finalize()
        except Exception as exc:
            log.warning("ASR finalize failed: %r", exc)
            final = self.partial
        try:
            await self.asr.reset()
        except Exception:
            pass
        text = (final or self.partial).strip()
        self.partial = ""
        self.last_partial = ""
        over_turn = self._user_over_turn
        self._user_over_turn = False
        if not text:
            return
        if over_turn and is_backchannel_text(text):
            # The user only acknowledged the assistant ("嗯"/"对") while it
            # was talking; keep the reply going and do not create a new turn.
            log.info("ignored backchannel utterance while assistant speaking")
            return
        if self.turn is not None:
            # ASR did not expose a usable partial in time, but the final text
            # has content: stop the assistant before starting the new turn.
            # Keep the fast-follow-up merge when no audio had been produced.
            if (
                self.turn.t_tts == 0
                and time.monotonic() - self.turn.t_endpoint
                <= self.cfg.conversation.merge_window_s
            ):
                await self._begin_merge(self.turn)
            else:
                await self._cancel_turn("barge_in")
        t_ref = ev.last_voiced_time or (t_endpoint - self.cfg.conversation.endpoint_silence_ms / 1000.0)
        if self.merge_pending:
            sep = "" if self.merge_text and self.merge_text[-1] in "。！？!?；;，,、：:" else "，"
            text = (self.merge_text + sep + text).strip()
            self.merge_pending = False
            self.merge_text = ""
        log.info("user transcript: %s", text)
        end_ms = self.begin_ms + int(ev.duration * 1000)
        await self.writer.send_json(events.asr_final(text, self.sentence_id, self.begin_ms, end_ms))
        self.sentence_id += 1
        await self._start_turn(text, t_ref, t_endpoint)

    async def _maybe_barge_in(self, text: str) -> None:
        """Cancel the current reply once the user's speech has real content.

        This runs on every ASR partial, so the assistant stops as soon as the
        first meaningful words appear rather than waiting for the final
        transcript.  Pure fillers ("嗯", "对", "好的") keep the reply going.
        """
        turn = self.turn
        if turn is None or not self.vad.active:
            return
        if not text:
            # Partial transcripts can lag behind the microphone when the ASR
            # engine shares the GPU with TTS.  A user who keeps talking must
            # still be able to interrupt, so fall back to sustained speech.
            await self._barge_in_without_text(turn)
            return
        if is_backchannel_text(text):
            return
        if (
            turn.t_tts == 0
            and time.monotonic() - turn.t_endpoint
            <= self.cfg.conversation.merge_window_s
        ):
            await self._begin_merge(turn)
            return
        log.info("barge-in from partial transcript (chars=%d)", len(text))
        await self._cancel_turn("barge_in")

    async def _barge_in_without_text(self, turn: Turn) -> None:
        """Interrupt on sustained speech even if no ASR partial has arrived."""
        waited_ms = self.stream_ms - self._speech_started_ms
        if waited_ms < max(0, self.cfg.conversation.barge_content_wait_ms):
            return
        if (
            turn.t_tts == 0
            and time.monotonic() - turn.t_endpoint
            <= self.cfg.conversation.merge_window_s
        ):
            await self._begin_merge(turn)
            return
        log.info("barge-in from sustained speech (%.0f ms, no partial yet)", waited_ms)
        await self._cancel_turn("barge_in")

    async def _emit_partial(self, text: str) -> None:
        if not text or text == self.last_partial:
            return
        self.last_partial = text
        self.partial = text
        await self.writer.send_json(events.asr_partial(text, self.sentence_id, self.begin_ms))

    # ---------------------------------------------------------- backchannel
    def _maybe_backchannel(self) -> None:
        """Schedule a short acknowledgement while the user is still talking."""
        c = self.cfg.conversation
        if (
            not c.backchannel_enabled
            or self.turn is not None
            or not self.vad.active
            or not c.backchannel_phrases
        ):
            return
        if self._bc_task is not None and not self._bc_task.done():
            return
        now_ms = self.stream_ms
        if (
            self._bc_started_ms <= 0
            or now_ms - self._bc_started_ms
            < max(0.0, c.backchannel_min_speech_s) * 1000.0
        ):
            return
        if self._bc_count >= max(0, c.backchannel_max_per_utterance):
            return
        if (
            self._bc_last_ms
            and now_ms - self._bc_last_ms < max(0.0, c.backchannel_interval_s) * 1000.0
        ):
            return
        self._bc_count += 1
        self._bc_last_ms = now_ms
        self._bc_task = asyncio.create_task(
            self._emit_backchannel(self._bc_utterance, self._bc_count),
            name="conv-backchannel",
        )

    async def _emit_backchannel(self, utterance: int, index: int) -> None:
        try:
            phrases = self.cfg.conversation.backchannel_phrases
            phrase = phrases[(index - 1) % len(phrases)]
            key = (self.voice, phrase)
            pcm = self._bc_cache.get(key)
            if pcm is None:
                self._bc_cancel.clear()
                chunks = [
                    chunk
                    async for chunk in self.backends.tts.synthesize(
                        phrase, self.voice, self._bc_cancel, priority=2
                    )
                ]
                if not chunks or self._bc_cancel.is_set():
                    return
                pcm = b"".join(float32_to_pcm16(chunk) for chunk in chunks)
                self._bc_cache[key] = pcm
            if (
                utterance != self._bc_utterance
                or self.turn is not None
                or not self.vad.active
            ):
                return
            bc_id = uuid.uuid4().hex
            await self.writer.send_json(
                events.backchannel_start(
                    bc_id,
                    self.cfg.conversation.output_rate,
                    self.cfg.conversation.backchannel_volume,
                )
            )
            await self.writer.send_bytes(pcm)
            await self.writer.send_json(events.backchannel_done(bc_id))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("backchannel failed: %r", exc)
        finally:
            if self._bc_task is asyncio.current_task():
                self._bc_task = None

    # ------------------------------------------------------------ user input
    async def input_text(self, text: str) -> None:
        text = (text or "").strip()[:2000]
        if not text:
            return
        if self.turn is not None:
            await self._cancel_turn("text_input")
        self.merge_pending = False
        self.merge_text = ""
        log.info("user text input: %s", text)
        await self._start_turn(text, time.monotonic(), time.monotonic())

    async def interrupt(self) -> None:
        if self.turn is not None:
            await self._cancel_turn("client_interrupt")

    # --------------------------------------------------------------- turns
    def _messages(self) -> list[dict]:
        guidance = (
            "对话规则：这是同一通连续语音通话。不要重复上一轮已经说过的内容；"
            "如果用户是在回答你上一轮的问题，直接顺着回答，不要重新介绍或重问。"
        )
        system = f"{self.system_prompt.rstrip()}\n{guidance}".strip()
        messages = [{"role": "system", "content": system}] + list(self.history)
        budget = max(500, self.cfg.llm.prompt_char_budget)
        total = sum(len(m["content"]) for m in messages)
        while len(messages) > 1 and total > budget:
            total -= len(messages.pop(1)["content"])
        return messages

    def _save_assistant_history(self, turn: Turn) -> None:
        """Keep what the user actually heard, including a reply cut by barge-in."""
        if turn.history_saved or turn.merged:
            return
        text = turn.text.strip()
        if text:
            self.history.append({"role": "assistant", "content": text})
            self._trim_history()
        turn.history_saved = True

    def _trim_history(self) -> None:
        limit = max(2, self.cfg.llm.history_turns * 2)
        while len(self.history) > limit:
            self.history.pop(0)
        while self.history and self.history[0]["role"] != "user":
            self.history.pop(0)

    async def _start_turn(self, text: str, t_ref: float, t_endpoint: float) -> None:
        self.history.append({"role": "user", "content": text})
        self._trim_history()
        turn = Turn(
            turn_id=str(uuid.uuid4()),
            user_text=text,
            t_ref=t_ref,
            t_endpoint=t_endpoint,
            t_text=time.monotonic(),
        )
        self.turn = turn
        self.vad.set_agent_speaking(True)
        turn.tasks = [
            asyncio.create_task(self._llm_loop(turn), name="conv-llm"),
            asyncio.create_task(self._tts_loop(turn), name="conv-tts"),
        ]

    async def _llm_loop(self, turn: Turn) -> None:
        # Start TTS on the first short clause instead of waiting for a full
        # sentence. Very short limits (2/6 chars) put the first audio on the
        # wire ASAP; the trade-off is a slightly choppier first phrase.
        # After the first clause, use the normal sentence limits.
        splitter = SentenceSplitter(soft_limit=2, hard_limit=6)
        first_sentence = True
        messages = self._messages()
        log.info(
            "LLM turn roles=%s",
            [(m["role"], len(m["content"])) for m in messages],
        )
        try:
            async for delta in self.llm.chat_stream(messages, turn.cancel):
                if turn.cancel.is_set():
                    break
                if turn.t_llm == 0:
                    turn.t_llm = time.monotonic()
                turn.text += delta
                await self.writer.send_json(events.text_delta(turn.turn_id, delta))
                for sentence in splitter.feed(delta):
                    await turn.queue.put(sentence)
                    if first_sentence:
                        first_sentence = False
                        splitter.soft_limit = 18
                        splitter.hard_limit = 42
        except Exception as exc:
            log.warning("conversation LLM failed: %r", exc)
            if not turn.cancel.is_set():
                await self.writer.send_json(events.error("LLMError", str(exc)))
        finally:
            if not turn.cancel.is_set():
                for sentence in splitter.flush():
                    await turn.queue.put(sentence)
                await self.writer.send_json(events.text_done(turn.turn_id, turn.text))
            reply = turn.text.strip()
            if reply:
                log.info("assistant reply: %s", reply)
            self._save_assistant_history(turn)
            await turn.queue.put(None)
            turn.llm_done = True
            await self._maybe_metrics(turn)

    async def _tts_loop(self, turn: Turn) -> None:
        try:
            while True:
                sentence = await turn.queue.get()
                if sentence is None:
                    break
                if turn.cancel.is_set():
                    continue
                index = turn.index
                turn.index += 1
                await self.writer.send_json(
                    events.audio_start(turn.turn_id, index, self.cfg.conversation.output_rate)
                )
                try:
                    async for chunk in self.backends.tts.synthesize(
                        sentence, self.voice, turn.cancel, priority=0 if index == 0 else 1
                    ):
                        if turn.cancel.is_set():
                            break
                        if turn.t_tts == 0:
                            turn.t_tts = time.monotonic()
                            METRICS.tts_first_chunk.observe(turn.t_tts - turn.t_text)
                        await self.writer.send_bytes(float32_to_pcm16(chunk))
                except Exception as exc:
                    log.warning("conversation TTS failed: %r", exc)
                    if not turn.cancel.is_set():
                        await self.writer.send_json(events.error("TTSError", str(exc)))
                if not turn.cancel.is_set():
                    await self.writer.send_json(events.audio_done(turn.turn_id, index))
        finally:
            if not turn.cancel.is_set():
                await self.writer.send_json(events.all_audio_done(turn.turn_id))
            turn.audio_done = True
            await self._maybe_metrics(turn)

    async def _maybe_metrics(self, turn: Turn) -> None:
        if turn.metrics_sent or not (turn.llm_done and turn.audio_done):
            return
        turn.metrics_sent = True
        base = turn.t_ref
        values = {
            "endpoint_ms": round((turn.t_endpoint - base) * 1000),
            "asr_ms": round((turn.t_text - turn.t_endpoint) * 1000),
            "llm_first_ms": round((turn.t_llm - turn.t_text) * 1000) if turn.t_llm else 0,
            "tts_first_ms": round((turn.t_tts - turn.t_llm) * 1000) if turn.t_tts and turn.t_llm else 0,
            "total_ms": round((turn.t_tts - base) * 1000) if turn.t_tts else 0,
        }
        await self.writer.send_json(events.turn_metrics(turn.turn_id, **values))

    # ------------------------------------------------------------- playback
    async def playback_started(self, turn_id: str) -> None:
        turn = self.turn
        if turn is None or turn.turn_id != turn_id:
            return
        turn.playback_started = True
        turn.playback_started_at = time.monotonic()
        self.vad.on_playback_started()
        latency = (turn.playback_started_at - turn.t_tts) * 1000 if turn.t_tts else 0
        total = (turn.playback_started_at - turn.t_ref) * 1000
        await self.writer.send_json(
            events.playback_metrics(turn_id, playback_ms=round(latency), audible_total_ms=round(total))
        )

    async def playback_done(self, turn_id: str) -> None:
        turn = self.turn
        if turn is None or turn.turn_id != turn_id:
            return
        if turn.llm_done and turn.audio_done:
            self.vad.set_agent_speaking(False)
            self.turn = None

    # ------------------------------------------------------------ barge/merge
    async def _begin_merge(self, turn: Turn) -> None:
        self.merge_pending = True
        self.merge_text = turn.user_text
        turn.merged = True
        turn.cancel.set()
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()
        await self.writer.send_json(events.cancelled(turn.turn_id, "merge"))
        self.turn = None
        self.vad.set_agent_speaking(False)

    async def _cancel_turn(self, reason: str, emit: bool = True) -> None:
        turn = self.turn
        if turn is None:
            return
        turn.cancel.set()
        self._save_assistant_history(turn)
        self.turn = None
        self.vad.set_agent_speaking(False)
        if emit and not self._closed:
            await self.writer.send_json(events.interrupted(turn.turn_id, reason))
