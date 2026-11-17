"""Full-duplex conversation loop: VAD -> ASR -> streaming LLM -> streaming TTS.

Turns are pipelined. While the assistant is thinking or speaking the microphone
stays live; sustained speech cancels the current response (barge-in) and the
utterance that caused the interruption becomes the next user turn.
"""
from __future__ import annotations

import difflib
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .vad import EndpointVad, VadEvent

_SENT_END = "。！？!?；;\n"
_SOFT = "，,、：:"
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF☀-➿️‍]+"
)


class SentenceSplitter:
    """Split streamed LLM deltas into speakable sentences."""

    def __init__(self, soft_limit: int = 18, hard_limit: int = 42):
        self.buf = ""
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit

    def _clean(self, s: str) -> str:
        s = _EMOJI.sub("", s)
        s = s.replace("*", "").replace("#", "").replace("`", "")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        while True:
            cut = -1
            for i, ch in enumerate(self.buf):
                if ch in _SENT_END:
                    cut = i + 1
                    # keep closing quotes/brackets with the sentence
                    while cut < len(self.buf) and self.buf[cut] in "”’」』）)\"'":
                        cut += 1
                    break
            if cut < 0 and len(self.buf) >= self.soft_limit:
                soft = [i for i, ch in enumerate(self.buf) if ch in _SOFT]
                if soft:
                    cut = soft[-1] + 1
            if cut < 0 and len(self.buf) >= self.hard_limit:
                cut = len(self.buf)
            if cut < 0:
                break
            piece = self._clean(self.buf[:cut])
            self.buf = self.buf[cut:]
            if piece and re.search(r"[\w一-鿿]", piece):
                out.append(piece)
        return out

    def flush(self) -> list[str]:
        piece = self._clean(self.buf)
        self.buf = ""
        return [piece] if piece and re.search(r"[\w一-鿿]", piece) else []


@dataclass
class Turn:
    cancel: threading.Event = field(default_factory=threading.Event)
    sentences: queue.Queue = field(default_factory=queue.Queue)
    done: threading.Event = field(default_factory=threading.Event)
    text: str = ""
    cancelled: bool = False
    llm_done: threading.Event = field(default_factory=threading.Event)
    audio: list[np.ndarray] = field(default_factory=list)
    audio_started: bool = False
    # latency measurement
    t_ref: float = 0.0          # estimated physical end of user speech
    t_endpoint: float = 0.0     # VAD end decision
    t_text: float = 0.0         # final transcript ready
    t_llm: float = 0.0          # first LLM token
    t_tts: float = 0.0          # first TTS chunk generated
    t_audible: float = 0.0      # first non-silent output (incl. device latency)
    latency_logged: bool = False
    first_write: bool = False
    user_text: str = ""


class VoiceCall:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vad = EndpointVad(cfg)
        self.mic = None
        self.speaker = None
        self.asr = None
        self.stream_asr = None
        self.llm = None
        self.tts = None
        self._partial_shown = False
        self.turn: Turn | None = None
        self.history: list[dict] = []
        self._kb: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._status = ""
        self._last_reply = ""
        self._last_reply_at = 0.0
        self._utterance_seq = 0
        self._merge_pending = False
        self._merge_text = ""

    # --------------------------------------------------------------- setup
    def load(self):
        import logging

        from .asr import ASR
        from .llm import LlamaClient
        from .stream_asr import StreamingASR
        from .tts import CosyVoiceTTS

        if not self.cfg.verbose:
            logging.disable(logging.INFO)

        self.llm = LlamaClient(self.cfg)
        self.llm.start()
        if self.cfg.warm_llm:
            self.llm.warmup(self.cfg.system_prompt)

        self.tts = CosyVoiceTTS(self.cfg)

        if not self.cfg.text_mode:
            mode = self.cfg.asr_mode
            if mode not in ("hybrid", "streaming", "sensevoice"):
                raise ValueError(f"unknown asr mode: {mode}")
            if mode in ("hybrid", "streaming"):
                self.stream_asr = StreamingASR(self.cfg)
                self.stream_asr.on_partial = self._on_partial
                self.stream_asr.load()
            if mode in ("hybrid", "sensevoice"):
                self.asr = ASR(self.cfg)
                self.asr.load()
            print(f"[ASR] 模式: {mode}", flush=True)
            try:
                import torch

                if torch.cuda.is_available():
                    print(
                        f"[显存] 分配 {torch.cuda.memory_allocated() / 1e9:.2f}GB "
                        f"峰值 {torch.cuda.max_memory_allocated() / 1e9:.2f}GB",
                        flush=True,
                    )
            except Exception:
                pass
        else:
            print("[ASR] 文字模式，跳过加载", flush=True)

    def _open_audio(self):
        from .audio import Mic, Speaker, WavMic

        if self.cfg.wav_input is not None:
            self.mic = WavMic(self.cfg.wav_input, self.cfg.input_rate, self.cfg.frame_ms)
            print(f"[音频] 使用 wav 模拟麦克风: {self.cfg.wav_input}", flush=True)
        else:
            self.mic = Mic(self.cfg.mic_device, self.cfg.input_rate, self.cfg.frame_ms)
            print("[音频] 麦克风已打开", flush=True)
        self.mic.start()
        if not self.cfg.no_play:
            self.speaker = Speaker(self.cfg.speaker_device, self.cfg.output_rate, self.cfg.output_gain)
            self.speaker.start()
            print("[音频] 扬声器已打开", flush=True)
        in_ms = float(getattr(self.mic, "latency", 0.0) or 0.0) * 1000.0
        out_ms = float(getattr(self.speaker, "latency", 0.0) or 0.0) * 1000.0 if self.speaker else 0.0
        print(f"[音频] 设备延迟: 输入 ~{in_ms:.0f}ms, 输出 ~{out_ms:.0f}ms", flush=True)

    def _start_keyboard(self):
        def reader():
            try:
                for line in sys.stdin:
                    self._kb.put(line.rstrip("\r\n"))
            except Exception:
                pass

        threading.Thread(target=reader, daemon=True, name="stdin").start()

    # ---------------------------------------------------------------- main
    def run(self):
        import logging

        if not self.cfg.verbose:
            logging.disable(logging.INFO)
        for name in ("urllib3", "requests", "modelscope", "funasr"):
            logging.getLogger(name).setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.WARNING)
        self.load()
        if not self.cfg.verbose:
            logging.disable(logging.INFO)
        if not self.cfg.text_mode:
            self._open_audio()
        self._start_keyboard()

        mode = "文字输入" if self.cfg.text_mode else "语音通话"
        asr_desc = {
            "hybrid": "流式 Paraformer + SenseVoice 校正",
            "streaming": "流式 Paraformer",
            "sensevoice": "SenseVoice 整句",
        }.get(self.cfg.asr_mode, self.cfg.asr_mode)
        print()
        print(f"===== Native Voice · {mode} =====")
        print(f"LLM {self.cfg.llm_model.name} | TTS CosyVoice3 | ASR {asr_desc}")
        if not self.cfg.text_mode:
            print("直接说话即可；AI 说话时插话可以打断。建议戴耳机以避免回声误触发。")
        print("键盘：输入文字 = 发送一条消息；直接回车 = 打断；Ctrl+C 退出。")
        print("=" * 34, flush=True)
        self._set_status("聆听中" if not self.cfg.text_mode else "等待输入")

        last_hb = 0.0
        try:
            while not self._stop.is_set():
                if self.cfg.verbose and time.monotonic() - last_hb > 2.0:
                    last_hb = time.monotonic()
                    print(
                        f"[VAD] level={self.vad.level_db:.0f} noise={self.vad.noise_floor_db:.0f} "
                        f"active={self.vad.active}",
                        flush=True,
                    )
                if not self.cfg.text_mode:
                    frame = self.mic.read(timeout=0.02)
                    if frame is not None:
                        started = False
                        for ev in self.vad.feed(frame):
                            if ev.kind == "start":
                                started = True
                                self._stream_reset(ev)
                            self._on_vad(ev)
                        # The triggering frame is already part of ev.audio.
                        if not started and self.vad.active and self.stream_asr is not None:
                            self.stream_asr.feed(frame)
                    self._poll_keyboard(0)
                else:
                    self._poll_keyboard(0.02)
                self._tick()
        except KeyboardInterrupt:
            print("\n[退出] 收到 Ctrl+C")
        finally:
            self._shutdown()

    def _poll_keyboard(self, timeout: float):
        try:
            line = self._kb.get(timeout=timeout) if timeout else self._kb.get_nowait()
        except queue.Empty:
            return
        line = line.strip()
        if line in ("/quit", "/exit", "退出", "quit", "exit"):
            self._stop.set()
            return
        if not line:
            if self.turn is not None:
                self._cancel_turn("键盘打断")
            return
        if self.turn is not None:
            self._cancel_turn("键盘发送新消息")
        self._start_turn(line, source="键盘")

    def _on_vad(self, ev: VadEvent):
        if ev.kind == "start":
            if self.turn is not None:
                if self._can_merge(self.turn):
                    self._begin_merge(self.turn)
                else:
                    self._cancel_turn("用户插话")
            else:
                self._set_status("聆听中")
            return
        if ev.kind == "end" and ev.audio is not None:
            if self.turn is not None:
                self._cancel_turn("用户插话")
            self._handle_utterance(ev.audio, ev)

    def _stream_reset(self, ev: VadEvent):
        if self.stream_asr is None:
            return
        self._partial_shown = False
        try:
            self.stream_asr.reset(ev.audio)
        except Exception as exc:
            print(f"\n[ASR] 流式识别重置失败: {exc!r}", flush=True)

    def _on_partial(self, text: str):
        if not self.cfg.show_partials or not text:
            return
        print(f"\r[识别] {text}", end="", flush=True)
        self._partial_shown = True

    def _handle_utterance(self, audio: np.ndarray, ev: VadEvent):
        t_endpoint = time.monotonic()
        input_delay = float(getattr(self.mic, "latency", 0.0) or 0.0) if self.mic is not None else 0.0
        # The mic buffer delivers the last speech frame ~input_delay after it was
        # spoken; subtract it so t_ref approximates the physical end of speech.
        t_ref = ev.last_voiced_time - input_delay if ev.last_voiced_time else (
            t_endpoint - self.cfg.endpoint_silence_ms / 1000.0
        )
        t0 = time.time()
        stream_text = ""
        if self.stream_asr is not None:
            try:
                stream_text = self.stream_asr.finalize()
            except Exception as exc:
                print(f"\n[ASR] 流式收尾失败: {exc!r}", flush=True)
            if self._partial_shown:
                print(flush=True)
                self._partial_shown = False
        self._set_status("识别中")
        if self.cfg.save_audio:
            self._save_wav(audio, f"utt_{self._utterance_seq:02d}.wav")
            self._utterance_seq += 1
        text = ""
        if self.cfg.asr_mode == "streaming":
            text = stream_text
        else:
            try:
                text = self.asr.transcribe(audio)
            except Exception as exc:
                print(f"\n[ASR] 失败: {exc!r}", flush=True)
            if not text:
                text = stream_text
        if self.cfg.verbose:
            extra = f" stream={stream_text!r}" if self.stream_asr is not None else ""
            print(f"[ASR] {time.time() - t0:.2f}s -> {text!r}{extra}", flush=True)
        if self._merge_pending:
            prev, self._merge_text = self._merge_text, ""
            self._merge_pending = False
            if text:
                sep = "" if prev and prev[-1] in (_SENT_END + _SOFT) else "，"
                text = (prev + sep + text).strip()
            else:
                text = prev
            if text:
                print(f"[接续] 合并同一轮语音: {text}", flush=True)
        if not text:
            self._set_status("聆听中")
            return
        if self._looks_like_echo(text):
            if self.cfg.verbose:
                print(f"[ASR] 忽略回声: {text!r}", flush=True)
            self._set_status("聆听中")
            return
        print(f"\n你: {text}", flush=True)
        self._start_turn(text, source="语音", t_ref=t_ref, t_endpoint=t_endpoint)

    def _looks_like_echo(self, text: str) -> bool:
        if not self._last_reply or time.monotonic() - self._last_reply_at > 12:
            return False
        a = re.sub(r"\s+", "", text)
        b = re.sub(r"\s+", "", self._last_reply)
        if not a or not b:
            return False
        return difflib.SequenceMatcher(None, a, b).ratio() > 0.72

    # --------------------------------------------------------------- turns
    def _messages(self) -> list[dict]:
        return [{"role": "system", "content": self.cfg.system_prompt}] + self.history

    def _start_turn(self, user_text: str, source: str, t_ref: float = 0.0, t_endpoint: float = 0.0):
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()
        turn = Turn()
        turn.user_text = user_text
        turn.t_ref = t_ref or time.monotonic()
        turn.t_endpoint = t_endpoint or time.monotonic()
        turn.t_text = time.monotonic()
        self.turn = turn
        if self.mic is not None:
            self.vad.set_agent_speaking(True)
        self._set_status("思考中")
        print("AI: ", end="", flush=True)
        threading.Thread(target=self._llm_worker, args=(turn,), daemon=True, name="llm").start()
        threading.Thread(target=self._tts_worker, args=(turn,), daemon=True, name="tts").start()

    def _llm_worker(self, turn: Turn):
        splitter = SentenceSplitter()
        messages = self._messages()
        t0 = time.time()
        first = None
        try:
            for delta in self.llm.chat_stream(messages, turn.cancel):
                if turn.cancel.is_set():
                    break
                if first is None:
                    first = time.time() - t0
                    turn.t_llm = time.monotonic()
                    if self.cfg.verbose:
                        print(f"\n[LLM] 首 token {first:.2f}s", flush=True)
                turn.text += delta
                print(delta, end="", flush=True)
                for sent in splitter.feed(delta):
                    turn.sentences.put(sent)
        except Exception as exc:
            if not (self._stop.is_set() or turn.cancelled):
                print(f"\n[LLM] 失败: {exc!r}", flush=True)
        finally:
            if not turn.cancel.is_set():
                for sent in splitter.flush():
                    turn.sentences.put(sent)
                if turn.text.strip():
                    self.history.append({"role": "assistant", "content": turn.text.strip()})
                    self._trim_history()
            turn.sentences.put(None)
            turn.llm_done.set()
            if not turn.cancelled:
                print(flush=True)

    def _tts_worker(self, turn: Turn):
        try:
            while True:
                sent = turn.sentences.get()
                if sent is None:
                    break
                if turn.cancel.is_set():
                    continue
                self._set_status("说话中")
                t_sent = time.time()
                first_chunk = None
                try:
                    for chunk in self.tts.synthesize_stream(sent, turn.cancel):
                        if turn.cancel.is_set():
                            break
                        if first_chunk is None:
                            first_chunk = time.time() - t_sent
                            if turn.t_tts == 0:
                                turn.t_tts = time.monotonic()
                            if self.cfg.verbose:
                                print(f"\n[TTS] 首块 {first_chunk:.2f}s", flush=True)
                        if self.cfg.save_audio:
                            turn.audio.append(chunk.copy())
                        if self.speaker is not None:
                            if not turn.audio_started:
                                turn.audio_started = True
                                if self.mic is not None:
                                    self.vad.on_playback_started()
                            if not turn.first_write:
                                turn.first_write = True
                                self.speaker.arm_audible(
                                    lambda ts, t=turn: setattr(t, "t_audible", ts)
                                )
                            self.speaker.write(chunk)
                        elif not turn.audio_started:
                            # No physical playback: use the first generated chunk as proxy.
                            turn.audio_started = True
                            turn.t_audible = turn.t_tts
                except Exception as exc:
                    if not (self._stop.is_set() or turn.cancelled):
                        print(f"\n[TTS] 失败: {exc!r}", flush=True)
        finally:
            turn.done.set()
            if self.cfg.save_audio and turn.audio:
                self._save_wav(np.concatenate(turn.audio), f"reply_{int(time.time() * 1000) % 100000}.wav")

    def _tick(self):
        # Drain any finished turn once its audio has actually played out.
        turn = self.turn
        if turn is None:
            return
        if turn.done.is_set() and (self.speaker is None or self.speaker.is_idle()):
            self.turn = None
            if not turn.cancelled:
                self._last_reply = re.sub(r"\s+", "", turn.text)
                self._last_reply_at = time.monotonic()
                self._set_status("聆听中")
                self._report_latency(turn)
            if self.mic is not None:
                self.vad.set_agent_speaking(False)

    def _report_latency(self, turn: Turn):
        if turn.latency_logged or not turn.t_audible or not turn.t_llm:
            return
        turn.latency_logged = True
        total = turn.t_audible - turn.t_ref
        marks = [
            ("端点", turn.t_ref, turn.t_endpoint),
            ("ASR", turn.t_endpoint, turn.t_text),
            ("LLM", turn.t_text, turn.t_llm),
            ("TTS", turn.t_llm, turn.t_tts),
            ("播放", turn.t_tts, turn.t_audible),
        ]
        detail = " ".join(f"{k} {b - a:.2f}" for k, a, b in marks if a and b and b >= a)
        kind = "出声" if self.speaker is not None else "首块生成"
        print(f"[时延] 说完→{kind} {total:.2f}s | {detail}", flush=True)

    def _can_merge(self, turn: Turn) -> bool:
        """A quick follow-up before the AI has been heard belongs to the same turn."""
        if self.cfg.merge_window_s <= 0 or not turn.user_text:
            return False
        if turn.t_audible != 0:
            return False
        return (time.monotonic() - turn.t_endpoint) <= self.cfg.merge_window_s

    def _begin_merge(self, turn: Turn):
        self._merge_pending = True
        self._merge_text = turn.user_text
        turn.cancelled = True
        turn.cancel.set()
        if self.speaker is not None:
            self.speaker.stop()
        self.turn = None
        # Drop the not-yet-answered user message (and a very fast assistant echo).
        while self.history and self.history[-1]["role"] == "assistant":
            self.history.pop()
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()
        print("\n[接续] 上一段还没回答，合并后继续听...", flush=True)
        self._set_status("聆听中")

    def _cancel_turn(self, reason: str):
        self._merge_pending = False
        self._merge_text = ""
        turn = self.turn
        if turn is None:
            return
        turn.cancelled = True
        turn.cancel.set()
        if self.speaker is not None:
            self.speaker.stop()
        if self.mic is not None:
            self.vad.set_agent_speaking(False)
        self.turn = None
        if turn.text.strip():
            self._last_reply = re.sub(r"\s+", "", turn.text)
            self._last_reply_at = time.monotonic()
        print(f"\n[打断] {reason}", flush=True)
        self._set_status("聆听中")

    def _trim_history(self):
        limit = max(2, self.cfg.history_turns * 2)
        while len(self.history) > limit:
            self.history.pop(0)
        while self.history and self.history[0]["role"] != "user":
            self.history.pop(0)

    # --------------------------------------------------------------- misc
    def _set_status(self, text: str):
        if text != self._status:
            self._status = text
            print(f"[状态] {text}", flush=True)

    def _save_wav(self, audio: np.ndarray, name: str):
        try:
            import soundfile as sf

            self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
            rate = self.cfg.output_rate if name.startswith("reply") else self.cfg.input_rate
            sf.write(
                str(Path(self.cfg.log_dir) / name),
                np.asarray(audio, dtype=np.float32),
                rate,
            )
        except Exception as exc:
            print(f"[保存音频] 失败: {exc!r}", flush=True)

    def _shutdown(self):
        if self.turn is not None:
            self.turn.cancelled = True
            self.turn.cancel.set()
            if self.speaker is not None:
                self.speaker.stop()
            self.turn = None
        if self.mic is not None:
            self.mic.close()
        if self.speaker is not None:
            self.speaker.close()
        if self.llm is not None:
            self.llm.close()
        if self.tts is not None:
            self.tts.close()
