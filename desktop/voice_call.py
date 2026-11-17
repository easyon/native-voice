"""Native Voice: local full-duplex voice call.

Usage (conda env `cosyvoice`):
    python desktop/voice_call.py                 # microphone + speakers
    python desktop/voice_call.py --text          # type instead of speaking (no audio I/O)
    python desktop/voice_call.py --list-devices  # show PortAudio device indices
    python desktop/voice_call.py --save-audio    # write utterances/replies to audio/calls/
"""
from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from nv.config import Config  # noqa: E402
from nv.conversation import VoiceCall  # noqa: E402


def build_config(args) -> Config:
    cfg = Config()
    if args.text:
        cfg.text_mode = True
    if args.asr:
        cfg.asr_mode = args.asr
    if args.asr_chunk_ms is not None:
        cfg.asr_stream_chunk_ms = args.asr_chunk_ms
    cfg.show_partials = not args.no_partials
    if args.wav:
        cfg.wav_input = Path(args.wav)
    if args.no_play:
        cfg.no_play = True
    if args.mic is not None:
        cfg.mic_device = args.mic
    if args.speaker is not None:
        cfg.speaker_device = args.speaker
    if args.barge_db is not None:
        cfg.barge_db = args.barge_db
    if args.endpoint_ms is not None:
        cfg.endpoint_silence_ms = args.endpoint_ms
    if args.tts_hop is not None:
        cfg.tts_hop = args.tts_hop
    if args.port is not None:
        cfg.llama_port = args.port
    if args.llm is not None:
        cfg.llm_model = Path(args.llm)
    if args.gain is not None:
        cfg.output_gain = args.gain
    cfg.save_audio = args.save_audio
    cfg.verbose = args.verbose
    cfg.use_cuda_graph = not args.no_graph
    return cfg


def main():
    if os.environ.get("NV_DUMP_STACKS"):
        faulthandler.dump_traceback_later(float(os.environ["NV_DUMP_STACKS"]), repeat=False)
    parser = argparse.ArgumentParser(description="Native Voice full-duplex voice call")
    parser.add_argument("--text", action="store_true", help="text input mode, no microphone/speaker")
    parser.add_argument(
        "--wav", type=str, default=None, help="replay a wav as the microphone (pipeline test)"
    )
    parser.add_argument(
        "--asr",
        choices=["hybrid", "streaming", "sensevoice"],
        default="streaming",
        help="streaming=仅流式(默认), hybrid=流式+SenseVoice 定稿, sensevoice=仅整句",
    )
    parser.add_argument(
        "--asr-chunk-ms",
        type=int,
        default=None,
        help="streaming ASR chunk cadence in ms (default 480; >=480 keeps accuracy)",
    )
    parser.add_argument("--no-partials", action="store_true", help="hide live partial transcripts")
    parser.add_argument("--no-play", action="store_true", help="run TTS but do not open the speaker")
    parser.add_argument("--mic", type=int, default=None, help="input device index")
    parser.add_argument("--speaker", type=int, default=None, help="output device index")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument(
        "--barge-db", type=float, default=None, help="barge-in threshold in dBFS (default -30)"
    )
    parser.add_argument(
        "--endpoint-ms", type=int, default=None, help="silence before end of turn (default 500)"
    )
    parser.add_argument(
        "--tts-hop", type=int, default=None, help="TTS speech tokens per first chunk (default 15)"
    )
    parser.add_argument("--port", type=int, default=None, help="llama-server port (default 8081)")
    parser.add_argument("--llm", type=str, default=None, help="path to a GGUF model for llama-server")
    parser.add_argument("--gain", type=float, default=None, help="speaker gain (default 1.0)")
    parser.add_argument("--save-audio", action="store_true", help="save utterances and replies")
    parser.add_argument("--no-graph", action="store_true", help="disable the CUDA-graphed TTS decoder")
    parser.add_argument("--verbose", action="store_true", help="verbose logs and VAD/ASR timings")
    args = parser.parse_args()

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.list_devices:
        from nv.audio import list_devices

        print(list_devices())
        return

    logging.getLogger().setLevel(logging.WARNING if not args.verbose else logging.INFO)
    cfg = build_config(args)
    VoiceCall(cfg).run()


if __name__ == "__main__":
    main()
