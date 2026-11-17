"""Check the streaming Paraformer: partial text per 600 ms chunk + final text."""
import sys
import time
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "desktop"
if str(DESKTOP) not in sys.path:
    sys.path.insert(0, str(DESKTOP))

from nv.config import Config  # noqa: E402
from nv.stream_asr import StreamingASR  # noqa: E402


def main():
    cfg = Config()
    asr = StreamingASR(cfg)
    asr.load()
    asr.on_partial = lambda text: print(f"  partial: {text}")
    print(f"chunk={asr.chunk_ms}ms size={asr.chunk_size} stride={asr.chunk} samples")

    for name in ("audio/asr_test.wav", "audio/calls/reply_15157.wav"):
        path = ROOT / name
        if not path.exists():
            continue
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]
        if sr != cfg.input_rate:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(int(sr), cfg.input_rate)
            wav = resample_poly(wav, cfg.input_rate // g, int(sr) // g)
        print(f"\n{name}  {len(wav) / cfg.input_rate:.2f}s")
        asr.reset()
        t0 = time.time()
        step = int(0.02 * cfg.input_rate)
        for i in range(0, len(wav), step):
            asr.feed(wav[i:i + step])
        t_fin = time.time()
        final = asr.finalize()
        fin = time.time() - t_fin
        dt = time.time() - t0
        print(
            f"  final: {final!r}  (flush {fin * 1000:.0f} ms, "
            f"total {dt:.2f}s, RTF {dt / (len(wav) / cfg.input_rate):.2f})"
        )


if __name__ == "__main__":
    main()
