# Qwen3-TTS / CosyVoice3 streaming with vLLM-Omni

Runs `Qwen3-TTS-12Hz-0.6B-Base` (or `Fun-CosyVoice3-0.5B`) as a real streaming
TTS service in a Linux Docker container on WSL2. The gateway's
`qwen3tts_vllm` / `cosyvoice3_vllm` providers are thin WebSocket proxies; they
have no torch dependency and do not load the model in the engine process.

## Why a separate service

The `qwen-tts` Python package returns a complete waveform per utterance.
vLLM-Omni serves `/v1/audio/speech/stream`, which emits PCM chunks while the
talker generates codec frames. That is the only path that gives true
frame-level streaming for Qwen3-TTS.

## Prerequisites

- NVIDIA driver **>= 580** (CUDA 13). Verify:
  ```powershell
  nvidia-smi
  ```
- Docker Desktop with the WSL2 backend and GPU passthrough. Verify:
  ```powershell
  docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
  ```
- Local model at `models/Qwen3-TTS-12Hz-0.6B-Base` (already downloaded by
  `scripts/setup_qwen_engines.ps1`).
- The repo's `.data/voices` and `third_party` directories are mounted into the
  container at `/workspace/...`; reference audio must stay inside the repo.

## Start

From `deploy/vllm-omni`:

```powershell
docker compose build   # first run only; derives from vllm/vllm-omni:nightly
docker compose up -d
docker compose logs -f tts
```

The first start pulls the `vllm/vllm-omni:nightly` image (several GB), loads
both stages, and warms their caches; on the 8 GB laptop this takes about 90 s.
Wait until the health check passes:

```powershell
curl http://127.0.0.1:8091/health
```

Then start the gateway with the vLLM provider:

```powershell
scripts\serve_dev.ps1 -Backend worker -Tts qwen3tts_vllm
```

The Docker container is **not** started by the engine supervisor. Keep it
running independently (`restart: unless-stopped`).

## CosyVoice3 instead of Qwen3-TTS

The same image serves `Fun-CosyVoice3-0.5B` with vLLM-Omni's packaged
`cosyvoice3` pipeline. `cosyvoice3_8gb.yaml` is the 8 GB variant (lower stage
reservations, `max_num_seqs: 2`), and compose exposes it as a profile:

```powershell
cd deploy/vllm-omni
docker compose --profile cosyvoice3 up -d tts-cosyvoice3
curl http://127.0.0.1:8092/health
```

Then start the gateway with:

```powershell
scripts\serve_dev.ps1 -Backend worker -Tts cosyvoice3_vllm
```

CosyVoice3 clones from `ref_audio` + `ref_text` only: it rejects Qwen3-TTS'
`task_type` / `x_vector_only_mode` / `language` session fields, which the
`cosyvoice3_vllm` provider omits. Output is 24 kHz PCM, the same as Qwen3-TTS.

Run **one** of the two containers, not both: on 8 GB a single two-stage model
plus the streaming ASR already fills the card. The CosyVoice3 numbers here are
not yet measured on the RTX 4070 Laptop; tune `cosyvoice3_8gb.yaml` with
`nvidia-smi` on first bring-up.

## Image workarounds

The upstream `vllm/vllm-omni:nightly` image pulled on 2026-11-10 had two
truncated files because the first pull ran out of disk space:

- `z3/lib/libz3.so` was 4 MiB instead of ~22.6 MiB. Importing `z3` caused
  `SIGBUS` or `undefined symbol` errors inside `torch.compile`.
- `triton/backends/nvidia/bin/ptxas` segfaulted, so Triton reported
  `RuntimeError: Cannot find ptxas`.

`Dockerfile` derives from the nightly image, reinstalls the exact
`z3-solver` version already present, and points Triton at the CUDA toolkit's
working `ptxas`. Always build with `docker compose build`; using the upstream
image directly will fail on this machine.

If you force a clean re-pull of the upstream image on a healthy disk, these
workarounds become unnecessary, but they are harmless.

## Streaming test

```powershell
$body = @{
  input = "你好，这是流式语音合成的测试。"
  task_type = "Base"
  language = "Chinese"
  stream = $true
  stream_format = "audio"
  response_format = "pcm"
  ref_audio = "file:///workspace/third_party/CosyVoice-main/asset/zero_shot_prompt.wav"
  ref_text = "希望你以后能够做的比我还好呦。"
} | ConvertTo-Json
curl.exe -N -X POST http://127.0.0.1:8091/v1/audio/speech `
  -H "Content-Type: application/json" -d $body --output stream.pcm
```

Or use the WebSocket protocol our provider uses:

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8091/v1/audio/speech/stream") as ws:
        await ws.send(json.dumps({
            "type": "session.config",
            "task_type": "Base",
            "language": "Chinese",
            "response_format": "pcm",
            "stream_audio": True,
            "ref_audio": "/workspace/third_party/CosyVoice-main/asset/zero_shot_prompt.wav",
            "ref_text": "希望你以后能够做的比我还好呦。",
        }))
        await ws.send(json.dumps({"type": "input.text", "text": "你好，这是流式合成。"}))
        await ws.send(json.dumps({"type": "input.done"}))
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                print("pcm", len(msg))
            elif json.loads(msg).get("type") == "session.done":
                break

asyncio.run(main())
```

## Measured latency (RTX 4070 Laptop 8 GB)

After the gateway's TTS engine has loaded, which itself sends one warmup
request:

- First PCM chunk: **0.420 s** after request start (the first chunk is 80 ms
  of audio; later chunks arrive roughly every 0.85–1.0 s).
- A 15-character sentence: **2.18 s** to synthesize **3.76 s** of audio,
  **RTF 0.58**.
- Cold first request on a freshly started container: **18.35 s** first audio.
  The gateway provider's `warmup: true` performs this request during `load()`
  (measured `cold load+warmup=18.80 s`), so `/readyz` turns green only after
  the expensive initialization is done; the first user turn then sees ~0.55 s
  first audio.
- End-to-end with streaming Paraformer + Qwen3-1.7B (CPU): `tts_first_ms=406`,
  `total_ms=1078`; a cloned voice from `.data/voices` measured `tts_first_ms=594`,
  `total_ms=766`.

## 8 GB tuning

`qwen3_tts_8gb.yaml` reserves `0.38` of total GPU memory for the talker and
`0.17` for code2wav, both with `enforce_eager: true` for the first bring-up.
Measured on the RTX 4070 Laptop (8188 MiB):

- Talker weights: **2.14 GiB**; KV cache for a 4096-token sequence: **0.44 GiB**.
  At `gpu_memory_utilization: 0.30` only 0.04 GiB of KV cache was left, so
  startup failed with "larger than the available KV cache memory".
- Eager mode skips the AOT compile and its `z3`/`ptxas` dependencies, at the
  cost of some decode speed. After the service is stable, try
  `enforce_eager: false` to enable CUDA graphs.

| Component | Approx. VRAM |
|---|---:|
| Qwen3-TTS talker + code2wav (reserved) | 4.5 GB |
| Streaming ASR | 1.6 GB |
| Windows desktop / CUDA context | 0.7–0.9 GB |

The two stages plus ASR are close to the 8 GB limit. Run the vLLM container
alone when measuring TTS; when the full pipeline is active, keep the ASR
provider lean and watch `nvidia-smi`.

If startup fails with `No available memory for the cache blocks`:

1. Raise stage 0 (`0.40`) if its KV cache is too small; lower stage 1
   (`0.14`) if code2wav has spare reservation.
2. Keep `enforce_eager: true` on both stages.
3. Lower `max_num_seqs` to `2`.

Do not run the transformers `qwen3tts` provider at the same time as the vLLM
container: both would load a model on the same 8 GB GPU.

## Known limits

- `task_type: Base` (voice cloning) needs `ref_audio` and `ref_text`; a
  mismatched transcript degrades quality. The reference path is server-side,
  hence the `/workspace` mounts.
- vLLM-Omni serves one checkpoint per container. Switching between Base and
  CustomVoice requires restarting the container.
- If the talker hits its codec budget without EOS, vLLM reports an error after
  possibly emitting partial audio. The engine surfaces it as a TTS job error;
  already-sent audio cannot be retracted.
- The gateway still receives float32 chunks from the engine and converts them
  to PCM16 for clients; vLLM's PCM16 is converted to float32 inside the
  provider.
