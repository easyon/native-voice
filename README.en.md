# Native Voice

**English** | [简体中文](README.md)

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](pyproject.toml)

**A self-hostable, full-duplex voice conversation service.** Always-on microphone,
VAD-based turn detection, streaming ASR, streaming LLM, and streaming TTS with
natural barge-in. Ships with Bailian-compatible ASR/TTS WebSocket APIs, a
conversation WebSocket, API keys / quotas / Prometheus metrics, and a
build-free browser client.

Everything can run offline on a single 8 GB consumer GPU:
Paraformer + Qwen3.5 + CosyVoice3.

> Detailed documentation is currently maintained in Chinese under [`docs/`](docs/README.md).
> English translations are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Highlights

- **Full-duplex and natural interruption** — continuous capture, VAD plus a
  semantic gate, backchannels, and 1.6 s fragment merging.
- **Multiple ASR / TTS providers** — Paraformer, Qwen3-ASR, CosyVoice3,
  Qwen3-TTS, and true streaming TTS through vLLM-Omni.
- **Tool calling** — the conversation loop supports function calling (current
  time, web search, web page reading) with spoken fillers, tool-result round
  trips, and a per-turn round limit.
- **Bailian-compatible protocol** — `/api-ws/v1/inference` works with existing clients.
- **Production gateway** — API keys, scopes, concurrency / RPM / daily quotas,
  60-second tickets, structured logs, low-cardinality metrics, health checks.
- **Process isolation** — ASR and TTS run as separate GPU engines; the gateway
  itself is CPU-only.
- **Build-free web client** — live transcripts, audio playback, latency
  metrics, and voice cloning.

## Architecture

```text
Browser / Bailian client / native client
              │ WSS
              ▼
      Gateway (FastAPI, CPU)
        ├── Bailian ASR/TTS WS   /api-ws/v1/inference
        ├── Conversation WS      /ws/conversation
        ├── REST                 /healthz /readyz /metrics /v1/*
        ├── auth / quotas / metrics
        └── conversation engine (VAD, splitting, barge-in, playback ACK)
              │ internal WebSocket
              ├──────────────► ASR engine (GPU)
              └──────────────► TTS engine (GPU)
              └──HTTP────────► llama-server (CPU/GPU)  Qwen3.5
```

`microphone 16 kHz → VAD → ASR → LLM → TTS → 24 kHz playback`.

## Quick start

Requirements: Python 3.10, [uv](https://docs.astral.sh/uv/), and optionally an
NVIDIA GPU (8 GB recommended).

```bash
git clone https://github.com/easyon/native-voice.git
cd native-voice

uv sync --extra gpu
uv run --extra gpu python scripts/fetch_deps.py
```

Start the development gateway (put the LLM on CPU on an 8 GB GPU):

```powershell
# Windows
$env:LLM_DEVICE = "cpu"
scripts\serve_dev.ps1 -Backend local
```

```bash
# Linux / macOS
LLM_DEVICE=cpu scripts/serve_dev.sh
```

Open <http://127.0.0.1:8090/ui/>, paste the development key printed in the
logs, and click **Call**.

No GPU? Use the fake backend to develop the web UI and protocols:

```powershell
scripts\serve_dev.ps1 -Backend fake
```

A standalone desktop CLI that talks to the sound card directly is available in
[`desktop/`](desktop/README.md):

```powershell
conda activate cosyvoice
python desktop\voice_call.py
```

## Repository layout

```text
native-voice/
├── src/native_voice/     # gateway, engines, model adapters, conversation engine
├── desktop/              # standalone full-duplex CLI (optional)
├── web/                  # build-free browser client
├── tests/                # unit / protocol / manual GPU checks
├── benchmarks/           # historical tuning scripts
├── configs/              # service and voice configuration
├── deploy/vllm-omni/     # true streaming TTS container
├── docs/                 # full documentation (Chinese)
├── examples/             # protocol client examples
├── scripts/              # install / serve / load-test helpers
├── assets/               # Silero VAD
├── models/               # model weights (gitignored, fetched by script)
├── bin/                  # local binaries (gitignored)
└── third_party/          # third-party sources (gitignored)
```

## Documentation

Start at [`docs/README.md`](docs/README.md) (Chinese):

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Models and performance](docs/models.md)
- [API reference](docs/api/README.md)
- [Deployment](docs/deployment.md)
- [Development](docs/development.md)
- [Troubleshooting](docs/troubleshooting.md)

## Contributing

Issues and pull requests are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
first. Security issues should follow [SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE). Model weights and third-party
components keep their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and verify them before
commercial use.
