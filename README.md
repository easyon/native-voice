# Native Voice

**简体中文** | [English](README.en.md)

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-pytest-informational.svg)](tests/)
[![Docs](https://img.shields.io/badge/docs-中文-brightgreen.svg)](docs/README.md)

**可私有化部署的全双工语音对话服务。** 麦克风常开、VAD 自动断句、流式 ASR、
流式 LLM、流式 TTS，说话过程中随时可以插话打断。提供百炼兼容的 ASR/TTS
WebSocket 协议、对话 WebSocket、API Key / 配额 / Prometheus 指标，以及一个
无需构建的网页客户端。

> 全部推理可离线运行：Paraformer + Qwen3.5 + CosyVoice3，单张 8 GB 消费级显卡即可跑通。

## 特性

- **全双工与自然插话**：持续采集音频，VAD + 语义门控决定何时打断；支持附和
  （backchannel）和 1.6 s 内的碎片合并。
- **多 ASR / TTS 提供方**：Paraformer、Qwen3-ASR、CosyVoice3、Qwen3-TTS，
  以及通过 vLLM-Omni 实现的真流式 TTS。
- **工具调用**：对话链路支持 function calling（内置当前时间、联网搜索、
  网页读取），工具结果回填后继续生成口语回答，可用 `llm.tools` 配置启用或关闭。
- **百炼兼容协议**：`/api-ws/v1/inference` 可直接接入现有百炼客户端。
- **生产级网关**：API Key、scope、并发 / RPM / 每日额度、60 s 短票、
  结构化日志、低基数指标、健康检查。
- **进程隔离**：ASR / TTS 作为独立 GPU engine 运行，网关崩溃不会殃及模型进程。
- **开箱即用的网页**：实时字幕、AI 音频、时延指标、音色克隆，无前端构建步骤。
- **两种运行方式**：服务模式（网关 + 网页）与单机 `desktop/` CLI（直接使用声卡）。

## 架构

```text
浏览器 / 百炼客户端 / 原生客户端
              │  WSS
              ▼
      Gateway (FastAPI, CPU)
        ├── 百炼兼容 ASR/TTS WS      /api-ws/v1/inference
        ├── 全双工对话 WS            /ws/conversation
        ├── REST                     /healthz /readyz /metrics /v1/*
        ├── 认证 / 配额 / 指标
        └── 对话引擎（VAD、分句、打断、播放 ACK）
              │  内部 WebSocket
              ├──────────────► ASR engine (GPU)   流式 Paraformer / Qwen3-ASR
              └──────────────► TTS engine (GPU)   CosyVoice3 / Qwen3-TTS
              │
              └──HTTP────────► llama-server (CPU/GPU)   Qwen3.5
```

请求链路：`麦克风 16 kHz → VAD → ASR → LLM → TTS → 24 kHz 播放`。
网关只负责协议与调度，模型加载和推理都在独立 engine 进程中完成。

更详细的组件职责、数据流和插话状态机见 [docs/architecture.md](docs/architecture.md)。

## 快速开始

### 依赖

- Python 3.10
- [uv](https://docs.astral.sh/uv/)（推荐）或 conda
- NVIDIA GPU（推荐 ≥ 8 GB 显存）；仅调试页面 / 协议时可使用 `fake` 后端
- Windows 11 / Linux 均可运行；vLLM-Omni 真流式 TTS 需要 Linux Docker 或 WSL2

### 1. 安装

```bash
git clone https://github.com/easyon/native-voice.git
cd native-voice
uv sync --extra gpu
```

`--extra gpu` 会安装 GPU 版 torch（cu121）。只做协议/页面开发时用
`uv sync --extra dev` 即可。

### 2. 下载本地模型与二进制

```bash
uv run --extra gpu python scripts/fetch_deps.py
uv run --extra gpu python scripts/fetch_deps.py --check   # 只检查缺件
```

下载内容包括 CosyVoice3 权重、Qwen3.5 GGUF、llama.cpp、CosyVoice 源码等，
约 7 GB。模型默认优先从 ModelScope 获取，失败时回退 HuggingFace。

### 3. 启动服务

Windows（8 GB 显卡建议把 LLM 放在 CPU）：

```powershell
$env:LLM_DEVICE = "cpu"
scripts\serve_dev.ps1 -Backend local
```

Linux / macOS：

```bash
LLM_DEVICE=cpu scripts/serve_dev.sh
```

如果已经安装 CUDA 版 llama.cpp（`bin/llama-cuda/`），可以去掉 `LLM_DEVICE=cpu`
让 LLM 使用 GPU；`configs/service.yaml` 默认指向 CUDA 版。

打开 <http://127.0.0.1:8090/ui/>，输入启动日志里打印的开发 Key，点击“呼叫”。

没有 GPU、只想联调页面和协议时：

```powershell
scripts\serve_dev.ps1 -Backend fake
```

### 4. 单机 CLI（可选）

`desktop/` 是一个不经过网关的轻量全双工 CLI，直接访问声卡，适合快速验证：

```powershell
conda activate cosyvoice
python desktop\voice_call.py
```

完整参数、设备选择和排障见 [desktop/README.md](desktop/README.md)。

## 项目结构

```text
native-voice/
├── src/native_voice/        # 主 Python 包
│   ├── api/                 # FastAPI 应用、路由、百炼兼容 WS 协议
│   ├── backends/            # local / fake / worker 后端抽象
│   ├── conversation/        # 服务端全双工状态机、VAD、分句、打断
│   ├── engine/              # 独立 ASR / TTS GPU 引擎
│   ├── models/              # ASR / TTS / LLM 模型适配器
│   ├── security/            # API Key、配额、短票
│   ├── service/             # 进程监督、音色存储
│   ├── config/              # 配置模型与加载
│   ├── audio/               # PCM / 重采样 / 编码
│   └── observability/       # 日志与 Prometheus 指标
├── desktop/                 # 单机全双工 CLI（声卡直连，可选）
├── web/                     # 无构建网页客户端（ES modules + AudioWorklet）
├── tests/                   # unit / protocol / manual（GPU 手工检查）
├── benchmarks/              # 历史调优与性能对比脚本
├── configs/                 # service.yaml、voices.yaml
├── deploy/vllm-omni/        # vLLM-Omni 真流式 TTS 容器
├── docs/                    # 完整文档
├── examples/                # 协议客户端示例
├── scripts/                 # 安装、启动、压测脚本
├── assets/                  # Silero VAD 等运行时资源
├── models/                  # 模型权重（不入库，fetch_deps 下载）
├── bin/                     # 本地二进制（不入库）
└── third_party/             # 第三方源码（不入库）
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | 安装、依赖、两种运行方式、常用命令 |
| [docs/architecture.md](docs/architecture.md) | 架构、模块职责、数据流、插话状态机 |
| [docs/configuration.md](docs/configuration.md) | `service.yaml` 全字段与环境变量 |
| [docs/models.md](docs/models.md) | ASR/TTS 提供方对比、显存、实测性能、音色克隆 |
| [docs/api/](docs/api/README.md) | 百炼兼容 WS、对话 WS、REST 接口 |
| [docs/deployment.md](docs/deployment.md) | 生产部署、Docker、TLS、容量规划 |
| [docs/development.md](docs/development.md) | 开发环境、测试、代码规范、调试 |
| [docs/troubleshooting.md](docs/troubleshooting.md) | 常见问题与排障 |

## 参与贡献

欢迎提交 Issue 和 Pull Request。开始之前请阅读：

- [CONTRIBUTING.md](CONTRIBUTING.md) — 开发流程、提交规范、测试要求
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — 社区行为准则
- [SECURITY.md](SECURITY.md) — 安全问题报告方式

如果这个项目对你有帮助，欢迎 Star、分享，或从
[“good first issue”](https://github.com/easyon/native-voice/labels/good%20first%20issue)
开始参与。

## 许可证

本项目采用 [Apache License 2.0](LICENSE)。使用的模型与第三方组件遵循各自许可，
详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。商用前请确认模型许可与
声音克隆授权。

## 致谢

- [FunASR](https://github.com/modelscope/FunASR) / [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [Qwen3.5](https://www.modelscope.cn/models/Qwen/Qwen3.5-2B)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [vLLM](https://github.com/vllm-project/vllm) / [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [Silero VAD](https://github.com/snakers4/silero-vad)
