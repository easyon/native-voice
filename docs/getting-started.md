# 快速开始

本文介绍如何从零安装 Native Voice 并跑通两条运行路径：

1. **服务模式**（推荐）：FastAPI 网关 + GPU engine + 网页客户端。
2. **单机 CLI**：`desktop/voice_call.py`，直接使用声卡，不启动网关。

## 1. 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10/11、Linux |
| Python | 3.10 |
| 包管理 | [uv](https://docs.astral.sh/uv/)（推荐）或 conda |
| GPU | NVIDIA，建议 ≥ 8 GB 显存；CPU 也可跑 LLM |
| 音频 | 麦克风 + 扬声器；建议戴耳机以避免回声误触打断 |
| Docker | 可选，仅 vLLM-Omni 真流式 TTS 需要 Linux 容器 / WSL2 |

开发验证过的最低配置：Windows 11 + RTX 4070 Laptop 8 GB。

## 2. 安装 Python 依赖

```bash
git clone https://github.com/easyon/native-voice.git
cd native-voice
uv sync --extra gpu
```

- `--extra gpu` 安装 torch / torchaudio（cu121）、FunASR、transformers 等模型运行时。
- 只在无 GPU 的环境里开发协议或网页时，使用 `uv sync --extra dev` 即可。
- **注意**：`uv run` 默认只同步基础依赖，会卸载 `gpu` extra。需要 GPU 时请始终写
  `uv run --extra gpu ...`，或直接使用 `.venv` 里的解释器 / 脚本。

也可以使用 conda（Python 3.10）。注意 `pyproject.toml` 里的 cu121 源只对
`uv` 生效，pip 需要显式安装 CUDA 版 torch：

```bash
conda create -n native-voice python=3.10 -y
conda activate native-voice
pip install torch==2.3.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[gpu]"
```

只跑 CPU / fake 后端时省略 torch 的 index-url，直接 `pip install -e ".[dev]"` 即可。

## 3. 下载模型与本地二进制

仓库不包含模型权重、llama.cpp 二进制和 CosyVoice 源码，首次使用需要下载：

```bash
uv run --extra gpu python scripts/fetch_deps.py
```

| 组件 | 默认来源 |
|---|---|
| `third_party/CosyVoice-main/` + Matcha-TTS | GitHub `FunAudioLLM/CosyVoice`、`shivammehta25/Matcha-TTS` |
| `models/Fun-CosyVoice3-0.5B/` | ModelScope `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| `models/Qwen3.5-2B-Q8_0.gguf` | ModelScope `unsloth/Qwen3.5-2B-GGUF` |
| `bin/llama/` | llama.cpp release win-cpu-x64 |
| 流式 Paraformer | ModelScope `iic/speech_paraformer-...-online`（首次启动也会自动下载） |

常用参数：

```bash
uv run --extra gpu python scripts/fetch_deps.py --check     # 只检查缺件
uv run --extra gpu python scripts/fetch_deps.py --dry-run   # 只显示下载计划
uv run --extra gpu python scripts/fetch_deps.py --only llama,tts
uv run --extra gpu python scripts/fetch_deps.py --source hf # 强制 HuggingFace
```

模型默认 **ModelScope 优先**，探测失败或单文件失败时回退 HuggingFace（`--source auto`）。
国内网络可设置 `NV_HF_ENDPOINT=https://hf-mirror.com`。

## 4. 启动服务模式

### 开发模式（gateway 自动拉起 engine）

```powershell
# Windows：8 GB 显卡建议把 LLM 放在 CPU
$env:LLM_DEVICE = "cpu"
scripts\serve_dev.ps1 -Backend local
```

```bash
# Linux / macOS
LLM_DEVICE=cpu scripts/serve_dev.sh
```

脚本会设置 `PYTHONPATH=src`，启动网关，并按 `configs/service.yaml` 的
`engine.mode: spawn` 拉起 ASR / TTS engine 和 llama-server。

> `LLM_DEVICE=cpu` 会选择 `bin/llama/`（CPU 版）并传 `-ngl 0`；不设置时使用
> `configs/service.yaml` 的默认值（CUDA 版 + `-ngl 99`）。`fetch_deps.py`
> 默认只下载 CPU 版 llama.cpp，因此全新环境请先设置 `LLM_DEVICE=cpu`，
> 或在 `bin/llama-cuda/` 准备 CUDA 版二进制。

打开 <http://127.0.0.1:8090/ui/>：

1. 从启动日志复制开发 Key（`--dev-key` 会打印；数据库已有 Key 时不打印）。
2. 粘贴到页面，点击“呼叫”。
3. 允许浏览器使用麦克风，开始说话。

### 无 GPU 联调

```powershell
scripts\serve_dev.ps1 -Backend fake
```

`fake` 后端使用假 ASR/TTS/LLM，适合开发网页、协议和 CI。

### 手动启动

```powershell
.venv\Scripts\native-voice.exe serve --backend local --dev-key --port 8090
```

或：

```bash
PYTHONPATH=src python -m native_voice.cli.main serve --backend local --dev-key --port 8090
```

生产模式（`engine.mode: external`）下分别启动 engine：

```bash
python -m native_voice.engine.cli asr --config configs/service.yaml --port 8101
python -m native_voice.engine.cli tts --config configs/service.yaml --port 8102
python -m native_voice.cli.main serve --backend worker --config configs/service.yaml
```

更完整的部署方式见 [deployment.md](deployment.md)。

### 创建正式 API Key

```powershell
python -m native_voice.cli.main key create --name pilot --scopes asr,tts,conversation
```

Token 只显示一次。`key list / disable / enable` 可管理已有 Key。

## 5. 启动单机 CLI（可选）

`desktop/` 是项目早期的轻量全双工实现，不经过网关，直接访问声卡。适合在
没有部署服务、只想快速验证模型链路时使用。

前置条件：

- conda 环境 `cosyvoice`（Python 3.10，torch 2.3.1+cu121，`x-transformers==2.11.24`）
- 已下载 `models/Fun-CosyVoice3-0.5B`、`models/Qwen3.5-2B-Q8_0.gguf`、
  `bin/llama/llama-server.exe`，以及 SenseVoiceSmall / 流式 Paraformer 缓存
- 安装 `sounddevice`：`pip install sounddevice`

```powershell
conda activate cosyvoice
cd D:\git\native-voice
python desktop\voice_call.py
```

或双击 `run_voice_call.bat`（如 Python 路径不同，请修改脚本中的 `PY`）。

首次启动约 20 秒加载模型。看到 `[状态] 聆听中` 后直接说话。

常用参数：

```powershell
python desktop\voice_call.py --list-devices      # 查看声卡设备编号
python desktop\voice_call.py --mic 1 --speaker 3 # 指定设备
python desktop\voice_call.py --text              # 纯文字模式，不开麦克风/扬声器
python desktop\voice_call.py --no-play           # 只跑 TTS 不出声
python desktop\voice_call.py --wav audio\asr_test.wav --save-audio
python desktop\voice_call.py --asr hybrid        # 流式 + SenseVoice 定稿
python desktop\voice_call.py --barge-db -24      # 提高打断阈值
python desktop\voice_call.py --endpoint-ms 650   # 句尾静音判定
python desktop\voice_call.py --verbose           # 打印识别耗时
```

完整说明与键盘操作见 [../desktop/README.md](../desktop/README.md)。

## 6. 下一步

- 了解系统设计：[architecture.md](architecture.md)
- 修改配置：[configuration.md](configuration.md)
- 选择模型：[models.md](models.md)
- 接入自己的客户端：[api/README.md](api/README.md)
