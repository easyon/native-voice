# CosyVoice 训练与运行时工程

本仓库以 `Fun-CosyVoice3-0.5B-2512` 为基线，覆盖数据清洗、训练、评测和在线推理。在线服务兼容阿里云百炼 CosyVoice 的 HTTP / SSE / WebSocket 协议，业务层只需切换 `base_url`、`api_key`、`model`、`voice`，无需改请求/响应代码。

设计文档：

- [总体方案](docs/design/cosyvoice-overview.md)
- [运行时方案](docs/design/cosyvoice-runtime.md)
- [训练方案](docs/design/cosyvoice-training.md)

## 工程分布

同一个仓库，**多个独立发行包**。每个包有自己的 `pyproject.toml`，可以单独构建、安装、发布；依赖不会互相污染。

```text
packages/
  common/       cosyvoice-common       共享音频/文本/错误模型，不依赖其他包
  runtime/      cosyvoice-runtime      在线服务，只依赖 common + FastAPI，不装 torch
  data/         cosyvoice-data         清洗、切分、manifest
  training/     cosyvoice-training     SFT / Flow / HiFiGAN，不依赖 runtime
  evaluation/   cosyvoice-evaluation   客观指标、MOS、回归门禁
configs/        runtime.yaml；后续 training/、evaluation/
scripts/        开发与运维脚本
tests/          common/runtime/data/training/evaluation
```

依赖方向：

```text
runtime ─┐
data ────┼──▶ common
training ┘
evaluation ─▶ common
```

`common` 不依赖任何其他包，`runtime` 与 `training` 互不依赖。运行时镜像只安装 `cosyvoice-runtime`，不会带入 torch/DeepSpeed；训练镜像只安装训练相关包，不会带入 FastAPI/uvicorn。

## 开发安装

需要 Python 3.10+。先装 `common`，再装目标包：

```bash
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate

pip install -e packages/common
pip install -e "packages/runtime[dev]"
```

Windows 可执行：

```powershell
.\scripts\dev-install.ps1
```

只做运行时开发时不需要安装 `training`、`data`、`evaluation`。

训练/数据/评测侧单独安装，不会带入运行时依赖：

```bash
pip install -e "packages/data[dev]"
pip install -e "packages/training[sft]"
pip install -e "packages/evaluation[metrics]"
```

## 快速开始（Mock 引擎）

Mock 引擎生成确定性音频，不需要 GPU 和模型，用于联调协议与压测网关。

```bash
cosyvoice-runtime
```

默认读取 `configs/runtime.yaml`，监听 `0.0.0.0:8000`。

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
curl http://localhost:8000/metrics
```

### HTTP 非流式

```bash
curl -X POST http://localhost:8000/api/v1/services/audio/tts/SpeechSynthesizer \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "model": "my-tts-prod",
    "input": {
      "text": "你好，今天想了解点什么？",
      "voice": "voice-a",
      "format": "wav",
      "sample_rate": 24000
    }
  }'
```

响应中的 `output.audio.data` 是 base64 音频；启用本地存储时还会返回 `output.audio.url`。

### HTTP SSE

增加请求头 `X-DashScope-SSE: enable`。事件顺序固定为：

```text
sentence-begin -> sentence-synthesis (可多次，每次带一个音频块) -> sentence-end -> 最终 finish_reason=stop
```

### WebSocket 双工

```text
ws://localhost:8000/api-ws/v1/inference
Authorization: Bearer test-key
```

1. `run-task`：传 `model` 和 `parameters`，`input` 为空。
2. 收到 `task-started` 后发 `continue-task`，`payload.input.text` 放文本。
3. 服务端自动分句，返回 `sentence-begin` / `sentence-synthesis` + binary 音频 / `sentence-end`。
4. `finish-task` 结束；`payload.input.directive="cancel"` 可取消当前轮次。
5. 同一连接可复用，每次任务使用新的 `task_id`。

消息样例见 [运行时方案 6.6 节](docs/design/cosyvoice-runtime.md#66-websocket-契约)。

## 配置

`configs/runtime.yaml` 的关键项：

| 配置 | 说明 |
|---|---|
| `engine.type` | `mock` 或 `cosyvoice` |
| `models.<name>` | 业务层直接使用的 model key，不做百炼别名映射 |
| `voices.<id>` | 业务层直接使用的音色 ID，不做映射；`mode` 支持 `sft/instruct2/zero_shot/auto` |
| `limits.max_concurrent_streams` | 单进程并发流上限（背压） |
| `limits.sentence_timeout_s` | 单句合成超时 |
| `limits.ws_queue_size` | WS 文本队列长度，满时返回 `Throttling` |
| `limits.ws_cancel_timeout_s` | `finish-task` 取消时等待合成任务停止的上限 |
| `storage.type` | `local` 落盘并返回 URL，`none` 仅返回 base64 |

环境变量覆盖：`COSYVOICE_RUNTIME_CONFIG`、`COSYVOICE_ENGINE_TYPE`、`COSYVOICE_API_KEYS`、`COSYVOICE_STORAGE_DIR`、`COSYVOICE_STORAGE_BASE_URL`、`COSYVOICE_LOG_LEVEL`。

## 接入真实 CosyVoice

```bash
pip install -e "packages/runtime[cosyvoice]"
```

1. 按 [训练方案](docs/design/cosyvoice-training.md) 产出模型包。
2. 把 `engine.type` 改为 `cosyvoice`，设置 `engine.cosyvoice.preload` 和 `models.<name>.artifact`。
3. 模型包目录需包含官方运行时所需的 `cosyvoice.yaml`、`llm.pt`、`flow.pt`、`hift.pt`、`spk2info.pt`、tokenizer 等。
4. 该模式为进程内加载；生产高并发建议按运行时方案的 LLM / Token2Wav 分离部署演进。

## 真实模型 / GPU 验证

单卡进程内模式（一个进程独占一张 GPU）：

```bash
# 0) 新机器一键部署：venv → 依赖 → 模型 → 配置 → smoke → 启动
bash deploy/bootstrap-gpu.sh
#    只准备不启动：--no-start；覆盖路径/端口用 PORT/MODEL_DIR/PROMPT_WAV/PROMPT_TEXT

# 1) 先脱离网关，确认模型、CUDA、声卡链路本身正常
COSYVOICE_REPO=/path/to/CosyVoice python scripts/gpu_smoke.py \
  --model-dir /models/Fun-CosyVoice3-0.5B \
  --prompt-wav /path/to/CosyVoice/asset/zero_shot_prompt.wav \
  --prompt-text 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。' \
  --text '你好，我是你的语音助手，很高兴为你服务。' --out /tmp/smoke.wav --runs 2

# 2) 起网关（预加载模型，storage=none）
COSYVOICE_RUNTIME_CONFIG=configs/runtime.gpu.yaml cosyvoice-runtime
```

`scripts/gpu_smoke.py` 输出 `load_s`、首包时延 `first_chunk_s`、`rtf`、`peak_gpu_mb`；
`configs/runtime.gpu.yaml` 是单卡验收模板，模型/音色路径按部署环境修改。
CosyVoice3 零样本克隆的 `prompt_text` 必须包含 `<|endofprompt|>` 标记。

单卡实测（A100-40G，2026-11-12）：fp32 单流 RTF ≈0.70、首包 2.0–3.0s、显存 3.4GB；
1/2/4 路并发的聚合实时倍率约 1.17×/1.56×/1.47×——上游所有请求共用一个 CUDA stream，
线程并发不能提升 GPU 并行度。**生产参数取 `max_concurrent=2`**，更高并发需要
LLM/Flow 批处理，详见[容量实测](docs/design/cosyvoice-runtime.md#859-单卡实测校准a100-40g--cosyvoice30-5b2026-11-12)。

批处理路径已接好、默认关闭：`bash deploy/bootstrap-gpu.sh --vllm` 启用 vLLM 连续 batching，
`--trt` 启用 Flow 的 TensorRT（首次启动编译 plan）。两者都必须先在目标 GPU 上用
`scripts/gpu_smoke.py --vllm/--trt --concurrency 4 --tasks 16` 与基线对比通过再切流量，
原因、预期收益与风险见[加速栈](docs/design/cosyvoice-runtime.md#8510-加速栈vllm-连续批处理--tensorrt-flow当前最优路径)，
操作与回滚见[运维手册 7.3](docs/ops/cosyvoice-runtime-runbook.md#73-加速栈vllm--tensorrt启用与回滚)。
零样本音色的 reference 特征会在加载时预热（`engine.cosyvoice.cache_prompt_features`，默认开），
fp16 在 A100 上更慢更占显存，保持关闭。


- [部署说明](deploy/README.md)：镜像、Helm、Secret、压测与上线检查。
- [运维手册](docs/ops/cosyvoice-runtime-runbook.md)：指标、告警、发布/回滚、故障处理、容量与验收。

```bash
helm upgrade --install cosyvoice deploy/helm/cosyvoice-runtime \
  --namespace cosyvoice --create-namespace \
  --set runtime.auth.apiKeys='replace-with-a-long-random-key'
```

网关压测（先 mock，再真实模型）：

```bash
python scripts/loadtest.py --mode ws  --concurrency 100 --tasks 5000 --api-key test-key
python scripts/loadtest.py --mode sse --concurrency 100 --tasks 5000 --api-key test-key
python scripts/loadtest.py --mode http --concurrency 100 --tasks 5000 --api-key test-key
```

生产容器以非 root、只读根文件系统、drop ALL capabilities 运行；`/healthz`、`/readyz`、`/metrics` 已提供。当前镜像和 chart 是**可部署的运行时基线**，不是已通过真实 GPU/真实模型容量验收的最终生产版本；上线前必须完成 [压测与验收流程](docs/ops/cosyvoice-runtime-runbook.md#7-压测与验收流程)。

## 测试

```bash
pytest
ruff check packages tests scripts
```

## 容器

只构建运行时包：

```bash
docker build -t cosyvoice-runtime .
docker run --rm -p 8000:8000 cosyvoice-runtime
```

真实模型镜像（GPU target）：

```bash
docker build --target gpu -t cosyvoice-runtime:gpu .
```

GPU 镜像的 torch 来自 conda（2.10.0+cu128），模型权重与 CosyVoice 上游代码不打包进镜像；构建、挂载和已知坑见 [镜像说明](docs/ops/cosyvoice-runtime-image.md)。

训练镜像应基于 `packages/training` 单独构建，不把 torch 带进运行时镜像。

## 当前边界

- v1 聚焦单说话人/少说话人实时通话，不做多说话人零样本平台。
- 未实现百炼全部 SSML/LaTeX 和 AIGC 隐性标识；超出范围的参数返回 `UnsupportedParameter`。
- 进程内引擎适合单机与联调；100/1000 并发的生产拓扑见运行时方案 8.5 节。
