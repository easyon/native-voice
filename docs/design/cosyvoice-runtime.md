# CosyVoice 运行时技术方案

> 目标：把训练后的 CosyVoice3 模型部署成可私有化的实时 TTS 服务，兼容阿里云百炼 CosyVoice HTTP / SSE / WebSocket 接口；业务层只切换接入地址即可在云端与私有化之间切换。  
> 关联文档：[总体方案](./cosyvoice-overview.md) · [训练方案](./cosyvoice-training.md)

---

## 1. 范围

### 1.1 包含

- 模型加载与推理服务：LLM、Flow/DiT、HiFiGAN vocoder、speaker embedding、speech tokenizer。
- 阿里云百炼兼容层：HTTP 非流式、HTTP SSE 流式、WebSocket 双工。
- 音色管理：系统音色、复刻音色、speaker embedding、参考音频。
- 流式编排、音频转码、对象存储 URL、错误码、限流、观测。
- 单机部署与 K8s 生产部署。

### 1.2 不包含

- 模型训练/微调（见 [cosyvoice-training.md](./cosyvoice-training.md)）。
- 百炼全部音色、全部 SSML/LaTeX、AIGC 隐性标识的完整实现（v1 按能力矩阵提供）。
- 多租户计费系统（只保留计量字段和扩展点）。

---

## 2. 总体架构

```text
                         ┌──────────────────────────────┐
HTTP / SSE / WebSocket → │ API Gateway                  │
                         │ - TLS / Auth / Tenant        │
                         │ - 限流 / 配额 / 审计          │
                         │ - 阿里云兼容路由              │
                         └──────────────┬───────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │ Request Orchestrator         │
                         │ - task_id / 会话生命周期      │
                         │ - model/voice 别名映射        │
                         │ - 文本前端 / SSML 子集        │
                         │ - hot_fix / markdown filter  │
                         │ - 流式编排 / 取消 / 超时      │
                         └───────┬──────────────┬───────┘
                                 │              │
                    speech token │              │ audio chunks
                                 ▼              ▼
                 ┌───────────────────┐  ┌─────────────────────┐
                 │ LLM Service       │  │ Audio Pipeline      │
                 │ vLLM / TRT-LLM    │  │ - token2wav (Triton)│
                 │ text+spk+instruct │  │ - vocoder           │
                 │ → speech tokens   │  │ - ffmpeg transcode  │
                 └───────────────────┘  │ - URL / base64      │
                                        └──────────┬──────────┘
                                                   ▼
                                        ┌─────────────────────┐
                                        │ Object Storage      │
                                        │ MinIO / S3 / OSS    │
                                        └─────────────────────┘
```

关键原则：

- **LLM 与 Token2Wav 分开部署、分开扩缩容**。Token2Wav 通常是并发瓶颈。
- **API 层不做重计算**，只做协议、编排、转码和治理。
- **模型 artifact 外挂**，镜像不打包大模型，便于灰度与回滚。

---

## 3. 部署形态

### 3.1 开发/单机形态

```text
docker compose:
  - gateway (FastAPI + Uvicorn)
  - llm (vLLM OpenAI-compatible 或 transformers)
  - token2wav (Triton 或 Python 进程)
  - minio
  - prometheus / grafana (可选)
```

适用：功能联调、1–4 路并发、模型验收。

### 3.2 生产形态（推荐）

```text
K8s Namespace: cosyvoice
  Deployment: gateway            (CPU, 2–N 副本)
  Deployment: llm                (GPU pool A, vLLM/TRT-LLM)
  Deployment: token2wav          (GPU pool B, Triton)
  Deployment: audio-transcoder   (CPU, 可选，独立)
  StatefulSet/External: minio/s3
  HPA: gateway 按 QPS/连接数；llm/token2wav 按队列深度/TTFC
```

GPU 规划：

| GPU | 用途 | 说明 |
|---|---|---|
| L20 48GB | LLM + Token2Wav 同卡（小规模） | 开发/低并发；注意资源竞争 |
| L20/A100 | LLM 池 | TRT-LLM/vLLM，连续 batching |
| L20/A100 | Token2Wav 池 | Triton flow + hifigan，动态 batching |
| 多卡 | 分离部署 | LLM 一卡，Token2Wav 多卡；按 TTFC 扩容 |

### 3.3 分离部署

Token2Wav 是瓶颈时，按官方 Triton 方案拆分：

- LLM：`trtllm-serve` 或 vLLM，启用 in-flight batching。
- Token2Wav：Triton 部署 flow + vocoder，独立 GPU。
- Gateway 通过 gRPC/HTTP 调用两个服务。

好处：LLM 批处理不被 vocoder 阻塞；Token2Wav 可以独立扩副本。

---

## 4. 模型服务

### 4.1 LLM 服务

| 项 | 方案 |
|---|---|
| 引擎 | vLLM（快速落地）或 TensorRT-LLM（低延迟/高吞吐） |
| 输入 | text tokens + speaker embedding + instruct tokens |
| 输出 | 25Hz speech tokens |
| batching | 连续 batching / in-flight batching |
| 精度 | FP16/BF16；以精度回归为准 |
| 版本 | vLLM 仅支持 0.11.x+/0.9.0；0.10.x 未验证，必须锁版本 |

注意：

- vLLM 只加速 LLM，不加速 token2wav；端到端并发提升有限。
- LLM 输出 token 要按 `task_id` 分流，避免多路流式串包。
- 需要 warmup：不同文本长度、不同 speaker/instruct 各跑几条。

### 4.2 Token2Wav 服务

| 项 | 方案 |
|---|---|
| 引擎 | Triton + TensorRT/ONNX；或 Python + PyTorch |
| 模块 | Flow (Causal DiT 22 层) + CausalHiFT vocoder |
| 精度 | **FP32**；CosyVoice3 TRT `fp16=True` 有 NaN 问题 |
| batching | 动态 batching；按帧数/请求数双阈值 |
| 流式 | 按 token hop 生成音频 chunk |
| 缓存 | speaker embedding / prompt feature 缓存 |

### 4.3 Speaker embedding / Speech tokenizer

- Speaker embedding：`campplus.onnx`，CPU/GPU ONNX Runtime；结果缓存到 `spk2info`。
- Speech tokenizer：`speech_tokenizer_v3.onnx`，用于参考音频/提示音频转 token。
- 参考音频建议 ≤30 秒；超长会拖慢首包。

### 4.4 模型包布局

```text
model/
  Fun-CosyVoice3-0.5B/
    llm.pt
    flow.pt
    hifigan.pt
    spk2info.pt
    campplus.onnx
    speech_tokenizer_v3.onnx
    CosyVoice-BlankEN/
    config.yaml
  manifest.json          # 版本、commit、训练数据、评测指标
```

`manifest.json` 至少包含：

```json
{
  "model_id": "cosyvoice3-sft-20261112",
  "base_model": "Fun-CosyVoice3-0.5B-2512",
  "repo_commit": "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc",
  "sample_rate": 24000,
  "token_rate": 25,
  "voices": ["longanyang"],
  "metrics": {"cer": 1.2, "secs": 0.78, "mos": 4.2},
  "known_issues": []
}
```

### 4.5 推理路径选择

| 场景 | 路径 | 说明 |
|---|---|---|
| 固定人设 + 固定 instruct | `inference_sft(text, spk_id, instruct_text=...)` | 最快；不需要参考音频；需训练侧补丁 |
| 固定人设 + 动态情绪 | `inference_instruct2(text, instruct_text, prompt_wav, ...)` | 参考音频 + instruct；注意 spk_id 缓存 |
| 临时复刻/零样本 | `inference_zero_shot(text, prompt_text, prompt_wav)` | POC/兜底，不作为生产主路径 |
| 非流式变速 | `speed` 参数 | 流式下不要依赖 `speed`，用时间伸缩/分段 |

运行时按 `voice` 配置选择路径，不要在同一请求里混用。

---

## 5. 请求链路

### 5.1 HTTP 非流式

```text
Client → Gateway → 参数校验/映射 → LLM 全句 token → Token2Wav 全句音频
       → ffmpeg 转码 → 对象存储 → 返回 output.audio.url
```

特点：实现简单；首包 = 全句合成时间；适合短文本、离线播报、对延迟不敏感的接口。

### 5.2 HTTP SSE 流式

```text
Client → Gateway → LLM 流式 token → Token2Wav chunk → base64 音频
       → SSE data: {...} → Client 追加写文件
```

特点：兼容百炼 `X-DashScope-SSE: enable`；每个 `sentence-synthesis` 事件后跟一个音频数据块。

### 5.3 WebSocket 双工

```text
Client ── run-task ──▶ Gateway ──▶ LLM/Token2Wav
Client ◀─ task-started ─ Gateway
Client ── continue-task(text) ──▶ Gateway
Client ◀─ result-generated(sentence-begin/synthesis/end) + binary audio
Client ── finish-task ──▶ Gateway
Client ◀─ task-finished
```

特点：适合实时通话；连接复用；支持中途取消。

---

## 6. 阿里云兼容层

### 6.1 接口地址

#### HTTP

```text
POST https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer
```

私有化：

```text
POST https://tts.internal.example.com/api/v1/services/audio/tts/SpeechSynthesizer
```

路径保持不变。`{WorkspaceId}` 在私有化中可作为虚拟租户标识，也允许忽略。

#### WebSocket

```text
wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference
wss://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/inference
```

私有化：

```text
wss://tts.internal.example.com/api-ws/v1/inference
```

### 6.2 鉴权与请求头

| Header | 必选 | 说明 |
|---|---|---|
| `Authorization` | 是 | `Bearer <api_key>`；私有化映射到租户/配额 |
| `Content-Type` | HTTP 是 | `application/json` |
| `X-DashScope-SSE` | 流式时 | 固定 `enable` |
| `X-DashScope-Workspace` | 否 | 业务空间 ID；私有化映射到 tenant |
| `user-agent` | 否 | 客户端标识 |
| `X-DashScope-DataInspection` | 否 | 私有化默认不启用；如启用需单独合规配置 |

鉴权失败返回 HTTP 401/403；WebSocket 在握手阶段失败。

> 百炼新域名带 `{WorkspaceId}`，旧域名 `dashscope.aliyuncs.com` 仍可能被业务使用。私有化只需保证路径一致，域名由业务配置；建议在接入文档中明确“以控制台当前域名为准”。

### 6.3 HTTP 请求契约

```json
{
  "model": "cosyvoice-v3-flash",
  "input": {
    "text": "我家的后面有一个很大的花园。",
    "voice": "longanyang",
    "format": "wav",
    "sample_rate": 24000,
    "volume": 50,
    "rate": 1.0,
    "pitch": 1.0,
    "seed": 0,
    "language_hints": ["zh"],
    "instruction": "用温柔、稍慢的语气说。",
    "enable_ssml": false,
    "word_timestamp_enabled": false,
    "hot_fix": {
      "pronunciation": [{"天气": "tian1 qi4"}],
      "replace": [{"今天": "金天"}]
    },
    "enable_markdown_filter": false
  }
}
```

字段说明：

| 字段 | 类型 | 私有化行为 |
|---|---|---|
| `model` | string | 别名映射到本地模型；未知模型返回 `InvalidParameter` |
| `text` | string | 文本前端；支持纯文本；SSML 见 6.7 |
| `voice` | string | 别名映射到 `spk_id`/embedding/参考音频 |
| `format` | string | `mp3`/`pcm`/`wav`/`opus`；ffmpeg 转码 |
| `sample_rate` | int | 8k/12k/16k/22.05k/24k/44.1k/48k |
| `volume` | int | 0–100，增益映射；避免削波 |
| `rate` | float | 0.5–2.0；非流式用 `speed`，流式用时间伸缩/分段 |
| `pitch` | float | 0.5–2.0；变调不改时长优先，否则提示音质风险 |
| `bit_rate` | int | mp3/opus 码率 |
| `enable_ssml` | bool | v1 支持子集；不支持标签返回 `UnsupportedParameter` |
| `word_timestamp_enabled` | bool | 可选；用强制对齐生成；未启用时返回空 `words` |
| `seed` | int | 尽力复现；采样链路固定 seed |
| `language_hints` | array | 取第一个，映射到前端语言 |
| `instruction` | string | 追加 `<|endofprompt|>`，走 v3 instruct 路径 |
| `enable_aigc_tag` | bool | v1 可选；未实现时返回 `UnsupportedParameter` |
| `hot_fix.replace` | array | 文本替换，服务端预处理 |
| `hot_fix.pronunciation` | array | 拼音词典/前端注音；能力有限，未命中走默认 |
| `enable_markdown_filter` | bool | 服务端剥离 Markdown 标记 |

### 6.4 HTTP 非流式响应

```json
{
  "request_id": "ee88b03d-0457-9286-8c67-xxxxxxxxxxxx",
  "output": {
    "finish_reason": "stop",
    "audio": {
      "data": "",
      "url": "https://tts.internal.example.com/files/audio_xxx.wav",
      "id": "audio_ee88b03d-0457-9286-8c67-xxxxxxxxxxxx",
      "expires_at": 1772697707
    }
  },
  "usage": {
    "characters": 15
  }
}
```

私有化实现：

- 音频写入 MinIO/S3/OSS，返回预签名 URL，TTL 默认 24h。
- 若对象存储不可用，可降级返回 `data:audio/wav;base64,...` 作为扩展；业务需兼容。
- `usage.characters` 按有效字符统计，供计量/配额使用。

### 6.5 HTTP SSE 流式响应

请求头：

```text
X-DashScope-SSE: enable
```

响应为 `text/event-stream`，每个 `data:` 是一段 JSON，字段与非流式响应一致：

```text
data: {"request_id":"...","output":{"finish_reason":null,"type":"sentence-begin","original_text":"我家的后面有一个很大的花园。","sentence":{"index":0,"words":[]},"audio":{"data":""}},"usage":{"characters":0}}

data: {"request_id":"...","output":{"finish_reason":null,"type":"sentence-synthesis","sentence":{"index":0,"words":[]},"audio":{"data":"<base64-audio-chunk>"}},"usage":{"characters":0}}

data: {"request_id":"...","output":{"finish_reason":null,"type":"sentence-end","original_text":"我家的后面有一个很大的花园。","sentence":{"index":0,"words":[]},"audio":{"data":""}},"usage":{"characters":15}}

data: {"request_id":"...","output":{"finish_reason":"stop","audio":{"data":"","url":"https://.../audio_xxx.wav","id":"audio_xxx","expires_at":1772697707}},"usage":{"characters":15}}
```

实现要点：

- `sentence-synthesis` 与音频块一一对应，客户端按顺序追加。
- 最后一个句子可能没有 `original_text`。
- `finish_reason`：合成中 `null`，结束 `stop`。
- 连接中断时，服务端停止合成并释放 GPU。

### 6.6 WebSocket 契约

#### 6.6.1 连接

```text
wss://tts.internal.example.com/api-ws/v1/inference
Authorization: Bearer <api_key>
```

同一次合成任务中，`run-task`、所有 `continue-task`、`finish-task` 必须使用相同的 `task_id`；每次新任务生成新的 UUID。连接建议复用，避免每轮重新握手。

#### 6.6.2 run-task

```json
{
  "header": {
    "action": "run-task",
    "task_id": "2bf83b9a-baeb-4fda-8d9a-xxxxxxxxxxxx",
    "streaming": "duplex"
  },
  "payload": {
    "task_group": "audio",
    "task": "tts",
    "function": "SpeechSynthesizer",
    "model": "cosyvoice-v3-flash",
    "parameters": {
      "text_type": "PlainText",
      "voice": "longanyang",
      "format": "mp3",
      "sample_rate": 22050,
      "volume": 50,
      "rate": 1.0,
      "pitch": 1.0,
      "enable_ssml": false,
      "word_timestamp_enabled": false,
      "seed": 0,
      "language_hints": ["zh"],
      "instruction": "用温柔的语气说。"
    },
    "input": {}
  }
}
```

#### 6.6.3 task-started

```json
{
  "header": {
    "task_id": "2bf83b9a-baeb-4fda-8d9a-xxxxxxxxxxxx",
    "event": "task-started",
    "attributes": {}
  },
  "payload": {}
}
```

收到 `task-started` 后才能发 `continue-task`。

#### 6.6.4 continue-task

```json
{
  "header": {
    "action": "continue-task",
    "task_id": "2bf83b9a-baeb-4fda-8d9a-xxxxxxxxxxxx",
    "streaming": "duplex"
  },
  "payload": {
    "input": {
      "text": "床前明月光，疑是地上霜"
    }
  }
}
```

限制：

- 单次最多 20,000 字符；累计最多 200,000 字符。
- 发送间隔不得超过 23 秒，否则连接超时。
- 服务端自动分句；完整句子立即合成，不完整句子缓存。
- `enable_ssml=true` 时只允许一次 `continue-task`。

#### 6.6.5 result-generated

```json
{
  "header": {
    "task_id": "2bf83b9a-baeb-4fda-8d9a-xxxxxxxxxxxx",
    "event": "result-generated",
    "attributes": {}
  },
  "payload": {
    "output": {
      "type": "sentence-begin",
      "original_text": "床前明月光，",
      "sentence": {
        "index": 0,
        "words": []
      }
    }
  }
}
```

`type` 取值：

- `sentence-begin`：句子开始，返回文本。
- `sentence-synthesis`：音频块；随后立即通过 WebSocket binary 通道发送一帧音频。
- `sentence-end`：句子结束，返回累计计费字符数。

`words[]` 元素：

```json
{
  "text": "床",
  "begin_index": 0,
  "end_index": 1,
  "begin_time": 0,
  "end_time": 120
}
```

未启用字级时间戳时返回空数组。

#### 6.6.6 finish-task

正常结束：

```json
{
  "header": {
    "action": "finish-task",
    "task_id": "2bf83b9a-baeb-4fda-8d9a-xxxxxxxxxxxx",
    "streaming": "duplex"
  },
  "payload": {
    "input": {}
  }
}
```

取消当前轮次：

```json
{
  "header": {
    "action": "finish-task",
    "task_id": "2bf83b9a-baeb-4fda-8d9a-xxxxxxxxxxxx",
    "streaming": "duplex"
  },
  "payload": {
    "input": {
      "directive": "cancel"
    }
  }
}
```

取消后可在同一连接上发新的 `run-task`。注意：百炼北京地域 CosyVoice v2+ 支持 cancel，新加坡地域不支持；私有化 v1 统一支持 cancel。

实现上取消是“协作标记 + 队列哨兵 + `Task.cancel()`”三重保证：仅在任务协程上调用 `cancel()` 不够可靠——任务若正阻塞在 `asyncio.wait_for`（WS 每次发送都有发送超时）里，内层操作恰好完成时 CPython 可能吞掉 `CancelledError`。因此取消先置标记并投递哨兵，保证不再产生新的音频；同时以 `limits.ws_cancel_timeout_s` 为上限等待协程退出，慢客户端不会让 `finish-task` 永久挂起。服务端随后仍会返回 `task-finished`。

#### 6.6.7 task-finished / task-failed

```json
{
  "header": {
    "task_id": "2bf83b9a-baeb-4fda-8d9a-xxxxxxxxxxxx",
    "event": "task-finished",
    "attributes": {
      "request_uuid": "0a9dba9e-d3a6-45a4-be6d-xxxxxxxxxxxx"
    }
  },
  "payload": {
    "usage": {
      "characters": 13
    }
  }
}
```

```json
{
  "header": {
    "task_id": "2bf83b9a-baeb-4fda-8d9a-xxxxxxxxxxxx",
    "event": "task-failed",
    "error_code": "InvalidParameter",
    "error_message": "[tts:]Engine return error code: 418",
    "attributes": {}
  },
  "payload": {}
}
```

`task-failed` 后客户端应关闭或复用连接；服务端释放任务资源。

### 6.7 参数映射与兼容性矩阵

| 百炼能力 | 私有化 v1 | 实现方式 | 备注 |
|---|---|---|---|
| HTTP 非流式 | ✅ | 同步合成 + 对象存储 | 返回 URL |
| HTTP SSE 流式 | ✅ | `X-DashScope-SSE: enable` | base64 音频块 |
| WebSocket 双工 | ✅ | `/api-ws/v1/inference` | binary 音频 |
| 系统音色 | ✅ | 别名映射到 `spk2info` | 需导入百炼 voice ID |
| 复刻音色 | ✅ | 注册参考音频/embedding | 私有化管理接口 |
| 声音设计音色 | ⚠️ | 通过 instruct + 参考音色近似 | v1 不承诺 |
| `instruction` | ✅ | v3 instruct + `<|endofprompt|>` | 逐句 instruct 需拆分或打补丁 |
| 方言/多语言 | ✅ | 模型能力 + `language_hints` | 以模型覆盖为准 |
| `rate` | ⚠️ | 非流式 `speed`；流式时间伸缩 | 流式变速可能影响音质 |
| `pitch` | ⚠️ | 变调算法 | 可能引入伪影 |
| `volume` | ✅ | 增益 + limiter | 防削波 |
| `format`/`sample_rate` | ✅ | ffmpeg | mp3/pcm/wav/opus |
| `bit_rate` | ✅ | ffmpeg | mp3/opus |
| `seed` | ⚠️ | 固定随机种子 | 尽力复现，不保证跨版本 |
| `word_timestamp_enabled` | ⚠️ | 强制对齐/ token 时间 | v1 可选；未启用返回空 |
| `enable_ssml` | ⚠️ | 子集 | 见 6.8 |
| LaTeX | ❌ | 不支持 | 业务侧转文本 |
| `hot_fix.replace` | ✅ | 文本替换 | 简单可靠 |
| `hot_fix.pronunciation` | ⚠️ | 拼音词典/注音 | 覆盖有限 |
| `enable_markdown_filter` | ✅ | 服务端过滤 | 简单 |
| `enable_aigc_tag` | ⚠️ | 可选 | v1 默认不支持；需合规实现 |
| 连接复用 | ✅ | 连接池/会话 | 推荐 |
| 取消任务 | ✅ | `finish-task` + cancel | 私有化统一支持 |
| 错误码 | ✅ | 映射表 | 见 6.9 |

> 兼容细节：WebSocket 当前文档的 `sample_rate` 列表不含 12000，遇到该值应返回 `UnsupportedParameter` 或明确降级，不要静默改采样率；`enable_ssml=true` 时只允许一次 `continue-task`。

### 6.8 SSML 子集

CosyVoice3 开源前端不原生支持完整 SSML。v1 建议实现：

| 标签 | 支持 | 处理 |
|---|---|---|
| `<break time="300ms"/>` | ✅ | 转标点/插入静音；长停顿分段合成 |
| `<prosody rate="slow/fast">` | ⚠️ | 映射到 `rate`；仅整段有效 |
| `<prosody pitch="+2st">` | ⚠️ | 映射到 `pitch`；音质风险 |
| `<emphasis>` | ⚠️ | 文本前后加标点/重音提示；效果有限 |
| `<say-as interpret-as="...">` | ⚠️ | 数字/日期/电话等文本规范化 |
| `<voice>` | ❌ | 不支持多音色混排 |
| `<lang>` | ⚠️ | 映射到 `language_hints`；需分段 |

实现策略：

- 先解析 SSML 为 segment 列表：`{text, pause_ms, rate, pitch}`。
- 逐段合成，段间插入静音；流式时把静音作为音频块发送。
- 不支持的标签：默认降级为纯文本并记录 warning；严格模式返回 `UnsupportedParameter`。

### 6.9 错误码映射

| 场景 | HTTP | WS `error_code` | 说明 |
|---|---|---|---|
| 缺少/无效 API Key | 401/403 | `Unauthorized` | 握手失败 |
| 参数缺失/非法 | 400 | `InvalidParameter` | 字段名、范围、未知 model/voice |
| 文本过长 | 400 | `TextTooLong` | 单次/累计限制 |
| 音色不存在 | 400 | `VoiceNotFound` | 别名未配置 |
| 不支持的格式 | 400 | `UnsupportedFormat` | format/sample_rate |
| 不支持的 SSML | 400 | `UnsupportedParameter` | 严格模式 |
| 队列满/限流 | 429 | `Throttling` | 返回 Retry-After |
| 上游超时 | 504 | `Timeout` | LLM/Token2Wav |
| 模型未就绪 | 503 | `ServiceUnavailable` | readiness 失败 |
| 内部错误 | 500 | `InternalError` | 记录 request_id |

HTTP 错误体建议：

```json
{
  "request_id": "req-xxx",
  "code": "InvalidParameter",
  "message": "voice not found: longanyang",
  "retryable": false
}
```

### 6.10 model / voice 配置（不做别名映射）

业务层直接使用这里的 key，运行时不做百炼 model/voice 别名映射，也不把云端 ID 翻译成内部 ID。切换云端/私有化时，业务配置里的 `model`/`voice` 由业务自己保持一致。

```yaml
models:
  my-tts-prod:
    engine: cosyvoice3
    artifact: /models/cosyvoice3-sft-20261112
    default_voice: voice-a

voices:
  voice-a:
    spk_id: voice-a
    embedding: /models/voices/voice-a/embedding.pt
    prompt_wav: /models/voices/voice-a/prompt.wav
    prompt_text: "参考音频对应的文本"
    instruct: "你是一个温柔、自然的助手。"
```

要求：

- **未知 `model`/`voice` 不能静默降级**，返回 `InvalidParameter`，避免业务以为切换成功但音色不对。
- 配置通过 ConfigMap/配置文件在发布时更新；当前版本不支持热更新，变更走滚动发布。
- `default_voice` 必须存在于 `voices`，启动时校验。

---

## 7. 能力提供

### 7.1 对外能力

| 能力 | 接口 | 说明 |
|---|---|---|
| 非流式合成 | HTTP | 返回音频 URL |
| SSE 流式合成 | HTTP | 返回 base64 音频块 |
| WebSocket 实时合成 | WSS | 双工、可取消、连接复用 |
| 音色列表 | 私有化管理 API | 业务可选音色 |
| 音色注册/删除 | 私有化管理 API | 参考音频 + prompt text |
| 健康检查 | `/healthz` `/readyz` | K8s |
| 指标 | `/metrics` | Prometheus |
| 模型/音色元数据 | `/admin/models` `/admin/voices` | 运维 |

### 7.2 私有化管理 API（非百炼兼容）

```text
POST   /admin/voices
       multipart: voice_id, prompt_wav, prompt_text, instruct
GET    /admin/voices
GET    /admin/voices/{voice_id}
DELETE /admin/voices/{voice_id}
POST   /admin/models/reload
GET    /admin/models
```

注册流程：

1. 校验音频：单声道、≥16kHz、≤30s、SNR 达标。
2. 提取 CAM++ embedding；可选提取 speech token。
3. 写 `spk2info.pt`/voice 目录；原子更新别名表。
4. 预热：用固定文本跑一条，确认首包和音质。

> 动态 instruct 注意：`frontend_zero_shot` 在 `zero_shot_spk_id` 非空时会复用缓存的 `model_input`，同一个 spk_id 换 instruct 可能不生效。生产建议按“音色 + 情绪模板”注册多个 speaker id，或 patch `frontend_instruct2` 强制覆盖 `prompt_text`。

### 7.3 能力降级

| 故障 | 降级 |
|---|---|
| Token2Wav 队列满 | 429 + Retry-After；或转非流式 |
| LLM 超时 | 返回 `Timeout`；不缓存半句音频 |
| 对象存储不可用 | 非流式返回 base64 data URI（扩展） |
| GPU OOM | 拒绝新请求，触发告警；不杀已有流 |
| 模型未加载 | readiness 失败，流量摘除 |

---

## 8. 并发与性能

### 8.1 瓶颈

端到端链路：

```text
文本前端 → LLM prefill/decode → Token2Wav flow → vocoder → 转码
```

- LLM 可 batching，但 **Token2Wav/vocoder 是主要瓶颈**。
- 25Hz token 率使 token 数偏多；Token2Wav 计算量随音频时长增长。
- 同卡部署时 LLM 与 Token2Wav 争抢 SM/显存，首包和吞吐都会抖。

### 8.2 优化手段

| 手段 | 收益 | 代价 |
|---|---|---|
| LLM/Token2Wav 分离部署 | 避免资源竞争；独立扩容 | 多卡/多服务 |
| TensorRT-LLM | LLM 延迟/吞吐提升 | 编译、版本锁定 |
| 进程内 vLLM / TensorRT（8.5.10） | 单卡连续 batching，突破共享 CUDA stream | 首次导出/编译；锁版本；需 GPU 验收 |
| Triton 动态 batching | Token2Wav 吞吐提升 | 首包略增 |
| FP32（避免 FP16 NaN） | 稳定 | 显存/算力增加 |
| CUDA Graph / TensorRT | 降低 kernel 开销 | 输入 shape 约束 |
| speaker embedding 缓存 | 降低首包 | 音色变更需失效 |
| 流式 chunk 调参 | 降低首包 | 太小影响韵律 |
| 文本分句/短句优先 | 首包更快 | 长句韵律可能受影响 |
| 连接复用 | 降低握手/冷启动 | 需要会话管理 |

### 8.3 流式参数

CosyVoice3 关键参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `token_hop_len` | 25 | 每 25 个 token（1s 音频）一个 chunk |
| `token_max_hop_len` | 100 | 最大 hop |
| `stream_scale_factor` | 2 | 流式倍率 |
| `pre_lookahead_len` | 3 | 前瞻 token |
| `prompt_token_pad` | 0–24 | 参考音频 prompt padding |

首包逻辑：

- 第一包需要 `token_hop_len + pre_lookahead_len = 28` 个 token，约 1.12s 音频。
- 首包延迟是“生成这 28 个 token + flow/vocoder 时间”，不是 1.12s 本身。
- 参考音频越长，prompt token 越多，首包越慢；建议 ≤10s，最多 30s。

### 8.4 SLO（初始目标）

| 指标 | 目标 | 说明 |
|---|---:|---|
| TTFC p50 | ≤ 600 ms | 4 路并发/L20 起 |
| TTFC p95 | ≤ 1000 ms | 生产告警线 |
| TTFC p99 | ≤ 1500 ms | 容量规划 |
| RTF | ≤ 0.10 | 离线/批量 |
| inter-chunk gap p95 | ≤ 200 ms | 流式播放不卡顿 |
| 错误率 | < 0.1% | 不含 4xx |
| 可用性 | 99.9% | 多副本 |

> 以上是目标值，不是开箱承诺；以实际 GPU、模型、并发压测为准。
>
> 分级目标：100 active 按 p95 ≤1.0s；1000 active 默认按 p95 ≤1.5s 规划。容量主要取决于**起播速率**，不是单纯的 active 数：正常分散起播约 110–190×L20，1000 路同时起播才需要约 360–430×L20；详见 8.5.2。

### 8.5 100 / 1000 并发容量方案

#### 8.5.1 先定义“并发”

容量规划必须区分三个概念：

| 概念 | 含义 | 是否占 GPU |
|---|---|---|
| 连接数 `C_conn` | WebSocket 已建立、可能空闲的会话 | 否，占 Gateway 内存/FD |
| 活跃合成数 `C_active` | 正在生成音频的流 | 是，主要成本 |
| 请求速率 `RPS` | 非流式 HTTP 每秒请求数 | 由 Little's law 折算 |

```text
C_active = C_conn × active_ratio × burst_factor
C_active ≈ RPS × avg_synthesis_seconds     # 非流式
```

示例：

- 1000 路通话，30% 时间在播 TTS → 约 300 路活跃；叠加 1.2 突发 → 约 360 路；按 5s 一句估算，Token2Wav 约 40–70 张 L20，而不是 130–150 张。
- 如果业务说“1000 并发”指 **1000 路同时合成**，按最坏情况 1000 active 规划；若还要求 1000 路在同一秒起播，才需要 360–430 张 L20。
- 实时通话有明显波峰；`burst_factor` 建议 1.2–1.5。

#### 8.5.2 Token2Wav GPU 估算：稳态 + 首包两段式

之前把官方“4 路并发 TTFC p95≈978ms”直接当成每卡只能带 2.8 路，这是**最坏口径**：它假设所有流同时处于首包阶段。真实通话里，大部分流已经起播，稳态消耗由 RTF 决定；首包只是短时突发。两者要分开算。

官方数据：

- 离线 RTF（batch 1/4/16）：0.109 / 0.063 / 0.050
- 4 路并发 TTFC p95：978ms

**口径 A：稳态吞吐（流已经起播）**

```text
N_safe = 0.7 / RTF
GPU_steady = ceil(C_active / N_safe)
```

| GPU | RTF 0.109（batch 1） | RTF 0.063（batch 4） |
|---|---:|---:|
| L20 | 6.4 路/卡 | 11.1 路/卡 |
| 100 active | 16 张 | 9 张 |
| 1000 active | 156 张 | 90 张 |

**口径 B：首包突发（新句子/新通话起播）**

官方 4 路并发 TTFC p95≈1s；按 70% 水位，约 2.8 路同时起播/卡。

```text
GPU_burst = ceil(S × 1s / 2.8) = ceil(S / 2.8)
```

`S` 是峰值起播速率（起播/s），不是连接数。

| 峰值起播 S | L20 |
|---:|---:|
| 10/s | 4 |
| 20/s | 8 |
| 100/s | 36 |
| 200/s | 72 |
| 1000/s | 358 |

**总规划**

```text
GPU_total ≈ ceil(max(GPU_steady, GPU_burst) × 1.2)
```

| 场景 | L20 48GB | A100 80GB | H100 80GB |
|---|---:|---:|---:|
| 100 active，起播 20/s | 11–20 | 6–10 | 4–7 |
| 100 active，同时起播 | 36–44 | 18–22 | 12–15 |
| 1000 active，起播 200/s | 110–190 | 55–95 | 36–65 |
| 1000 active，同时起播 | 360–430 | 215–225 | 144–150 |

说明：

- 100 active、5s 一句 → S≈20/s；1000 active → S≈200/s。这是正常通话的估算。
- 只有外呼集中拨号/全量同时起播，才会落到“同时起播”一行；不要把它当常态。
- 注意：官方 4 路并发是“TTFC p95≈1s”的压测点，不代表单卡上限；稳态上限由 RTF 决定。
- A100≈2×L20、H100≈3×L20 是**规划假设**，不是承诺；必须用同一模型/同一文本/同一并发压测校准。
- 上表是 Token2Wav 池，不含 LLM/Gateway；已含约 20% 余量，仍建议再加 N+1 或 10% 热备。
- `max` 是工程近似；稳态接近 70% 时，首包突发会叠加，需压测确认，必要时按 `GPU_steady + GPU_burst` 预留。
- 如果 1000 路是“连接数”而非“活跃流”，按 `C_active` 重算。例如 30% 活跃、5s 一句 → 360 active、S≈72/s，约 40–70 张 L20。

#### 8.5.3 LLM / Gateway 估算

| 组件 | 100 active | 1000 active | 说明 |
|---|---:|---:|---|
| LLM 池 | 2–4×L20 或 1–2×A100 | 8–16×L20 或 4–8×A100 | LLM 通常不是瓶颈；按 TTFT/队列实测 |
| Gateway（WS） | 4–6 pods | 12–20 pods | 2–4 vCPU/pod；粘性会话；连接复用 |
| Gateway（HTTP 非流式） | 8–12 pods | 30–50 pods | 等待完整音频，连接占用更久 |
| Redis / 会话 | 3 节点起 | 6 节点起 | 任务状态、限流、配额 |

LLM 建议用 vLLM 连续 batching 或 TensorRT-LLM；1000 路时重点看 **decode token/s** 和 **TTFT**，不要只看 GPU 利用率。

#### 8.5.4 网络与对象存储

单路 24kHz/16bit 单声道 PCM：

```text
24,000 × 16 bit = 384 kbps ≈ 48 KB/s
100 路  ≈ 4.8 MB/s ≈ 38 Mbps
1000 路 ≈ 48 MB/s  ≈ 384 Mbps
```

规划建议：

- 接入层按 2× 峰值规划：100 路至少 100Mbps，1000 路至少 1Gbps；跨 AZ/跨区再留余量。
- 若输出 mp3/opus 32–128kbps，带宽显著更低；PCM/WAV 流式最占带宽。
- 非流式 1000 路、平均 10s 音频：约 480MB/批；按 100 请求/s 估算，对象存储写入约 50MB/s，建议按 100MB/s 留余量。
- WebSocket 场景不落盘；只在需要审计/回放时异步写对象存储。

#### 8.5.5 100 路部署拓扑

```text
单 Region / 2–3 AZ
  └── K8s Cluster
      ├── gateway: 4–6 pods
      ├── llm: 1–2 A100（或 2–4 L20）
      ├── token2wav: 11–20 L20（分散起播）/ 36–44 L20（同时起播）
      ├── redis: 3 节点
      └── minio/s3: 对象存储
```

适用：单业务线、单模型、单说话人；可用 2 AZ + 反亲和保证单 AZ 故障不影响整体。

#### 8.5.6 1000 路部署拓扑：Cell 化

不建议把 100+ 张 L20 塞进一个 K8s 集群。建议按 **Cell** 拆分：

```text
Global Router / GSLB
  ├── Cell A: ~330 active
  │     ├── gateway 6–8 pods
  │     ├── llm 2–4 A100
  │     ├── token2wav 60 L20
  │     └── redis/minio
  ├── Cell B: ~330 active / 60 L20
  ├── Cell C: ~340 active / 60 L20
  └── Cell D: burst 池（同时起播/故障接管，按需 60–240 L20）
```

规则：

- 正常分散起播：3 个 Cell（每 Cell 55–65 张 L20）可承载约 1000 active。
- 1000 路同时起播：需要额外 burst 池，等效约 6–7 个 Cell；不要把它当常态，优先用排队/错峰/外呼限速削峰。
- 每个 Cell 独立故障域；单 Cell 故障只影响约 1/3 流量。
- 按租户/会话做一致性哈希或粘性路由；同一通电话尽量落同一 Cell。
- 全局层负责配额、鉴权、模型版本、音色注册、观测；Cell 层负责推理。
- 1000 路跨地域时，每地域独立 Cell，数据/音频不出域；GSLB 按延迟和容量调度。
- 预留 10–20% 热备或 burst 池，用于故障接管和突发。

#### 8.5.7 扩缩容与成本

- 基线 60–70% 用预留/包年，峰值 30–40% 用按量；实时业务不要依赖 Spot。
- 离线批量/评测可用 Spot，但不能和在线池混部。
- HPA 指标：队列深度、TTFC p95、GPU 利用率；扩容冷启动 2–10 分钟，必须 warm pool。
- 1000 路总成本约是 100 路的 8–10 倍（同口径，GPU 数量近似线性）；p95 放宽到 1.5s 主要降低首包冗余，稳态 GPU 不变，总成本下降有限。1000 路可通过更高单卡并发、Cell 摊薄和预留折扣把单路成本降低 20–30%，但不会改变 GPU 是主要成本这一事实。
- 成本优化优先级：提高单卡安全并发（TRT/Triton）> 降低 TTFC 目标 > 混部 > Spot。

#### 8.5.9 单卡实测校准（A100-40G + CosyVoice3-0.5B，2026-11-12）

以下是本仓库运行时在真实 A100-40G 上的实测，用于校准上面的规划假设。测试条件：单进程、零样本音色（prompt ~5s）、fp32、TF32 开启、`max_concurrent=4`、每任务文本约 27 字（≈5s 音频）、WebSocket 流式、空载 GPU。

| 并发 | 请求数 | 成功 | TTFF p50 / p95 (ms) | 端到端 p50 (ms) | 吞吐 (req/s) | 聚合实时倍率 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 4 | 1993 / 2898 | 3458 | 0.24 | 1.17× |
| 2 | 8 | 8 | 3683 / 5014 | 6160 | 0.32 | 1.56× |
| 4 | 16 | 16 | 8363 / 8827 | 13312 | 0.30 | 1.47× |
| 8 | 32 | 0 | — | — | — | 全部超时（受 `max_concurrent=4` 限制） |

同期 GPU 采样：利用率 37–50%，显存 6.6 GB。

结论（重要，推翻 8.5.2 的部分乐观假设）：

- **单卡有效吞吐约 1.5× 实时**，不是按 RTF 0.1 推算的 6–11 路。线程并发只能把 CPU 侧空隙填满，不能增加 GPU 并行度。
- 原因是上游 `CosyVoice3Model` 的 `tts()` 全部进入同一个 `self.llm_context` CUDA stream（`cosyvoice/cli/model.py`），多线程的 kernel 在该 stream 上串行；增加线程只会让首包线性变差。
- 按此实测，100 路活跃约需 **60–70 张 A100**（或 ~130 张 L20），成本不可接受。要回到 8.5.2 的估算，必须先做**批处理**：LLM 侧 vLLM 连续 batching、Flow/DiT 侧 TensorRT 或 Triton 动态 batch，而不是继续加线程。
- 在批处理落地前，生产参数取 `max_concurrent=2`（首包 3.7s 已是通话可接受上限），并用多进程/多卡横向扩，不要用单进程高并发。

TF32 / fp16 实测：

| 配置 | RTF（5 次中位数） | 峰值显存 | 结论 |
|---|---:|---:|---|
| fp32（TF32 关） | 0.747 | 3.4 GB | 基线 |
| fp32（TF32 开） | 0.746 | 3.4 GB | 无可测差异（conv 默认已用 TF32），保留默认开 |
| fp16 | 1.05 | 5.0 GB | **更慢且更占显存**，A100 上禁用 |

首包构成（零样本）：文本前端 + prompt 特征 + LLM 预填 + 首批 28 token 的 Flow/Vocoder。把 prompt 特征缓存到 `add_zero_shot_spk` 后，同进程首包实测从 2.08–2.19s 降到 1.69–1.70s（-0.4s，约 20%）；网关侧 WS 首包 2.49s → 2.24s、HTTP 3.29s → 3.00s。说明 prompt 特征提取约占首包 0.4s，**剩余大头是 LLM 预填与首批 Flow**。要把首包压到 1s 以内，需要短参考音频（≤5s）+ Flow 的 JIT/TRT + 更小的首批 hop，单靠工程侧缓存不够。

#### 8.5.10 加速栈：vLLM 连续批处理 + TensorRT Flow（当前最优路径）

8.5.9 的结论是“单进程线程并发无法提高 GPU 并行度”。在本仓库后续验证中又确认了两点：

- **多进程 + CUDA MPS 可行但只是横向扩容**：4 个进程 + MPS 实测约 1.09 req/s，接近线性，说明 GPU 本身还有余量，浪费发生在单进程内部；代价是每卡 N 份显存和 N 份进程管理，单位请求成本不降。
- **并行度被上游实现锁死**：`CosyVoice3Model.tts()` 的 LLM 工作全部投递到同一个 `self.llm_context` CUDA stream，多线程只是在 CPU 侧填空隙。

因此唯一能降低单卡成本的路径是**批处理**，且优先用上游（FunAudioLLM）已经支持的两条路径，而不是在本仓库自研调度器：

| 加速项 | 解决的问题 | 上游实现 | 开关 |
|---|---|---|---|
| vLLM | LLM 逐 token 生成在单 stream 上串行 | `Qwen2LM.inference_wrapper` 的 vLLM 分支：`add_request` + 锁内 `step()`，跨请求连续 batching | `engine.cosyvoice.load_vllm` |
| TensorRT | Flow/DiT estimator 的 kernel 启动与推理开销 | 由 `flow.decoder.estimator.fp32.onnx` 构建 `fp32.<gpu>.plan`，`TrtContextWrapper(trt_concurrent)` 提供多 context/stream | `engine.cosyvoice.load_trt`、`trt_concurrent` |

实现约定：

- 两个开关默认 `false`；`load_vllm` 首次启动把 LLM 导出到 `<artifact>/vllm` 并卸载进程内 HF LLM 层，所以 artifact 目录必须可写，或预导出后只读挂载；`load_trt` 首次启动在 artifact 内编译 plan，同样需要可写或预编译。
- 运行前预检（`engines/cosyvoice.py::_check_acceleration`）：模块未安装、ONNX/plan 缺失、目录不可写时直接返回 503 `ServiceUnavailable` 并给出修复提示，不把上游的裸 `ImportError` 暴露给业务，也不在权重加载一分钟后才失败。
- plan 与 GPU 架构（含 MIG slice）绑定；换卡类型必须重新编译，不要把 A100 的 plan 带到 H100。

为什么不自己写 batched decode / batched flow：

- 需要同时改 `tts()`、`llm_job`、`token2wav` 三处状态机和跨线程 stream 归属，任何一处出错都表现为偶发串音或韵律退化，而这类问题只能靠 GPU 上的音质回归发现。
- 请求间还要隔离 prompt、instruct、取消和 KV cache 生命周期，等于重写上游推理核心，后续无法跟随上游修复。
- vLLM 路径已经由上游维护并覆盖连续 batching；TensorRT 路径是上游官方导出脚本的产物。自研版本只有在实测证明这两条路径不够时才值得做。

收益与验收（**尚未在 GPU 上完成验证，不能预先写进容量承诺**）：

- 判定标准：同一 GPU、同一音色、同一文本下，`c=4` 的聚合实时倍率必须显著高于 fp32 基线的 1.47×；TTFF p50 不得劣于基线；音频需人工抽检，确认无串音、无异常韵律。
- 若 vLLM 只提升吞吐但首包变差，保留 vLLM、单独回退 TensorRT（或反之），两个开关互不依赖。
- 两项都开启后仍不达标，才进入“自研 batched token2wav / 分离部署”的下一阶段。

风险：

- `vllm==0.19.1` 锁定 `torch==2.10.0`（与已验证栈一致），但会把 `transformers` 升到 4.57.1；非 vLLM 部署继续保持 4.51.3。升级后必须重跑文本前端与音色回归。
- vLLM 的采样实现与上游 Python sampler 不保证逐 bit 一致，需按音色抽检。
- TensorRT 与 fp16 不能叠加：上游明确 fp16 DiT engine 有性能与音质问题；`engine.cosyvoice.fp16` 保持 `false`。
- 首次启动会多出导出/编译时间（分钟级），K8s 需放宽 startupProbe 或预生成产物。

#### 8.5.11 100 / 1000 压测门槛

| 规模 | 必测 | 通过标准 |
|---:|---|---|
| 100 | 1/25/50/100 路阶梯，30min | p95 ≤1.0s，错误率 <0.1%，无 OOM |
| 1000 | 100/300/500/800/1000 路阶梯，2h soak | p95 ≤1.5s（分散起播；或扩容到 ≤1.0s），错误率 <0.5%，无雪崩 |
| 起播突发 | 100/500/1000 路同时起播 | TTFC 退化可观测，不雪崩；必要时限流/错峰 |
| 故障 | 杀 1 个 Token2Wav 实例/1 个 Cell | 自动摘除，TTFC 恢复 <5min |
| 长稳 | 1000 路 2h | 内存/显存无泄漏，FD 不增长，队列不累积 |

压测必须分别测“分散起播”和“同时起播”，否则会把最坏值当常态；文本用真实通话分布，不要只用 10 字短句，并覆盖长句、取消、重连、静音、异常断连。

> 100 路按 11–20 张 L20 是分散起播的起点；如果实测单卡稳态能带更多路，再按实测下调。1000 路同理，先小规模找 knee，再线性外推；同时起播只作为 burst 验收，不作为常态容量。

### 8.6 背压与配额

- 每租户并发上限、QPS 上限、文本长度上限。
- 队列满返回 429 + `Retry-After`。
- 长文本拆分；超过阈值建议异步/非流式。
- 超时：首包 5s、总时长 120s（可配置）。
- 熔断：Token2Wav 错误率/队列深度超阈值，暂停新流。
- 降级：流式失败可重试非流式（若业务允许）。

### 8.7 自动扩缩容

| 服务 | 指标 | 策略 |
|---|---|---|
| Gateway | QPS、活跃连接、CPU | HPA 2–N |
| LLM | 队列深度、TTFT、GPU 利用率 | HPA/自定义指标 |
| Token2Wav | 队列深度、TTFC p95、GPU 利用率 | 独立 HPA |
| 对象存储 | 容量/错误率 | 告警 |

扩容冷启动要预热模型；建议保持 warm pool，避免扩容后首包恶化。

---

## 9. 高可用与运维

### 9.1 健康检查

- `/healthz`：进程存活。
- `/readyz`：模型加载、GPU 可用、对象存储可达、warmup 完成。
- 启动顺序：加载模型 → 预热 → readiness 通过 → 接流量。
- 优雅下线：停止接新任务，等待已有流结束/超时。

### 9.2 发布

- 模型 artifact 与镜像分离；模型版本用 `model_id` 固定。
- 灰度：按租户/百分比导流；对比 TTFC、错误率、SECS/CER。
- 回滚：切回上一版 `model_id`；保留至少 2 个版本。
- 配置热更新：model/voice 别名原子切换。

### 9.3 可观测性

指标：

```text
cosyvoice_requests_total{model,voice,api,code}
cosyvoice_ttfc_seconds_bucket{model,api}
cosyvoice_rtf_bucket{model}
cosyvoice_stream_gap_seconds_bucket
cosyvoice_llm_tokens_total
cosyvoice_token2wav_frames_total
cosyvoice_queue_depth{service}
cosyvoice_gpu_memory_used_bytes
cosyvoice_errors_total{code}
```

日志：

- 必带 `request_id` / `task_id` / `tenant` / `model` / `voice`。
- 默认不记录完整文本和音频；需要排查时采样并脱敏。
- 记录分段耗时：frontend、LLM、flow、vocoder、transcode、upload。

追踪：

- OpenTelemetry：Gateway → LLM → Token2Wav → 对象存储。
- 慢请求采样 100%。

### 9.4 安全与合规

- TLS；API Key 定期轮换；支持 mTLS（可选）。
- 租户隔离：模型、音色、对象存储前缀、配额。
- 参考音频加密存储；访问审计。
- 数据保留：音频 URL TTL 24h；日志保留按合规策略。
- AIGC 标识：按业务合规要求实现；v1 可配置关闭。
- 禁止未授权声音复刻；注册时校验授权声明。

---

## 10. 测试与验收

### 10.1 契约测试

- 用同一份请求分别打百炼和私有化，比较 JSON schema/字段，不比较音频二进制。
- 覆盖：HTTP 非流式、SSE、WS run/continue/finish、cancel、错误码、边界值。
- 用官方 SDK（Python/Java）直连私有化地址，验证“只换 base_url”。

### 10.2 压测

| 场景 | 并发 | 文本 | 指标 |
|---|---:|---|---|
| 短句 | 1/2/4/8/16 | 10–20 字 | TTFC、RTF、gap |
| 长句 | 1/2/4/8 | 100–300 字 | 总时长、gap、错误 |
| 多轮 WS | 50/100 连接 | 连续任务 | 连接复用、内存 |
| 取消 | 20% 取消 | 混合 | 资源释放 |

每次压测记录：GPU、模型版本、精度、batch 参数、p50/p95/p99。

### 10.3 验收清单

- [ ] 官方 SDK 只改 base_url 可调用 HTTP/SSE/WS。
- [ ] `model`/`voice` 使用业务配置的 key，不做别名映射；未知值返回明确错误。
- [ ] 流式音频可连续播放，无爆音/断帧。
- [ ] TTFC p95 达标；长文本不超时。
- [ ] 取消任务后 GPU 资源释放。
- [ ] 对象存储 URL 可访问且 TTL 正确。
- [ ] 监控、日志、追踪可定位到单请求。
- [ ] 模型回滚验证通过。

---

## 11. 风险与演进

| 风险 | 影响 | 缓解 |
|---|---|---|
| Token2Wav 是瓶颈 | 并发上不去 | 分离部署、Triton、TRT、独立扩容 |
| FP16 NaN | 音频异常 | FP32；上线前精度回归 |
| SSML/字级时间戳不完整 | 业务兼容性 | 子集实现 + 明确错误码 |
| 百炼 voice ID 未映射 | 切换后音色不对 | 导入别名表；启动校验 |
| vLLM/TRT 版本冲突 | 启动失败 | 锁版本、镜像固化 |
| 流式 chunk 影响韵律 | 听感变差 | 调 hop/lookahead；短句非流式 |
| 对象存储依赖 | 非流式失败 | 本地缓存 + base64 降级 |
| 长参考音频拖慢首包 | TTFC 超标 | 限制 ≤10s；缓存 prompt feature |

演进：

1. v1：HTTP/SSE/WS 兼容、单说话人、LLM-only 模型、Triton Token2Wav。
2. v1.1：字级时间戳、SSML 子集、音色管理、灰度发布。
3. v2：多说话人、流式变速、AIGC 标识、更多百炼参数。
4. 持续：Token2Wav batching/TRT 优化，按压测数据扩缩容。

---

## 附录 A：业务接入配置

```yaml
tts:
  # 云端
  base_url: https://dashscope.aliyuncs.com
  ws_url: wss://dashscope.aliyuncs.com/api-ws/v1/inference
  api_key: ${DASHSCOPE_API_KEY}
  model: cosyvoice-v3-flash
  voice: longanyang

  # 私有化：只改地址和 key
  # base_url: https://tts.internal.example.com
  # ws_url: wss://tts.internal.example.com/api-ws/v1/inference
  # api_key: ${PRIVATE_TTS_API_KEY}
```

业务代码不应依赖：

- 云端返回的 OSS 域名（用 `output.audio.url` 原样下载）。
- 云端 request_id 格式（只做透传/日志）。
- 云端音色列表（通过配置/接口获取）。

## 附录 B：最小 curl

```bash
curl -X POST https://tts.internal.example.com/api/v1/services/audio/tts/SpeechSynthesizer \
  -H "Authorization: Bearer $PRIVATE_TTS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice-v3-flash",
    "input": {
      "text": "你好，今天想了解点什么？",
      "voice": "longanyang",
      "format": "wav",
      "sample_rate": 24000,
      "instruction": "用自然、亲切的语气说。"
    }
  }'
```

## 附录 C：WebSocket 最小流程

```text
1. 连接 wss://tts.internal.example.com/api-ws/v1/inference
2. 发送 run-task
3. 等待 task-started
4. 发送 continue-task("你好，今天想了解点什么？")
5. 接收 result-generated + binary audio
6. 发送 finish-task
7. 等待 task-finished
8. 复用连接或关闭
```

## 附录 D：实现状态（0.1.0）

已实现：

- 百炼兼容 HTTP 非流式、HTTP SSE、WebSocket 双工；`Authorization: Bearer`。
- 错误码映射、请求校验、文本预处理/分句、字级时间戳占位、mock 与进程内 CosyVoice 引擎。
- 全局/租户并发、RPM、字符速率限制；请求体、WS 消息、WS 连接数上限。
- 优雅摘流/在途 drain、首音频/单句/请求/任务时长超时、WS 取消与队列背压；取消为协作标记 + 队列哨兵 + 有界等待，慢客户端不会挂起 `finish-task`。
- 结构化日志、request_id/tenant/task_id 上下文、Prometheus 指标与直方图。
- 本地音频存储、HMAC 签名 URL、TTL 清理；签名 URL 文件名白名单。
- Docker 非 root/只读根文件系统；Helm chart（Deployment/Service/Ingress/HPA/PDB/NetworkPolicy/ServiceMonitor）。
- `scripts/loadtest.py` 网关级 HTTP/SSE/WS 压测。
- 进程内加速栈接线：`engine.cosyvoice.load_vllm` / `load_trt` / `trt_concurrent` 透传到上游，启动前预检模块、ONNX/plan 与目录可写性（见 8.5.10）；`scripts/gpu_smoke.py` 支持 `--vllm`、`--trt`、`--concurrency/--tasks` 做开关前后对比。默认关闭。

尚未完成，不能据此宣称生产验收通过：

- vLLM / TensorRT 加速栈在真实 GPU 上的吞吐、首包与音质验收（判定标准见 8.5.10）。
- 真实 GPU 上的 CosyVoice 模型加载、精度、RTF、显存与音质回归。
- 百炼官方 SDK 的全量契约回归（尤其边界参数、SSML、错误结构）。
- 远程 LLM/Token2Wav 模型服务适配器；当前只有进程内引擎。
- S3/MinIO 等共享对象存储；`local` 仅适合单副本。
- OpenTelemetry 分布式追踪；指标为进程本地，单 Pod 单 worker。
- 1000 真实并发 soak、同时启动突发、故障注入与跨版本回滚验证。
- `model`/`voice` 热更新、音色管理 API、AIGC 标识。

验收按 [运维手册第 7 节](../ops/cosyvoice-runtime-runbook.md#7-压测与验收流程) 执行。

