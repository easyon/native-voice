# 架构设计

## 1. 总览

Native Voice 采用“**薄网关 + 独立模型引擎**”的结构：

```text
┌─────────────────────────────────────────────────────────────────┐
│ 客户端                                                          │
│  浏览器 / 百炼兼容客户端 / 原生 WebSocket 客户端                 │
└───────────────┬─────────────────────────────────────────────────┘
                │ WSS / HTTPS
┌───────────────▼─────────────────────────────────────────────────┐
│ Gateway（FastAPI，CPU）                                         │
│  ├─ 百炼兼容 ASR/TTS WS   /api-ws/v1/inference                  │
│  ├─ 全双工对话 WS         /ws/conversation                      │
│  ├─ REST                  /healthz /readyz /metrics /v1/*       │
│  ├─ 认证 / 配额 / 限流 / 指标                                   │
│  └─ 对话引擎（VAD、分句、打断、播放 ACK）                       │
└──────┬───────────────────────────────┬──────────────────────────┘
       │ 内部 WS                       │ HTTP
┌──────▼──────────────┐        ┌───────▼──────────────────────────┐
│ ASR engine（GPU）   │        │ llama-server（CPU 或 GPU）       │
│ 流式 Paraformer     │        │ Qwen3.5-2B Q8                    │
│ 或 Qwen3-ASR        │        └──────────────────────────────────┘
└─────────────────────┘
┌─────────────────────┐        ┌──────────────────────────────────┐
│ TTS engine（GPU）   │──WS───►│ vLLM-Omni（可选，Linux Docker）  │
│ CosyVoice3 / Qwen3  │        │ 真流式 Qwen3-TTS / CosyVoice3    │
└─────────────────────┘        └──────────────────────────────────┘
```

设计目标：

1. **网关无重依赖**：网关进程不加载 torch，CPU 即可运行，崩溃/重启成本低。
2. **模型进程隔离**：ASR/TTS 各自独立进程，单模型崩溃不会拖垮整个服务。
3. **按需扩缩**：ASR 与 TTS 可以部署在不同机器、不同 venv、不同 GPU 上。
4. **协议兼容**：同一套 engine 同时服务百炼兼容 API、对话 API 和网页客户端。

## 2. 进程与组件

### Gateway

代码：`src/native_voice/api/`、`src/native_voice/conversation/`、`src/native_voice/security/`

职责：

- 终止客户端 WebSocket，完成认证、配额和消息大小限制。
- 把百炼兼容协议转换成 `ASRBackend` / `TTSBackend` 调用。
- 维护全双工对话状态（VAD、分句、打断、历史、播放 ACK）。
- 汇总每轮时延指标并暴露 Prometheus。
- 托管 `web/` 静态页面。

启动时根据 `backends.factory` 选择后端：

| backend | 用途 | 是否加载模型 |
|---|---|---|
| `fake` | 网页/协议/CI 联调 | 否 |
| `local` | 单机开发，进程内直接调用模型 | 是（与网关同进程） |
| `worker` | 生产/开发，通过内部 WS 调用独立 engine | 否 |

### ASR engine

代码：`src/native_voice/engine/asr_server.py`、`src/native_voice/models/asr/`

- 一个进程只加载一个 ASR provider。
- 一个共享模型支持多个并发会话；每个会话有独立的流式 cache，推理串行。
- 对外暴露内部 WebSocket `/asr`，支持 partial、final、cancel。
- provider：`paraformer`（默认，流式）、`qwen3asr`（transformers，滚动重转写近似流式）。

### TTS engine

代码：`src/native_voice/engine/tts_server.py`、`src/native_voice/models/tts/`

- 一个进程只加载一个 TTS provider。
- 句子任务进入优先级队列；同一轮回复的首句优先，之后按顺序串行使用 GPU。
- 取消时静默排空生成器，避免 CosyVoice 内部线程泄漏。
- provider：`cosyvoice3`（默认，本地 GPU）、`qwen3tts`（transformers）、
  `qwen3tts_vllm` / `cosyvoice3_vllm`（代理外部 vLLM-Omni 容器）。

### LLM

代码：`src/native_voice/service/llm_supervisor.py`、`src/native_voice/models/llm/`

- 默认由网关启动 `llama-server`，通过 SSE 流式读取 token。
- `llm.mode=external` 时不启动，直接连接已有服务。
- 过滤 `思考` 标签，按句切分后送入 TTS。

### 对话引擎

代码：`src/native_voice/conversation/`

负责全双工语义：

- 能量 VAD + Silero 语音概率门控。
- 端点检测（句尾静音）、最大语句长度、预卷（pre-roll）。
- ASR partial / final 合并与碎片合并窗口。
- 打断决策：等待有意义的 ASR partial，过滤“嗯/啊/对/好的”等附和。
- 附和（backchannel）生成：独立低音量音轨，不进入 LLM 历史。
- 播放 ACK：客户端确认真正开始播放后，才计入“可听时延”。

## 3. 一次对话的数据流

```text
客户端 16 kHz PCM16
   │
   ▼
[VAD] 连续语音检测 + Silero 门控
   │  partial（每 ~600 ms）
   ├──────────────► ASR engine ──► asr.partial ──► 客户端字幕
   │
   │  句尾静音达到阈值
   ▼
[finalize] ──► asr.final ──► 写入用户轮次
   │
   ▼
[LLM] 流式 token ──► 句子切分 ──► 每句立即提交 TTS
   │        │
   │        └─ 工具调用 ──► 执行工具 ──► 结果回填 ──► 继续生成
   ▼
[TTS engine] 句子队列 ──► 流式 PCM 块 ──► 客户端播放
   │
   ▼
[metrics] endpoint_ms / asr_ms / llm_first_ms / tts_first_ms / total_ms
           + playback_ms / audible_total_ms（客户端 ACK 后）
```

关键点：

- **分块粒度不影响定稿延迟**：VAD 判定说完时立即 flush 不足一个分块的尾音。
- **句级流水线**：LLM 还在生成时，第一句已经开始合成和播放。
- **首句优先**：TTS 队列按轮次优先级调度，缩短首个字到出声的时间。
- **工具调用不中断语音**：纯工具调用的一轮没有文本，网关先播一句
  `llm.tool_filler`（默认“我查一下。”），再执行工具并把结果回填给模型，
  最多 `llm.max_tool_rounds` 轮；工具消息只存在于本轮上下文，不写入历史。

## 4. 全双工与打断

### 打断（barge-in）

助手思考或说话时麦克风仍持续工作：

1. 检测到持续 `barge_sustain_ms`（默认 400 ms）的语音。
2. 等待 ASR partial 出现**有意义内容**；纯附和（“嗯/对/好的”）不打断。
3. 如果 ASR partial 迟迟不出现，用户持续说话超过 `barge_content_wait_ms`
   （默认 900 ms）也兜底打断（GPU 与 TTS 争用时 ASR 会变慢）。
4. 打断时清空播放队列，取消本轮 LLM/TTS，并保留 `preroll_ms`（默认 450 ms）
   的音频，避免丢掉句首。
5. 已经说出的部分回复写入历史，下一轮 LLM 知道“上一轮说到哪”。

`assistant.interrupted` 事件通知客户端立即停播。

### 附和（backchannel）

用户持续说话超过 `backchannel_min_speech_s`（默认 1.5 s）且当前没有 AI 回复时，
服务端用固定短语（“嗯 / 嗯嗯 / 我在听”）合成低音量 PCM，通过独立事件下发。
它不进入 LLM、不写历史、不计入轮次指标；同一句话最多 2 次、间隔 6 s。
不需要时设置 `conversation.backchannel_enabled: false`。

### 回声与噪声

- 播放开始后有 `350 ms` 回声探测窗，期间不触发打断，并估计扬声器回声电平。
- 打断阈值取 `max(噪声底 + 12 dB, barge_db, 回声估计 + 6 dB)`。
- Silero VAD 过滤咳嗽、敲击等短噪声；能量 VAD 负责端点。
- 浏览器端开启 AEC/NS/AGC；外放仍可能误触发，建议耳机。

## 5. 代码分层

```text
src/native_voice/
├── api/                 接入层：HTTP/WS、协议编解码、认证中间件
│   ├── routers/         FastAPI 路由
│   └── protocol/        百炼兼容 ASR/TTS 协议实现
├── conversation/        领域层：全双工会话状态机
├── backends/            适配层：fake / local / worker
├── engine/              运行时：独立 GPU engine 进程
├── models/              模型层：ASR / TTS / LLM 适配器
├── service/             编排层：进程监督、音色存储
├── security/            横切：Key、配额、短票
├── audio/               横切：PCM、重采样、编码
├── observability/       横切：日志、指标
├── config/              横切：配置模型与加载
└── cli/                 命令行入口
```

依赖方向：`api → conversation → backends → models`。横切模块不反向依赖业务模块。
新增一个 ASR/TTS 模型时，只需实现 `backends/base.py` 中的接口并在
`models/*/factory.py` 注册，不需要改协议层。

## 6. 单机 CLI 与服务的关系

`desktop/` 是独立的轻量实现，直接使用 sounddevice 访问声卡，包含自己的
`nv` 包（VAD / ASR / LLM / TTS / 会话）。它不依赖网关，也不走 WebSocket。

两者的模型调优经验共享，但代码不共享：

- 服务模式是主线，适合部署、多人使用和对外提供 API。
- `desktop/` 适合快速验证或离线单机使用；不作为生产部署方式。

如果只需要最简链路，从 `desktop/README.md` 开始；如果要接入客户端或部署服务，
从 [getting-started.md](getting-started.md) 开始。

## 7. 并发与容量

- 单 ASR engine：共享模型、多会话，推理串行。
- 单 TTS engine：句子级串行，队列等待计入指标。
- 8 GB 单卡建议承诺 **2 路**对话；4 路可用但 TTS 会排队。
- 扩容优先增加 TTS engine 或换更大显存；网关本身可无状态水平扩展
  （需要共享 Key 数据库与 `NV_TICKET_SECRET`）。

实测数据与 provider 对比见 [models.md](models.md)。
