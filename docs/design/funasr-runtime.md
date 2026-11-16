# FunASR 运行时方案

## 1. 目标与边界

`funasr-runtime` 是本仓库新增的独立 ASR 服务，对外保持两套接口：

1. **百炼 Fun-ASR Realtime 兼容 WebSocket**：`/api-ws/v1/inference`，客户端 `run-task` / 二进制音频 / `finish-task`，服务端 `task-started` / `result-generated` / `task-finished` / `task-failed`。
2. **OpenAI 兼容离线转写**：`POST /v1/audio/transcriptions`，支持 `json` / `text` / `verbose_json` / `srt` / `vtt`。
3. **浏览器测试台**：`GET /ui`，单文件页面，支持麦克风实时识别、音频文件 WS 回放、离线转写、事件日志与指标；由 `server.ui_enabled` 控制。

目标模型：`Fun-ASR-Nano-2512`、`Paraformer`、`SenseVoiceSmall`、`fsmn-vad`，以及对应的 GGUF 版本。

非目标（首版）：

- 说话人分离、预编译热词 ID、多路混音、SSML/情感标签输出。
- 百炼 8k 模型与任意采样率实时流；实时链路固定 16 kHz。
- 逐 token 的稳定增量解码；partial 是“有界窗口重识别”。

## 2. 架构

```text
client ──WS /api-ws/v1/inference──▶ api/ws.py ──▶ core/session.py
                                                   │
                                                   ├─ core/vad.py (fsmn-vad / energy)
                                                   └─ engines/* (ASREngine)
                                                        ├─ funasr.py   AutoModel (torch)
                                                        ├─ llamacpp.py llama-funasr-* 子进程
                                                        └─ mock.py     测试/联调

client ──HTTP /v1/audio/transcriptions──▶ api/http.py ──▶ engine.transcribe()
```

- `Runtime` 按 `models.<key>.engine` 路由到子引擎，一个部署可同时暴露多个模型（例如实时 Nano + GGUF 离线）。
- 引擎调用在各自的 `ThreadPoolExecutor` 中执行；FunASR `AutoModel` 每个模型一把 `threading.Lock`。事件循环只负责收包、VAD 事件和 JSON 发送。
- 通用能力（认证、指标、限流、优雅退出）在包内实现，仅依赖 `cosyvoice-common.errors` 与输入侧音频工具。

## 3. 实时协议契约

### 3.1 客户端

`run-task`：

```json
{"header":{"action":"run-task","task_id":"<uuid>","streaming":"duplex"},
 "payload":{"task_group":"audio","task":"asr","function":"recognition",
   "model":"fun-asr-realtime",
   "parameters":{"format":"pcm","sample_rate":16000,
     "max_sentence_silence":1300,"heartbeat":false,
     "language_hints":["zh"],
     "speech_noise_threshold":0.0,
     "special_word_filter":"{\"filter_with_signed\":{\"word_list\":[\"项目\"]}}"},
   "input":{"context":[]}}}
```

- `format`：实时只接受 `pcm` / `wav`；WAV 头可跨帧，校验 PCM / 16-bit / mono / 16 kHz。
- `sample_rate` 必须为 16000。
- `input.context` 尽力提取为热词；`special_word_filter` 是结果后处理，不是热词。
- 之后发送二进制音频帧；`finish-task` 表示音频结束：

```json
{"header":{"action":"finish-task","task_id":"<uuid>","streaming":"duplex"},
 "payload":{"input":{}}}
```

- `continue-task` 在 ASR 下返回 `UnsupportedParameter`。

### 3.2 服务端

```text
task-started
  → result-generated (sentence_begin=true, sentence_end=false)   # 首个 partial
  → result-generated (sentence_end=false)                        # 后续 partial
  → result-generated (sentence_end=true, usage.duration)         # 句末 final
  → ...
  → task-finished
```

字段与百炼一致：`begin_time` / `end_time`（任务内绝对毫秒，partial 的 `end_time` 为 `null`）、`text`、`heartbeat`、`sentence_begin`、`sentence_end`、`sentence_id`（从 1 递增）、`words`、`usage.duration`（累计音频秒数，仅句末）。

- 成功任务后连接可复用；`task-failed` 后关闭连接（百炼文档：失败连接不可复用）。
- 心跳：`heartbeat=true` 时无结果期间每 `heartbeat_interval_ms` 返回 `heartbeat=true`、`sentence_id=0` 的事件。
- 错误码：`InvalidParameter` / `UnsupportedParameter` / `UnsupportedFormat` / `AudioTooLong` / `Throttling` / `Timeout` / `ServiceUnavailable` / `InternalError`。
- 关闭码：4401 鉴权失败、1013 连接数上限、1008 客户端协议错误、1011 服务端错误、1000 正常结束。

## 4. 引擎与 VAD

### 4.1 funasr（实时首选）

- 加载：`AutoModel(model=<artifact>, device=..., disable_update=True, hub=...)`；Nano 加 `trust_remote_code=True`。
- VAD：共享 `AutoModel(model=<fsmn-vad>)`，每会话 `DynamicStreamingVAD`（`funasr.models.fsmn_vad_streaming.dynamic_vad`），`silence_schedule=[(inf, max_sentence_silence)]`、`speech_noise_thres=0.5+0.5*request`。
- partial：每 `partial_interval_ms` 对当前句段做一次解码，句长上限 `partial_window_s`；超过窗口后停止发 partial，句末仍用完整段重新解码并返回 final。
- 离线架构（Nano / Paraformer / SenseVoice）没有逐 chunk cache；partial 是重识别。
- 参数透传：`language` / `hotwords` / `hotword` / `use_itn` / `itn` / `sentence_timestamp` 按架构设置，并按 `inspect.signature` 过滤未知项。
- 文本清洗：去掉 `<|...|>` 标签；SenseVoice 的语种/情感/事件标签不进入正文。
- 时间戳：Paraformer 请求 `sentence_timestamp=True` 并映射 `timestamp`；其他模型不伪造，`words=[]`。

### 4.2 llamacpp（CPU/边缘、离线）

- `llama-funasr-cli`（Fun-ASR-Nano，需 encoder + LLM 两个 GGUF）、`llama-funasr-sensevoice`、`llama-funasr-paraformer`。
- 离线：PCM 写临时 WAV → 参数列表调用（`shell=False`）→ 解析 stdout；`verbose_json` 时加 `--srt` 解析分段；临时文件在 `finally` 删除。
- 实时：能量 VAD 分句，句末整段调用；`partial_interval_ms=0` 默认关闭 partial（每次解码都会重启进程并加载模型）。
- 超时/非零退出分别映射 `Timeout` / `InternalError`。

### 4.3 mock

无 torch/numpy 模型依赖，按 `segment_ms` 切句、按 `partial_interval_ms` 生成确定性 partial，用于 CI 与本地联调。

## 5. 配置与模型准备

- 配置：`configs/funasr.yaml`（mock）、`configs/funasr.cpu.yaml`（SenseVoiceSmall + fsmn-vad，纯 CPU）、`configs/funasr.gpu.yaml`（Nano + fsmn-vad）、`configs/funasr.llamacpp.yaml`（GGUF）。
- CPU 部署：`engine.funasr.device=cpu`、`vad_device=cpu`；推荐 SenseVoiceSmall 或 Paraformer，Fun-ASR-Nano 在 CPU 上可用但延迟明显更高。
- CPU 实测（i5-12500H / 16 GB / torch 2.10.0+cpu，SenseVoiceSmall + fsmn-vad）：5.62 s 音频的整段解码 RTF ≈0.053，WS 首 partial 约 0.38 s、`task-finished` 约 1.72 s；短音频单路可实时，长音频与并发按压测结果设置 `max_concurrent_streams`。
- 环境变量前缀 `FUNASR_*`；`FUNASR_RUNTIME_CONFIG` 指定配置文件。
- 模型默认从本地目录加载，`engine.funasr.allow_download=false`；显式下载用：

```bash
python scripts/download_funasr_models.py --models fun-asr-nano,fsmn-vad --dest ./models
```

- 模型目录/文件不打包进镜像，挂载到 `/models`。

## 6. 并发、背压与容量

- `limits.max_concurrent_streams` 控制每租户并发；`max_audio_bytes_per_task`、`max_task_duration_s`、`max_segment_ms` 限制单流资源。
- FunASR 单模型单线程 + 锁：并发流会排队。容量按目标机器的 RTF 与 `partial_interval_ms` 实测确定。
- 实时 partial 的解码窗口上限避免 O(L²)；`max_segment_ms` 强制切分长独白，避免单句无限增长。
- 离线转写与实时共用同一模型/线程池；长音频离线任务会推迟实时请求，生产建议离线/实时分开副本或分模型。

## 7. 观测与安全

- 指标：`funasr_requests_total`、`funasr_errors_total`、`funasr_active_streams`、`funasr_ws_connections`、`funasr_audio_seconds_total`、`funasr_first_partial_seconds`、`funasr_final_latency_seconds`、`funasr_rtf`、`funasr_ready`、GPU 采样指标。
- 认证：`Authorization: Bearer <key>`；`FUNASR_API_KEYS` 或 YAML `auth.api_keys`；`/metrics` 可用独立 `metrics_token`。
- 不记录原始音频或 API Key；GGUF 子进程使用参数列表，不拼 shell。

## 8. 测试与验收

```bash
pytest tests/asr_runtime -q
ruff check packages tests scripts
funasr-runtime -c configs/funasr.yaml
python scripts/asr_ws_smoke.py ws://127.0.0.1:8001 fun-asr-realtime sample.wav

# 真机
python scripts/funasr_smoke.py --model-dir ./models/Fun-ASR-Nano-2512 \
  --vad-dir ./models/fsmn-vad --audio sample.wav --device cuda:0 --stream
```

验收标准：mock 协议测试全绿；真机 smoke 输出首 partial / 句末延迟与 RTF；wav 回放产生连续 `sentence_id`、单调时间戳、累计 usage；同一连接可连续起第二个任务。

## 9. 未决项

- fsmn-vad 不同版本的 `value` 时间轴（绝对/相对）已通过 `DynamicStreamingVAD` 规避；若升级 FunASR 需重新跑 smoke。
- Fun-ASR-Nano 的 `generate` 参数集可能随模型版本变化；`inference_args` 与签名过滤是兼容手段。
- 百炼心跳事件的精确字段/间隔仍需用官方 SDK 对照；当前实现按文档与示例。
- GGUF 实时 partial 的成本需要用目标 CPU 实测后再决定是否开启。
- 超过 `partial_window_s` 的句子不再发 partial，只保证句末最终结果；这是为了避免尾窗文本被客户端误认为整句。
