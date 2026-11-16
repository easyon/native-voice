# FunASR 运行时运维手册

## 1. 服务表面

| 路径 | 用途 |
|---|---|
| `GET /healthz` / `/livez` | 进程存活 |
| `GET /readyz` | 引擎/模型就绪、draining、inflight |
| `GET /metrics` | Prometheus 文本 |
| `WS /api-ws/v1/inference` | 百炼 Fun-ASR Realtime 兼容 |
| `POST /v1/audio/transcriptions` | OpenAI 兼容离线转写 |
| `GET /ui` | 浏览器测试台（麦克风 / 文件回放 / 离线转写） |

默认端口 `8001`。配置由 `FUNASR_RUNTIME_CONFIG` 指定；生产建议 `configs/funasr.gpu.yaml`。

## 2. 指标与告警

| 指标 | 含义 / 建议 |
|---|---|
| `funasr_ready` | 0 表示模型未加载或正在 draining；持续 0 告警 |
| `funasr_ws_connections` | 当前 WebSocket 连接；接近 `max_ws_connections` 时扩容 |
| `funasr_active_streams` | 活跃识别任务；长期接近 `max_concurrent_streams` 说明排队 |
| `funasr_requests_total{api,status}` | 请求量与成功率；`status=error` 突增告警 |
| `funasr_errors_total{code}` | 按错误码定位：`Throttling` 扩容、`Timeout` 查模型/进程、`ServiceUnavailable` 查加载 |
| `funasr_first_partial_seconds` | 首 partial 延迟；p95 超出目标时降低 `partial_interval_ms` 或扩容 |
| `funasr_final_latency_seconds` | 句末解码延迟；接近静音阈值说明引擎排队 |
| `funasr_rtf` | 每句模型解码耗时 / 音频时长；>1 表示单流跟不上实时 |
| `funasr_last_rtf` | 最近一句的 RTF，浏览器测试台直接读取该 gauge |
| `funasr_audio_seconds_total` | 计费/容量口径 |
| `funasr_gpu_*` | `nvidia-smi` 采样的显存/利用率；共享 GPU 时仅供参考 |

## 3. 常见故障

### 3.1 `/readyz` 一直 not_ready

1. 看日志中的 `loading FunASR model ...` 与异常。
2. 检查模型路径：`engine.funasr.model_root` + `models.<key>.artifact`；默认 `allow_download=false`，缺失会直接失败。
3. GPU 场景确认 `device=cuda:0` 与容器 `nvidia.com/gpu`；`torch.cuda.is_available()` 为 false 时会回退 CPU 并很慢。
4. `funasr` 未安装会报 `funasr is not installed`；安装 `funasr-runtime[funasr]` 或使用 GPU 镜像。

### 3.2 连接后立刻 `task-failed`

| error_code | 排查 |
|---|---|
| `InvalidParameter` | `task_id`、`task_group=audio`、`task=asr`、`function=recognition`、`model` 是否存在 |
| `UnsupportedParameter` | `format` 只能 pcm/wav；`sample_rate` 必须 16000；`continue-task` 不支持 |
| `UnsupportedFormat` | WAV 非 PCM/mono/16-bit，或 WAV 头不完整 |
| `ServiceUnavailable` | 引擎/模型未就绪、连接数上限 |
| `Throttling` | 租户并发/请求/音频配额；检查 `limits` 与 tenant 配置 |

### 3.3 有连接但没有识别结果

- 确认发送的是二进制帧，不是 base64 文本。
- 确认音频有能量且为 16 kHz 单声道；Energy VAD 阈值 `engine.vad.energy_threshold_db` 过低会把静音当语音/过高会漏语音。
- fsmn-vad 场景确认 `vad_artifact` 可加载；加载失败会回退能量 VAD 并打印 exception。
- `partial_interval_ms=0` 时没有 partial，只会在句末/`finish-task` 返回 final。

### 3.4 首 partial 慢 / RTF 高

- 降低 `partial_window_s`（默认 8s）与 `partial_interval_ms`（默认 960ms）的乘积成本。
- 确认没有多路离线任务抢占同一模型线程池；拆分离线副本。
- GPU 显存不足会触发换页/降频；看 `funasr_gpu_memory_utilization_ratio`。
- GGUF 引擎每次解码启动进程；实时建议关闭 partial，或换 funasr 后端。

### 3.5 任务超时

- 单任务上限 `limits.max_task_duration_s`；长连接持续发送音频也会被该上限截断。
- 句末解码超时先看 `funasr_final_latency_seconds` 和模型是否 OOM。
- GGUF 超时看 `engine.llamacpp.timeout_s` 与二进制 stderr。

## 4. 发布与回滚

1. 构建：`docker build -f Dockerfile.asr --target gpu -t funasr-runtime:<tag> .`（CPU/mock 用默认 target）。
2. 模型与二进制不打包；确认 PVC/宿主机路径与 `configs/funasr.*.yaml` 一致。
3. 灰度：先单副本、低 `max_concurrent_streams`，用 `scripts/asr_ws_smoke.py` 与 `scripts/asr_loadtest.py` 验证。
4. 回滚：切回上一镜像 tag 与对应配置；模型目录保持兼容（artifact 路径不变）。

## 5. 容量与调参

| 参数 | 作用 | 调整建议 |
|---|---|---|
| `limits.max_concurrent_streams` | 单进程并发任务 | 按实测 RTF 与首 partial 目标设置；FunASR 单模型串行 |
| `engine.funasr.executor_threads` | 模型并行度 | 默认 1；同一 GPU 上提高通常只会增加排队 |
| `engine.funasr.partial_interval_ms` | partial 频率 | 越大越省算力、首 partial 越慢 |
| `engine.funasr.partial_window_s` | partial 窗口 | 越大长句越完整、成本越高 |
| `limits.max_segment_ms` | 强制切句 | 独白多时降低，避免单句解码过慢 |
| `limits.heartbeat_interval_ms` | 心跳间隔 | 默认 5s；过小增加空转发送 |
| `engine.llamacpp.max_processes` | GGUF 子进程并发 | 按 CPU 核数/内存设置；实时场景保持 1 |
| `engine.funasr.device` | `cpu` / `cuda:0` / `auto` | 纯 CPU 用 `configs/funasr.cpu.yaml`；SenseVoiceSmall 优先，Nano 在 CPU 上更慢 |

CPU 部署时把 `engine.funasr.device` 与 `vad_device` 设为 `cpu`，`executor_threads` 保持 1，
并按 `funasr_rtf` 与首 partial 延迟决定 `max_concurrent_streams`（通常 1–2）。GPU 不可用时
`device: auto` 会自动回退到 CPU，但生产应显式配置 `cpu` 以免静默降级。

CPU 实测参考（i5-12500H，16 GB，Windows，`torch 2.10.0+cpu`，SenseVoiceSmall + fsmn-vad）：

| 场景 | 结果 |
|---|---|
| 模型加载（ASR + VAD） | 1.9 s |
| 5.62 s 音频，离线整段 | 0.30 s 解码，RTF 0.053 |
| 5.62 s 音频，WS 实时 | 首 partial 0.38 s，`task-finished` 1.72 s |
| 5.62 s 音频，HTTP 离线接口 | ~0.64 s |
| 资源 | 单进程 CPU，`max_concurrent_streams` 建议 1–2 |

这是短音频、单路参考值；长音频与并发会线性增加排队，生产容量仍以目标机器压测为准。

## 6. 上线检查

- [ ] 生产开启 `auth.enabled` / `FUNASR_API_KEYS`，`/metrics` 使用独立 token 或内网。
- [ ] 模型/二进制为只读挂载，服务用户无写权限。
- [ ] 容器 `read_only`、`tmpfs /tmp`、`cap_drop ALL`、非 root。
- [ ] Ingress 为 WebSocket 配置足够 idle timeout（> `ws_idle_timeout_s`）。
- [ ] `/readyz` 探针与 `startupProbe` 覆盖模型加载时间（Nano 可能数十秒）。
- [ ] 已用目标音频跑过 `funasr_smoke.py` 与 `asr_loadtest.py`，记录首 partial/句末/RTF。
