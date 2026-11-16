# TTS 运行时运维手册

适用版本：`tts-runtime 0.1.0`。本文只描述已经实现的运行时行为；真实模型容量必须以本环境压测为准。

## 1. 服务面

| 路径 | 用途 | 鉴权 |
|---|---|---|
| `POST /api/v1/services/audio/tts/SpeechSynthesizer` | 百炼兼容 HTTP / SSE | Bearer |
| `WS /api-ws/v1/inference` | 百炼兼容 WebSocket 双工 | Bearer（握手前校验） |
| `GET /healthz` | 进程存活，不检查模型 | 无 |
| `GET /readyz` | 是否可接流量：未 drain、引擎 ready | 无 |
| `GET /metrics` | Prometheus 文本 | 独立 token 或 `metrics_public=true` |
| `GET /files/{name}` | 本地音频，带 `exp`/`sig` 签名 | 签名 URL |

## 2. 关键指标与告警

| 指标 | 含义 | 建议 |
|---|---|---|
| `cosyvoice_ready` | 1 表示可接流量 | `== 0` 持续 1m 告警 |
| `cosyvoice_active_streams` | 正在合成的流 | 接近 `limits.max_concurrent_streams` 时扩容/限流 |
| `cosyvoice_queue_wait_seconds` | 等待全局/租户槽位 | p95 持续升高说明容量不足 |
| `cosyvoice_first_audio_seconds` | 首音频延迟 | p95 超过业务 SLO 告警 |
| `cosyvoice_request_duration_seconds` | 端到端请求耗时 | 分 api/status 看 |
| `cosyvoice_sentence_duration_seconds` | 单句耗时 | 用于定位模型退化 |
| `cosyvoice_rtf` | 实时率 | 持续 > 0.7 时并发余量不足 |
| `cosyvoice_errors_total` | 按错误码 | `Throttling` 高是容量/配额问题；`InternalError` 高看日志 |
| `cosyvoice_ws_connections` | WS 连接数 | 接近 `max_ws_connections` 告警 |
| `cosyvoice_gpu_utilization_ratio` | GPU SM 利用率（0–1，nvidia-smi 采样） | 持续 > 0.85 说明单卡算力到顶：加副本或降并发，不要继续加线程 |
| `cosyvoice_gpu_memory_utilization_ratio` | 显存占用率（used/total，0–1） | 接近 1 且出现 OOM 时降 `max_concurrent` 或 `trt_concurrent` |
| `cosyvoice_gpu_memory_used_bytes` | 显存已用字节（设备级） | 用于容量记录与缓慢增长排查 |

GPU 三项由运行时内的采样线程每 5s 跑一次 `nvidia-smi` 发布，默认开启（`metrics.gpu_sampling_interval_s: 5`、
`metrics.gpu_device: 0`）；设为 `0` 可完全关闭。没有驱动/nvidia-smi 时这些序列不存在，不影响其它指标。
它们是**整卡**数值：部署模型是一个运行时进程独占一张卡，因此可直接作为副本级信号；同卡多进程不在架构内，不要据此做池化调优。

初始 SLO 建议（需压测确认）：首音频 p95 ≤ 500 ms、错误率 < 0.5%、RTF p95 < 0.7、无未处理 5xx。

## 3. 发布与回滚

1. 新副本先通过 startupProbe（模型加载可能很慢）和 readinessProbe。
2. `preStop` 等待 10s 摘流量，`shutdown_timeout_s` 内完成在途请求，`terminationGracePeriodSeconds` 大于两者之和。
3. 使用 `maxUnavailable: 0, maxSurge: 1` 滚动；WS 连接会在旧 Pod 终止时断开，客户端应自动重连并按 `task_id` 重建任务。
4. 回滚：`helm rollback <release> <revision>`，然后确认 `/readyz` 与首音频指标恢复。

## 4. 容量与扩缩容

稳态近似：`N_safe = 0.7 / RTF` 条流/GPU。首块突发按 `ceil(S / 2.8)` 张 L20 估算；这些是设计模型，不是本机实测值。

- 网关 CPU 副本按连接数、编码线程和错误率扩容；不要只看 CPU。
- 真实模型服务按 RTF、显存、首块延迟扩容。
- `limits.max_concurrent_streams` 必须小于压测得到的安全并发；否则队列等待会转化为超时。
- 单 Pod 只运行一个 uvicorn worker；多副本通过 Service/Ingress 负载均衡。
- 单卡 Pod 可把 `cosyvoice_gpu_utilization_ratio` 接进 HPA 作为辅助信号（目标 0.8，见下方示例）；
  主信号仍是 `cosyvoice_active_streams` 与 `cosyvoice_queue_wait_seconds`——GPU 利用率要等请求真正跑起来才升高，
  而排队延迟在它之前就恶化了。只按利用率扩容会让首包 SLO 在扩容前先破一次。

```yaml
# values.yaml：需要一个 Prometheus adapter 规则把该指标映射为 Pods 指标
autoscaling:
  metrics:
    - type: Pods
      pods:
        metric:
          name: cosyvoice_gpu_utilization_ratio
        target:
          type: AverageValue
          averageValue: "800m"      # 0.8；与 max_concurrent 一起压测后再定
```

让 HPA 生效的前提：每个副本声明 `nvidia.com/gpu: 1` 且独占一张卡（否则整卡利用率会混入其它 Pod 的负载），
并且 `minReplicas` ≥ 2，避免单副本抖动时被缩到 0。

## 5. 故障处理

### 401 Unauthorized
检查 `Authorization: Bearer`，以及 `COSYVOICE_API_KEYS`/Secret 是否注入。密钥比较是常量时间；不要在日志打印密钥。

### 429 Throttling
租户并发、RPM、字符速率或全局槽位达到上限。看 `cosyvoice_queue_wait_seconds` 与租户配置；先扩容，再考虑提高配额。

### 503 ServiceUnavailable
常见原因：drain 中、全局并发槽位等待超时、WS 连接数达到上限、引擎资源耗尽。看 `/readyz`、Pod 事件和 `cosyvoice_errors_total`。

### 504 / 请求超时
`request_timeout_s` 与 `limits.max_request_s` 取较小值生效。检查引擎 RTF、队列等待、文本长度和 `sentence_timeout_s`。

### WebSocket 1013
连接数达到 `max_ws_connections`。扩容或降低单连接并发；不要靠无限提高上限。

### WebSocket 4401
握手鉴权失败，连接未被接受。

### 签名 URL 403
`exp` 过期、`sig` 不匹配或文件名不在允许列表。检查 `storage.signing_secret` 是否在滚动发布中保持一致。

### 首音频超时
`limits.first_audio_timeout_s` 内没有音频块。检查模型冷启动、GPU OOM、队列等待和文本是否被分句器阻塞。

### 进程退出/被杀
确认 `terminationGracePeriodSeconds` > `shutdown_timeout_s`；查看是否有引擎流未在超时内结束。`cancel_grace_s` 是尽力取消，长生成可能超过该时间。

## 6. 安全检查

- 生产必须启用鉴权；`/metrics` 使用独立 token。
- 只通过 TLS Ingress 暴露；WS 使用 `wss://`。
- 容器以非 root、只读根文件系统、drop ALL capabilities 运行；本地音频目录单独挂载。
- `storage.type=local` 只适合单副本；URL 必须签名，且 TTL 到期由清理任务删除。
- `docs_enabled` 默认关闭；生产保持关闭。
- 不要把 API key、签名密钥、模型访问凭据写进 ConfigMap 或镜像。

## 7. 压测与验收流程

1. Mock 引擎验证协议、鉴权、限流、断连、取消、100/1000 连接。
2. 真实 GPU 单副本：1/10/25/50 流，记录 TTFF、RTF、显存。
3. 多副本：按容量模型扩展到 100/1000 流，做同时启动突发。
4. 长稳：至少 1 小时，含取消、重连、长文本、慢客户端。
5. 故障注入：杀 Pod、断网、引擎 OOM、存储不可写，确认降级与恢复。
6. 用百炼官方 SDK 做 HTTP/SSE/WebSocket 回归。

官方 SDK 回归（mock 引擎即可，不需要 GPU 或真实 API Key）：

```powershell
# Windows PowerShell（仓库根目录）
.\.venv\Scripts\python.exe -m pip install dashscope
.\.venv\Scripts\python.exe -m pytest tests/tts_runtime/test_dashscope_compat.py -vv
```

```bash
# Linux / macOS
.venv/bin/python -m pip install dashscope
.venv/bin/python -m pytest tests/tts_runtime/test_dashscope_compat.py -vv
```

测试会在随机空闲端口启动 runtime，等 `/readyz` 返回 200 后依次用官方 SDK 打通
`dashscope.audio.http_tts`（HTTP 非流式、HTTP SSE）、`dashscope.audio.tts_v2`
（WebSocket duplex）和 `dashscope.audio.tts`（旧版 WebSocket `streaming=out`）。
测试自动清理 `HTTP_PROXY`/`NO_PROXY` 等代理变量；业务迁移时只需把
`dashscope.base_http_api_url` 指向 `http://<host>:<port>/api/v1`、
`dashscope.base_websocket_api_url` 指向 `ws://<host>:<port>/api-ws/v1/inference`。

单卡验收的可执行步骤：

```bash
# 1) 脱离网关，先确认模型/CUDA/前端链路本身正常
COSYVOICE_REPO=/data/cvtest python scripts/gpu_smoke.py \
  --model-dir /data/models/Fun-CosyVoice3-0.5B \
  --prompt-wav /data/cvtest/asset/zero_shot_prompt.wav \
  --prompt-text 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。' \
  --out /tmp/smoke.wav --runs 2

# 2) 起真实模型网关
tts-runtime -c configs/runtime.gpu.yaml

# 3) 三条接入路径各打一次
python /path/to/api_smoke.py http://127.0.0.1:8000 cosyvoice3-0.5b assistant

# 3b) 浏览器验收：打开 http://<host>:8000/ui，选择三种模式各合成一次。
#     页面会显示 TTFF/总耗时/事件日志；开启鉴权时 WS 模式不可用（见 deploy/README 0.5）。

# 4) 阶梯并发：TTFF / RTF / 显存
#    先按 max_concurrent 的 1/2/4 打；超过 max_concurrent 只会排队并超时，
#    不会提高吞吐（见 docs/design/tts-runtime.md 8.5.9 实测）。
for c in 1 2 4; do
  python scripts/loadtest.py --mode ws --concurrency $c --tasks $((c * 4)) \
    --base-url http://127.0.0.1:8000 --model cosyvoice3-0.5b --voice assistant
done
```

CosyVoice3 的零样本 `prompt_text` 必须包含 `<|endofprompt|>`，否则上游 LLM 会断言失败；
模型目录需要 `cosyvoice3.yaml`、`llm.pt`、`flow.pt`、`hift.pt`、`campplus.onnx`、
`speech_tokenizer_v3.onnx` 和 `CosyVoice-BlankEN/` 子目录。

### 7.1 单卡验收记录（A100-40G，2026-11-12）

环境：A100-40G、CUDA 12.8、torch 2.10.0+cu128、onnxruntime-gpu 1.20.2、fp32、
`max_concurrent=4`、零样本音色（prompt ~5s）。

| 项 | 结果 |
|---|---|
| 进程启动到 `/readyz=200` | ~25–80s（含模型加载与 warmup；首次编译更慢） |
| 模型加载 | 16–24s，峰值显存 3.4 GB |
| HTTP 非流式 | 200，3.0s，245760 B |
| SSE 流式 | 7 个事件，TTFF 2.93s，249600 B |
| WebSocket 流式 | `task-started`/`result-generated`×6/`task-finished`，TTFF 2.24s，249600 B |
| 并发 1/2/4 | TTFF p50 1993 / 3683 / 8363 ms；聚合实时倍率 1.17× / 1.56× / 1.47× |
| 并发 8（> `max_concurrent`） | 32 个请求全部超时，属预期排队行为 |

部署时必须注意：

- `LD_LIBRARY_PATH` 需要包含 torch 自带的 `nvidia/*/lib`，否则 onnxruntime-gpu 报
  `libcublasLt.so.13`/`libcudnn` 缺失；onnxruntime-gpu 需锁 1.20.x（CUDA 12）。
- torchaudio ≥ 2.9 走 TorchCodec，镜像里未必有匹配的 FFmpeg；运行时已用
  `engines/cosyvoice_compat.py` 的 soundfile 垫片规避，不要删。
- fp16 在 A100 上实测 RTF 1.05（fp32 0.70），默认关闭。
- 零样本 prompt 特征会在模型加载时用 `add_zero_shot_spk` 预热（`cache_prompt_features`，
  默认开）；prompt 文件缺失或加载失败只告警，自动回退到逐请求提取。
- `wetext` 文本正则（中文数字/日期归一）依赖 ModelScope 资产；内网或受限网络下
  首次下载可能 403，此时前端会打印 `no frontend is avaliable` 并跳过归一化。
  部署前应在镜像/模型卷里预置该资产，详见 7.2。

### 7.2 文本正则资产（wetext）预置

CosyVoice 的中文数字/日期/金额归一化依赖 `wetext`，后者首次使用时通过 ModelScope
下载 `pengzhendong/wetext`（Kaldifst 文法，约 120 MB，`en/tn`、`zh/tn` 必需）。
实测在没有 ModelScope token 的网络上，该仓库会稳定返回 403（错误体却提示“操作过于频繁”），
CosyVoice 随后打印 `no frontend is avaliable` 并**静默跳过归一化**（“138元”不会被读成
“一百三十八元”），属于音质/可懂度回归，发布前必须确认。

预置方式（推荐）：

1. 在可访问 ModelScope 的机器上下载完整仓库，或从已缓存环境拷贝
   `.../modelscope/hub/pengzhendong/wetext`（`en/tn`、`zh/tn` 各含 `tagger.fst`、
   `verbalizer.fst`）。
2. 放进镜像或模型卷，例如 `/models/wetext`。
3. 启动时设置 `COSYVOICE_WETEXT_DIR=/models/wetext`；运行时会把这个目录直接交给
   wetext，不再联网（见 `engines/cosyvoice_compat.py::install_wetext_assets`）。
4. 验证：启动日志应出现 `use wetext frontend`，且不再出现 `no frontend is avaliable`；
   再用“今天气温23.5度，花了138元”合成一条，确认数字被读成中文。

> 不要用 curl 往缓存目录写失败响应：403 错误页只有 137 字节，会覆盖文法文件，
> kaldifst 报 `Magic number not matched`。文法文件应是 MB 级。

备选方案是上游的 `ttsfrd` 前端（`pip install ttsfrd`），但它要求
`pretrained_models/CosyVoice-ttsfrd/resource` 资源目录，只有完整 CosyVoice 仓库
才带；使用前需单独准备该目录。

### 7.3 加速栈（vLLM / TensorRT）启用与回滚

设计动机见 [技术方案 8.5.10](../design/tts-runtime.md#8510-加速栈vllm-连续批处理--tensorrt-flow当前最优路径)。
两个开关互相独立，**默认关闭**；上线前必须按本节在目标 GPU 上验收，不能只凭安装成功就切流量。

安装（二选一，或两者都装）：

```bash
bash deploy/bootstrap-gpu.sh --no-start --vllm      # 安装 vLLM 并生成配置
bash deploy/bootstrap-gpu.sh --no-start --trt       # 安装 TensorRT 并下载 fp32 ONNX
bash deploy/bootstrap-gpu.sh --no-start --vllm --trt
```

内网机器或已有兼容 CUDA/PyTorch 的环境，可加 `--bundle`（`deploy/prepare-bundle.sh` 生成的离线包）和/或
`--reuse-python /path/to/python`，不新建 venv、不下载 wheel；完全预装好的镜像加 `--no-install` 只做校验。
同型号 GPU 批量部署时，用 `prepare-bundle.sh --model-dir <已跑过的模型目录>` 把已生成的 TensorRT plan 和
vLLM 导出直接打进包，新机器首次启动可跳过编译和转换（plan 与 GPU 型号、TensorRT 版本绑定）。
见 [deploy/README 0.6](../../deploy/README.md#06-离线包与复用已有环境)。

手工等价步骤：

```bash
pip install -r requirements-vllm.lock        # vllm==0.19.1, 锁 torch 2.10.0
pip install -r requirements-trt.lock         # tensorrt-cu12==10.13.3.9
# ONNX 不在默认下载集合里，需显式包含：
python scripts/fetch_model.py --repo FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --dest /data/models/Fun-CosyVoice3-0.5B \
  --include flow.decoder.estimator.fp32.onnx
```

启用（`configs/runtime.local.gpu.yaml`）：

```yaml
engine:
  cosyvoice:
    fp16: false
    load_vllm: true
    load_trt: true
    trt_concurrent: 2
```

首次启动的产物与可写性：

| 开关 | 产物 | 要求 |
|---|---|---|
| `load_vllm` | `<artifact>/vllm`（LLM 导出；导出后进程内 HF LLM 层被卸载） | 目录可写，或预导出后只读挂载 |
| `load_trt` | `<artifact>/flow.decoder.estimator.fp32.mygpu.plan` | 目录可写，或预编译 plan；plan 与 GPU 架构/MIG slice 绑定（文件名固定为 `mygpu`，换卡后**无法从名字**判断是否过期，必须删掉重编或看加载报错） |

验收（开关前后各跑一遍，同 GPU、同音色、同文本）：

```bash
# 基线
python scripts/gpu_smoke.py --model-dir /data/models/Fun-CosyVoice3-0.5B \
  --prompt-wav ... --prompt-text ... --text '你好，我是你的语音助手，很高兴为你服务。' \
  --concurrency 4 --tasks 16

# vLLM（连续 batching）
python scripts/gpu_smoke.py ... --vllm --concurrency 4 --tasks 16

# TensorRT（首次运行会编译 plan，可能数分钟）
python scripts/gpu_smoke.py ... --trt --trt-concurrent 2 --concurrency 4 --tasks 16
```

通过标准：`aggregate_audio_per_wall_s` 显著高于基线（fp32 基线 c=4 约 1.47× 实时），
TTFF 不劣于基线，音频抽检无串音/异常韵律；单请求 RTF 允许小幅波动。
结果 JSON 里带 `vllm`/`trt` 字段，可归档对比。

常见启动失败：

| 现象 | 原因与处理 |
|---|---|
| 503 `load_vllm=true, but vllm is not installed` | 当前虚拟环境没装 vLLM，或 `COSYVOICE_PYTHON` 指向了别的解释器 |
| `import vllm` 报 CUDA/driver 错误 | vLLM wheel 的 CUDA 小版本高于驱动；升级驱动或换匹配 wheel，不要降级 torch |
| 503 `load_vllm exports ... not writable` | artifact 只读；预导出 `<artifact>/vllm` 后改为只读挂载，或临时放开写权限 |
| 503 `load_trt=true needs flow.decoder.estimator.fp32.onnx ...` | 缺 ONNX 且无 plan；用上面的 `--include` 重新拉取 |
| TensorRT 反序列化/显存错误 | plan 来自其他 GPU 架构或换了 MIG 配置；删除 `*.plan` 重新编译，不要跨卡拷贝 |
| 首包/吞吐反而变差 | 单独回退对应开关，保留另一个；两者可独立启用 |

回滚：把 `load_vllm`/`load_trt` 改回 `false` 并重启。`<artifact>/vllm` 可保留（不会影响非
vLLM 路径加载，但会占磁盘）；plan 文件同样可保留，便于下次直接复用。
vLLM 升级了 `transformers`（4.51.3 → 4.57.1），**回滚服务版本时要同时回滚依赖环境**，
否则非 vLLM 路径仍在用被升级的文本前端。

## 8. 已知边界

- 进程内引擎是单节点方案；推荐 CPU 网关 + 独立 LLM/Token2Wav 服务。
- 指标和租户限流是进程本地的；通过多副本而不是多 worker 扩展。
- 流式 `opus` 已拒绝；流式 `pcm`/`wav`/`mp3` 可用。WAV 只在首块发一次流式头。
- SSE 落盘会缓存整段 PCM，超过 `max_stream_audio_bytes` 后停止缓存但仍继续流式返回。
- 本仓库没有远程模型服务适配器、S3/MinIO 存储和 OpenTelemetry；这些属于下一步生产化工作。
