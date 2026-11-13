# 部署

运行时只安装 `cosyvoice-common` + `cosyvoice-runtime`。训练、数据、评测包不会进入运行时镜像。

## 0. 新 GPU 服务器一键部署（裸机，推荐先走这条）

前提：NVIDIA 驱动 + CUDA 12.x、Python 3.10–3.12、能访问 PyPI 和模型镜像。脚本会依次完成：建 venv → 装 torch/torchaudio 2.10.0+cu128（已存在则跳过）→ 装 runtime 包与 `requirements-cosyvoice.lock` → clone 上游 CosyVoice → 并行下载 checkpoint（跳过 `llm.rl.pt` 等训练/可选文件）→ 生成 GPU 配置 → 跑模型 smoke → `exec` 启动服务。

```bash
git clone <repo> voice-train && cd voice-train

bash deploy/bootstrap-gpu.sh                 # 默认 /data/... + 8000 端口，部署并启动
bash deploy/bootstrap-gpu.sh --no-start       # 只准备环境，不启动（CI/镜像构建用）
VENV=/opt/conda/envs/cosyvoice bash deploy/bootstrap-gpu.sh   # 复用已有 Python 3.12 + torch 2.10.0+cu128 环境，不重装 torch
```

常用覆盖（全部是环境变量）：

```bash
PORT=9000 \
MODEL_DIR=/data/models/Fun-CosyVoice3-0.5B \
COSYVOICE_REPO=/data/CosyVoice \
MODEL_SOURCE=modelscope \
PROMPT_WAV=/data/asset/prompt.wav \
PROMPT_TEXT='You are a helpful assistant.<|endofprompt|>参考音频对应的文本' \
MAX_CONCURRENT=2 \
bash deploy/bootstrap-gpu.sh
```

默认布局：`MODEL_DIR=/data/models/Fun-CosyVoice3-0.5B`、`COSYVOICE_REPO=/data/CosyVoice`、`VENV=$HOME/.venvs/cosyvoice`。已有 CosyVoice clone 就直接把 `COSYVOICE_REPO` 指过去（会复用；缺 Matcha-TTS 子模块时自动补）。GitHub 慢或不通时，先在其他机器 `git clone --recursive` 再上传到 `/data/CosyVoice`，或者用 `COSYVOICE_URL=<镜像地址>` 指定可访问的 clone 源。

几个关键点：

- 模型默认从 **ModelScope** 下（`MODEL_SOURCE=auto`：先 ModelScope，失败再回退 hf-mirror）。ModelScope 仓库 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` 与 HF 目录结构一致，运行时需要的 16 个文件约 5.05 GiB，另有 3 个训练/可选文件被跳过（`llm.rl.pt`、`flow.decoder.estimator.fp32.onnx`、`speech_tokenizer_v3.batch.onnx`）。两者都支持 HTTP Range，脚本按 16 MiB × 8 并发拉取。用 `--trt` 部署时，脚本会自动把 `flow.decoder.estimator.fp32.onnx` 加回下载清单（TensorRT 编译需要它）。
- 实测（同一台机器、同一时段）：hf-mirror 单连接 0.34 MB/s、8 并发 0.68 MB/s；ModelScope 单连接 6.6 MB/s、8 并发 8.4 MB/s。国内网络优先 ModelScope；海外或想用代理时 `MODEL_SOURCE=hf MODEL_BASE=https://hf-mirror.com`。
- `PROMPT_TEXT` 必须与 `PROMPT_WAV` 的内容一致，且包含 `<|endofprompt|>`；不一致会明显掉音色相似度。
- 脚本生成的是 `configs/runtime.local.gpu.yaml`，模板始终是受维护的 `configs/runtime.gpu.yaml`，不要手改生成物。
- 模型下载单独跑也可以：`python scripts/fetch_model.py --dest /data/models/... --list-only` 先看文件清单（默认 ModelScope），去掉 `--list-only` 再下。断点续跑安全：已存在的完整文件跳过，sha256 不匹配会删掉重下。
- 已经装好环境的机器，跳过准备工作直接起服务：`COSYVOICE_PYTHON=/data/venvs/cosyvoice/bin/python COSYVOICE_REPO=/data/CosyVoice . scripts/gpu_env.sh && /data/venvs/cosyvoice/bin/cosyvoice-runtime -c configs/runtime.gpu.yaml`。`gpu_env.sh` 会按 `COSYVOICE_PYTHON`（缺省 `python`）定位 torch 的 nvidia 库拼好 `LD_LIBRARY_PATH`（onnxruntime-gpu 不报 `libcublasLt.so.12` 的前提），并把 `COSYVOICE_REPO` 和 `third_party/Matcha-TTS` 加进 `PYTHONPATH`（否则报 `No module named 'cosyvoice'`）。
- 完全离线/内网机器走下面的镜像路径。
- 国内网络可以把 PyPI 和 torch 都换成国内源（`PIP_INDEX_URL` pip 会自动识别，torch 用 `TORCH_INDEX_URL`）：

  ```bash
  PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  TORCH_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cu128 \
  bash deploy/bootstrap-gpu.sh
  ```

  模型本身默认已经是 ModelScope；仓库 clone 慢时用 `COSYVOICE_URL` 指向可达镜像。下载模型、clone 仓库和装依赖彼此独立，可以并行跑（先 `fetch_model.py` + `git clone` 后台执行，再 `bootstrap-gpu.sh --skip-model`），但不要并发两个 pip 进程写同一个 venv。

## 0.1 单精度还是半精度？

结论：**默认 fp32，`tf32: true`，不要开 `fp16`**。

A100 单卡实测（同一段文本、同一 checkpoint）：

| 配置 | RTF | 显存(RSS) | 说明 |
|---|---|---|---|
| fp32 | 0.747 | 3.4 GB | 默认 |
| fp32 + TF32 | 0.746 | 3.4 GB | 无可测差异；Ampere+ 免费 |
| fp16 | 1.05 | 5.0 GB | 更慢、更占显存 |

原因：上游 CosyVoice3 只有部分子模块是 fp16-safe，开 `fp16` 后其余模块要么被强转回 fp32、要么每层多一次 cast，权重也多存一份。所以半精度在这里既不省显存也不提速。

新卡上仍应实测而不是照抄：`python scripts/gpu_smoke.py --fp16 ...` 与 `--no-tf32` 各跑 3 次取中位数。T4/V100 等没有 TF32 的卡，`tf32: true` 会被忽略，无需改配置；这类卡上若显存吃紧，可以试 fp16，但必须听一下音质并对比 RTF 再决定。

## 0.2 加速栈：vLLM + TensorRT（默认关闭）

单进程里上游把 LLM 全部投到同一个 CUDA stream，线程并发只能到约 1.5× 实时（A100 实测）。要真正提高单卡吞吐，只能做批处理；这里用上游支持的 vLLM 连续 batching 和 TensorRT Flow，不自研调度器（原因见 [技术方案 8.5.10](../docs/design/cosyvoice-runtime.md#8510-加速栈vllm-连续批处理--tensorrt-flow当前最优路径)）。

```bash
bash deploy/bootstrap-gpu.sh --vllm              # 装 vLLM + 开 load_vllm
bash deploy/bootstrap-gpu.sh --trt               # 装 TensorRT + 自动补 fp32 ONNX + 开 load_trt
bash deploy/bootstrap-gpu.sh --vllm --trt        # 两个都开（建议先分别验收）
TRT_CONCURRENT=4 bash deploy/bootstrap-gpu.sh --trt
```

脚本会安装 `requirements-vllm.lock` / `requirements-trt.lock`、检查 `<artifact>` 可写性，并把开关写进生成的配置。对应的运行时字段：

```yaml
engine:
  cosyvoice:
    load_vllm: true       # 首次启动导出 <artifact>/vllm，需可写或预导出
    load_trt: true        # 首次启动编译 flow.decoder.estimator.fp32.<gpu>.plan
    trt_concurrent: 2     # TRT context 池大小
```

上线前必须在目标 GPU 上验收（同文本同并发对比开关前后）：

```bash
python scripts/gpu_smoke.py ... --concurrency 4 --tasks 16            # 基线
python scripts/gpu_smoke.py ... --vllm --concurrency 4 --tasks 16
python scripts/gpu_smoke.py ... --trt --trt-concurrent 2 --concurrency 4 --tasks 16
```

通过标准：`aggregate_audio_per_wall_s` 明显高于基线（fp32 c=4 约 1.47× 实时），TTFF 不退化，音频抽检无串音。已知约束：`fp16` 不要和 TRT 叠加；plan 与 GPU 架构绑定；vLLM 会把 `transformers` 升到 4.57.1，回滚时依赖环境也要一起回退。启用/回滚与故障表见 [运维手册 7.3](../docs/ops/cosyvoice-runtime-runbook.md#73-加速栈vllm--tensorrt启用与回滚)。

## 0.3 非 Ampere 卡实测（Quadro RTX 6000 24GB，2026-11-13）

Turing（sm75，无 TF32/bf16）上同一 checkpoint、同一段 3.4s 文本：

| 场景 | RTF | 首包 | 聚合实时倍率 | 显存 |
|---|---|---|---|---|
| 单流（3 次） | 0.83–1.14 | 2.58–3.17s | 1.0× | 3.4 GB |
| c=2，8 任务 | 1.69 | p50 5.7s | 1.16× | 3.4 GB |
| c=4，16 任务 | 4.10 | p50 14.2s | 0.96× | 3.5 GB |

结论：这类卡上 `max_concurrent` 保持 1。c=2 只多 16% 吞吐却让首包翻倍，c=4 已经负收益（上游共用一条 CUDA stream，见技术方案 8.5.10）。要扩吞吐应多进程/CUDA MPS 或换 Ampere+；vLLM/TensorRT 在 sm75 上是否可用需按 0.2 的验收命令实测，不要直接照搬 A100 参数。

## 0.4 上游隐式依赖

checkpoint 里的 `cosyvoice3.yaml` 在加载时会解析训练段，所以 `cosyvoice.dataset.processor`（whisper、pyworld）和 Matcha-TTS 的 `matcha.utils`（gdown、matplotlib、wget）即使只做推理也会被 import。`requirements-cosyvoice.lock` 已包含这些包；其中 `pyworld` 需要 `python3.X-dev` + `g++`，`openai-whisper` 的 sdist 需要 `setuptools<81`，`bootstrap-gpu.sh` 会自动处理（缺头文件时 apt 安装、whisper 用 `--no-build-isolation` 单独装）。手工装环境时按 [lock 文件里的注释](../requirements-cosyvoice.lock) 操作。

## 0.3 浏览器测试页（/ui）

服务自带一个单文件测试台，部署后直接打开：

```text
http://<host>:8000/ui
```

页面会从 `/readyz` 读取当前的 model/voice key，支持同步 HTTP、SSE、WebSocket 三种调用方式，并显示首包 TTFF、总耗时、音频大小和事件日志；还能一键复制对应的 curl。页面是静态文件，不依赖 CDN、Node 或构建步骤，内网/离线机器也能用。

两个注意点：

- 页面不会被鉴权拦截，但调用 API 时会带上你在页面里填的 Bearer key。开启 `auth.enabled` 后，同步/SSE 模式可用；浏览器的 WebSocket API 不能自定义 `Authorization` 头，WS 模式会以 4401 关闭，只能在内网验收时临时关闭鉴权或改用同步/SSE。
- 反代/端口映射要同时放行 `/ui`、`/api/v1`、`/api-ws`、`/healthz`、`/readyz`，SSE 和 WS 仍需关闭缓冲并放大超时（Helm values 里已配置）。

不需要时在配置里关闭：

```yaml
server:
  ui_enabled: false
```

## 1. 构建镜像

CPU 网关镜像（mock 引擎或远程模型服务）：

```bash
docker build -t cosyvoice-runtime:0.1.0 .
```

进程内 CosyVoice GPU 镜像：

```bash
docker build \
  --target gpu \
  -t cosyvoice-runtime:0.1.0-gpu .
```

GPU 镜像基于 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`，torch 2.10.0+cu128 来自 conda，Python 依赖由 `requirements-runtime.lock` / `requirements-cosyvoice.lock` 锁定。镜像不包含模型权重，也不包含 CosyVoice 上游代码；两者都在运行时只读挂载。构建、启动、环境变量与已知坑见 [docs/ops/cosyvoice-runtime-image.md](../docs/ops/cosyvoice-runtime-image.md)。

GPU 镜像还需要宿主机/集群提供 CUDA 与 NVIDIA Container Toolkit；模型文件通过只读卷或对象存储挂载，不要打进镜像。

## 2. 直接运行

```bash
docker run --rm -p 8000:8000 \
  -e COSYVOICE_API_KEYS='replace-with-a-long-random-key' \
  -e COSYVOICE_LOG_FORMAT=json \
  -v "$PWD/data/audio:/app/data/audio" \
  cosyvoice-runtime:0.1.0
```

生产不要使用 `environment: dev`（`configs/runtime.yaml` 默认是 `dev`，仅为本地联调）。Helm chart 会把 `COSYVOICE_ENVIRONMENT` 设为 `prod`。

## 3. Helm

```bash
helm upgrade --install cosyvoice deploy/helm/cosyvoice-runtime \
  --namespace cosyvoice --create-namespace \
  --set runtime.auth.apiKeys='replace-with-a-long-random-key' \
  --set runtime.auth.metricsToken='replace-with-a-metrics-token' \
  --set image.repository=registry.example.com/cosyvoice-runtime \
  --set image.tag=0.1.0
```

更推荐把 Secret 放在 values 之外：

```bash
kubectl -n cosyvoice create secret generic cosyvoice-runtime-auth \
  --from-literal=api-keys='replace-with-a-long-random-key' \
  --from-literal=metrics-token='replace-with-a-metrics-token' \
  --from-literal=storage-signing-secret='replace-with-a-signing-secret'

helm upgrade --install cosyvoice deploy/helm/cosyvoice-runtime \
  --namespace cosyvoice --create-namespace \
  --set runtime.auth.existingSecret=cosyvoice-runtime-auth
```

`runtime.auth.apiKeys` 或 `existingSecret` 必须配置；没有 key 时 chart 会拒绝渲染，避免误部署一个无鉴权服务。

关键 values：

| value | 说明 |
|---|---|
| `replicaCount` / `autoscaling` | 网关副本数；连接型负载建议结合 `cosyvoice_active_streams` 自定义指标扩容 |
| `runtime.engine.type` | `mock` 用于协议/压测；`cosyvoice` 为进程内模型 |
| `gpu.enabled` | 仅进程内引擎时设置，向 Pod 申请 `nvidia.com/gpu` |
| `runtime.models` / `runtime.voices` | 直接使用业务 model key / voice ID，不做百炼别名映射 |
| `runtime.limits.maxConcurrentStreams` | 单 Pod 并发流上限，必须与压测结果一致 |
| `runtime.storage.type` | 多副本建议 `none`（返回 base64）；`local` 只适合单副本或每副本独立 PVC |
| `persistence.enabled` | `local` 存储时挂载可写卷；`none` 时也可关闭 |
| `serviceMonitor.enabled` | 集群有 Prometheus Operator 时启用 |
| `ingress.annotations` | 已包含 WebSocket/SSE 需要的长连接、关闭缓冲配置 |

## 4. 压测

先对 mock 引擎验证网关、鉴权、背压与连接生命周期：

```bash
python scripts/loadtest.py --mode ws  --concurrency 100 --tasks 5000 --api-key test-key
python scripts/loadtest.py --mode sse --concurrency 100 --tasks 5000 --api-key test-key
python scripts/loadtest.py --mode http --concurrency 100 --tasks 5000 --api-key test-key
```

再对真实模型服务重复同样的阶梯，并记录 TTFF、RTF、错误率、GPU 显存、队列等待。`scripts/loadtest.py` 只验证接入层，不代表真实模型吞吐。

建议阶梯：1 → 10 → 25 → 50 → 100 → 200（连接/流），每次至少 5 分钟，观察 p95/p99、`cosyvoice_queue_wait_seconds`、`cosyvoice_active_streams`、`cosyvoice_rtf`。1000 并发先做 1000 连接的空载/短文本 soak，再按容量模型拆分为多个模型服务副本。

## 5. 生产上线检查

- [ ] `auth.enabled` 生效，使用长随机 key；`/metrics` 使用独立 token。
- [ ] `environment=prod`；`storage.signing_secret` 已注入（若启用本地文件 URL）。
- [ ] 使用 JSON 日志并接入日志平台；request_id / tenant / task_id 可检索。
- [ ] `/healthz`、`/readyz`、`/metrics` 已接入探针与监控。
- [ ] `terminationGracePeriodSeconds` > `limits.shutdown_timeout_s`，且 preStop 足够摘流量。
- [ ] 每 Pod 只跑一个 uvicorn worker；指标与进程内引擎状态都是进程本地的。
- [ ] 用真实模型完成 100/1000 并发压测，确认 TTFF、RTF、显存与错误率。
- [ ] 确认流式格式：PCM/WAV/MP3；流式 `opus` 会被拒绝（独立 Ogg 块不能安全拼接）。
- [ ] 多副本时不要依赖节点本地 `local` 存储；使用 `none` 或共享对象存储。
- [ ] 完成百炼官方 SDK 的 HTTP/SSE/WebSocket 兼容性回归。

## 6. 当前边界

进程内 CosyVoice 引擎适合单机/小规模与联调。100/1000 并发的推荐拓扑仍是 CPU 网关 + 独立 LLM/Token2Wav 模型服务；本仓库目前提供进程内引擎适配器，远程模型服务适配器需要按 `TTSEngine` 接口补充。上线前必须以真实 GPU、真实模型和真实文本分布完成压测与验收。
