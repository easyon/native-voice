# 部署

运行时只安装 `cosyvoice-common` + `cosyvoice-runtime`。训练、数据、评测包不会进入运行时镜像。

## 0. 新 GPU 服务器一键部署（裸机，推荐先走这条）

前提：NVIDIA 驱动 + CUDA 12.x、Python 3.10–3.12、能访问 PyPI 和模型镜像。脚本会依次完成：建 venv → 装 torch/torchaudio 2.10.0+cu128（已存在则跳过）→ 装 runtime 包与 `requirements-cosyvoice.lock` → clone 上游 CosyVoice → 并行下载 checkpoint（跳过 `llm.rl.pt` 等训练/可选文件）→ 生成 GPU 配置 → 跑模型 smoke → `exec` 启动服务。

```bash
git clone <repo> voice-train && cd voice-train

bash deploy/bootstrap-gpu.sh                 # 默认 /data/... + 8000 端口，部署并启动
bash deploy/bootstrap-gpu.sh --no-start       # 只准备环境，不启动（CI/镜像构建用）
VENV=/opt/conda/envs/cosyvoice bash deploy/bootstrap-gpu.sh   # 复用已有 venv 里的 Python + torch，不重装 torch
bash deploy/bootstrap-gpu.sh --reuse-python /opt/conda/bin/python  # 直接复用 conda/系统环境，不建 venv
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
- 完全离线/内网机器可以先用 0.6 打离线包，或走下面的镜像路径。
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
    load_trt: true        # 首次启动编译 flow.decoder.estimator.fp32.mygpu.plan
    trt_concurrent: 2     # TRT context 池大小
```

上线前必须在目标 GPU 上验收（同文本同并发对比开关前后）：

```bash
python scripts/gpu_smoke.py ... --concurrency 4 --tasks 16            # 基线
python scripts/gpu_smoke.py ... --vllm --concurrency 4 --tasks 16
python scripts/gpu_smoke.py ... --trt --trt-concurrent 2 --concurrency 4 --tasks 16
```

通过标准：`aggregate_audio_per_wall_s` 明显高于基线（fp32 c=4 约 1.47× 实时），TTFF 不退化，音频抽检无串音。已知约束：`fp16` 不要和 TRT 叠加；plan 与 GPU 架构绑定（上游文件名固定为 `flow.decoder.estimator.fp32.mygpu.plan`，换卡后从文件名看不出差异，必须删掉重编）；vLLM 会把 `transformers` 升到 4.57.1，回滚时依赖环境也要一起回退。启用/回滚与故障表见 [运维手册 7.3](../docs/ops/cosyvoice-runtime-runbook.md#73-加速栈vllm--tensorrt启用与回滚)。

### L40S 48GB 起手配置

L40S 是 sm89（Ada），支持 bf16/TF32，走 0.2 的主路径（单进程 + vLLM + TensorRT）；不支持 bf16 的卡（sm75 及更老）不进入部署，原因见 0.3。

```bash
bash deploy/bootstrap-gpu.sh --vllm --trt --no-start
# 两个开关同时开启时，生成的配置默认 max_concurrent=4、trt_concurrent=2
```

- 一张卡只跑一个进程、一份权重：vLLM 负责 LLM 连续批处理，TensorRT 多 context 并行 flow 解码。同卡多进程会复制权重、切分 SM，吞吐通常低于 batching。
- 48GB 对这个 0.5B 栈不是瓶颈（见 0.3：整机 RSS 约 3.4GB；vLLM 的 KV cache 也很小）。并发上限由首包/RTF SLO 决定，不由显存决定。`bootstrap` 只做显存下限检查（加速栈 <16 GiB 时告警），不做自动调参。
- 调参顺序：先固定 `trt_concurrent=2`，用 `scripts/loadtest.py` 或 `gpu_smoke.py` 跑 c=2/4/6/8，取“首包 p95 达标且聚合倍率仍在上升”的最大并发；再按 flow 是否成为瓶颈把 `trt_concurrent` 调到 2–4。
- `gpu_smoke.py` 的并发模式会同时输出 `gpu_util_avg_pct` / `gpu_util_max_pct`（nvidia-smi 采样）：**利用率低但请求在排队**说明 GPU 在等串行阶段（提高 `trt_concurrent`、确认 vLLM 已生效，或检查 CPU 预处理）；**利用率接近满载而聚合倍率不再涨**说明算力到顶，再加并发只会抬高首包，应加卡而不是加线程。
- 线上不用手工跑 nvidia-smi：运行时自带 `cosyvoice_gpu_utilization_ratio` 等指标（默认 5s 采样，`metrics.gpu_sampling_interval_s: 0` 可关），判读规则与上面一致；单卡副本也可以拿它做 HPA 的辅助信号（见[运维手册 4](../docs/ops/cosyvoice-runtime-runbook.md#4-容量与扩缩容)）。
- `fp16` 仍按 0.1 的结论默认关闭，先实测；`tf32` 保持开启，Ada 上是免费收益。L40S 的 FP8 上游没有适配，不要开。

## 0.3 不支持 bf16 的卡（sm75，已在选型中排除）

Quadro RTX 6000 24GB（Turing，sm75，无 TF32/bf16）上的对照实测，同一 checkpoint、同一段 3.4s 文本：

| 场景 | RTF | 首包 | 聚合实时倍率 | 显存 |
|---|---|---|---|---|
| 单流（3 次） | 0.83–1.14 | 2.58–3.17s | 1.0× | 3.4 GB |
| c=2，8 任务 | 1.69 | p50 5.7s | 1.16× | 3.4 GB |
| c=4，16 任务 | 4.10 | p50 14.2s | 0.96× | 3.5 GB |

结论：这类卡上进程内并发已经负收益（c=2 只多 16% 吞吐却让首包翻倍，c=4 直接倒退；上游共用一条 CUDA stream，见技术方案 8.5.10），而 vLLM 路径依赖 bf16、在这类卡上不可用。**因此 sm75 及更老不进入部署**，不为其维护第二套多进程形态；扩容只能换 sm80+ 并走 0.2 的主路径。以下数据仅作为选型排除依据保留，不要照搬 A100/L40S 参数。

## 0.4 上游隐式依赖

checkpoint 里的 `cosyvoice3.yaml` 在加载时会解析训练段，所以 `cosyvoice.dataset.processor`（whisper、pyworld）和 Matcha-TTS 的 `matcha.utils`（gdown、matplotlib、wget）即使只做推理也会被 import。`requirements-cosyvoice.lock` 已包含这些包；其中 `pyworld` 需要 `python3.X-dev` + `g++`，`openai-whisper` 的 sdist 需要 `setuptools<81`，`bootstrap-gpu.sh` 会自动处理（缺头文件时 apt 安装、whisper 用 `--no-build-isolation` 单独装）。手工装环境时按 [lock 文件里的注释](../requirements-cosyvoice.lock) 操作。

## 0.5 浏览器测试页（/ui）

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

### 0.3.1 声音克隆

在测试台展开“声音克隆”，上传一段参考音频或直接用麦克风录制，填写这段音频的文字（必须与录音内容一致），即可创建一个 zero-shot 克隆音色。创建后它会出现在 voice 下拉框里，和 `configs/runtime.gpu.yaml` 里配置的音色一样使用。

- 参考音频 1–30 秒、≤16MB，支持 wav/mp3/m4a/aac/ogg/opus/webm/flac；非 WAV 需要 ffmpeg。
- 文字会作为 `prompt_text` 写入，服务端自动补 `<|endofprompt|>` 标记。
- 录音需要浏览器处于 `localhost` 或 HTTPS 环境；纯 HTTP 域名下只能用上传。
- 克隆音色保存在 `cloning.dir`，默认 `./data/voices`，GPU 模板是 `/data/voices`。每个音色一个目录（`prompt.wav` + `meta.json`），服务启动时自动加载。
- 关闭克隆：`cloning.enabled: false`；也可以用 `COSYVOICE_CLONING_DIR` 改存储目录。

接口（同样使用 `Authorization: Bearer`）：

```text
GET    /api/v1/voices                 列出配置音色和克隆音色
POST   /api/v1/voices/clone           multipart: audio, transcript, name, source
DELETE /api/v1/voices/{voice_id}      只允许删除克隆音色
```

新建的克隆音色在本次进程内走每请求提取 prompt 特征（比启动时缓存的音色多约 0.4s 首包）；重启后会和其他 zero-shot 音色一起缓存。

## 0.6 离线包与复用已有环境

前面几节默认部署机可以访问 PyPI 和模型镜像。内网或新机器可以先把模型、上游代码和 wheel 打成离线包，再整目录拷过去；目标机如果已经装好兼容的 CUDA/PyTorch，也可以直接复用，不新建 venv、不重装 torch。

### 打包（有网机器，Linux x86_64）

```bash
# 完整包：模型 + CosyVoice + 全部 wheel + 本仓库副本
bash deploy/prepare-bundle.sh --dest /data/bundles/cosyvoice-20261113

# 需要 vLLM / TensorRT 时把对应 wheel 也带上（TRT 还会自动补 flow ONNX）
bash deploy/prepare-bundle.sh --dest /data/bundles/cosyvoice-20261113 --with-vllm --with-trt

# 等价于上面的完整包（含加速栈），一条命令拉齐全部依赖
bash deploy/prepare-bundle.sh --dest /data/bundles/cosyvoice-20261113 --all

# 复用已经跑过的模型目录（含已生成的 TensorRT plan、vLLM 导出），避免重复下载和首次编译
bash deploy/prepare-bundle.sh --dest /data/bundles/cosyvoice-20261113 --all \
  --model-dir /data/models/Fun-CosyVoice3-0.5B

# 只搬资产（模型 + 上游代码），依赖由目标机自己联网装
bash deploy/prepare-bundle.sh --dest /data/bundles/cosyvoice-20261113 --no-wheels

# 可选：生成压缩包（模型大，gzip 较慢；直接 rsync 目录通常更快）
bash deploy/prepare-bundle.sh --dest /data/bundles/cosyvoice-20261113 --tar
```

包结构：`model/`（checkpoint）、`CosyVoice/`（上游固定在 `requirements-cosyvoice.lock` 记录的 commit，含 Matcha-TTS 子模块）、`wheels/`（目标 Python 的全部发行包）、`repo/`（本仓库副本，目标机不需要 git clone）、`manifest.env`（版本/commit/平台，供 bootstrap 校验）。

批量行为：模型由 `fetch_model.py` 一次拉取全部文件并做并行分片（8 连接），wheel 由四个 lock 文件一次性解析出完整传递依赖，上游仓库连同子模块一次浅克隆；重复执行会跳过已完成的文件和目录，中断后直接重跑即可续传。

约束：

- wheel 与 **Python 小版本**绑定（cp312 的包不能装进 3.11）：打包机与目标机的 Python 小版本必须一致，`bootstrap --bundle` 会在不一致时直接拒绝。跨架构（x86_64 → aarch64）需要在目标架构上重新打包。
- `openai-whisper` / `pyworld` / `antlr4-python3-runtime` 只有 sdist，打包机会在临时 venv 里预先编译成 wheel；成功后目标机不需要 g++ 和 Python 头文件，`bootstrap` 会自动跳过 apt 安装。只有打包机缺少编译工具、回退保留 sdist 时，目标机才需要现场编译。
- TensorRT plan 与 GPU 架构（以及 TensorRT 版本）绑定，首次启动在目标卡上编译；vLLM 的 `<model>/vllm` 导出目录也由首次启动写入，模型目录必须可写（或预生成后只读挂载）。同型号 GPU 批量部署时，用 `--model-dir` 把已经生成 plan 和 vLLM 导出的模型目录直接打进包，后续机器首次启动即可跳过编译和转换。
- 模型文件由 `fetch_model.py` 按上游 sha256 校验；需要整包校验时加 `--checksum`（会遍历所有文件，耗时几分钟）。

### 目标机部署（离线）

```bash
tar -xzf cosyvoice-20261113.tar.gz -C /data/bundles   # 或 rsync -a
tar --zstd -xf cosyvoice-20261113.tar.zst -C /data/bundles  # --tar 且打包机有 zstd 时
bash /data/bundles/cosyvoice-20261113/repo/deploy/bootstrap-gpu.sh \
  --bundle /data/bundles/cosyvoice-20261113 --no-start
```

`--bundle` 会把 `MODEL_DIR` / `COSYVOICE_REPO` 指向包内路径，并设置 `PIP_NO_INDEX=1` + `--find-links wheels`：任何一步需要联网都会失败，而不是静默下载。确实要联网时加 `--allow-network`。

### 部署提速清单

| 手段 | 省掉的开销 |
|---|---|
| 离线包 + `--reuse-python` | 不下载模型/wheel，不建 venv、不重装 torch |
| 打包时预编译 sdist | 目标机不装 g++/Python 头文件，跳过现场编译 |
| `--model-dir` 复用热目录 | 不重复下载模型；TensorRT plan、vLLM 导出随包带走，首次启动不再编译/转换 |
| `--tar`（zstd 多线程） | 归档比 gzip 快数倍；数据量大时直接 `rsync -a` 目录通常更快 |
| 幂等重跑 | 模型按文件跳过、已装依赖 pip 直接满足，失败后重跑只补缺的部分 |
| `--skip-smoke` | 跳过模型冒烟，仅限已验收过的同构机器；首次上线不要跳过 |

### 复用目标机已有的 Python/PyTorch

```bash
# 机器上已有 conda/系统环境，torch 2.10.0+cu128 可用
bash deploy/bootstrap-gpu.sh --reuse-python /opt/conda/bin/python \
  --bundle /data/bundles/cosyvoice-20261113 --no-start
```

- 不创建 venv、不安装或替换 torch；只校验 `torch.cuda.is_available()` 与 CUDA 12.x，然后把运行时和缺失的引擎依赖装进该环境。
- 完全预装好的镜像（依赖已齐全）加 `--no-install`：只做 import 校验，不执行 pip，适合无网、无编译工具的生产镜像。
- 想避免污染共享环境，优先指向**现有 venv** 的 `bin/python`；直接指向系统 Python 时，安装会落在系统环境里。
- 与 `--bundle` 组合时，wheel 的 Python 小版本不一致会直接报错；`--no-install` 不做该检查。
- `--skip-smoke` 可跳过模型冒烟（没有 GPU 的准备阶段或纯配置校验）；正式上线不要跳过。

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

报告里的 `aggregate_rtf`（合成音频秒数 / 墙钟秒数）是跨配置比较吞吐的指标；`audio_bytes` 已按 base64 解码后的原始字节统计。验证多副本或网关分流时用 `--base-urls` 把并发轮流打到各个后端（逗号分隔），再对统一入口跑一遍对比分布。

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
