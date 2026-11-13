# CosyVoice 运行时 GPU 镜像

适用对象：`packages/runtime` 的进程内 CosyVoice3 引擎（`engine.type=cosyvoice`）。本文覆盖镜像定义、依赖锁定、构建/启动、模型挂载、健康检查、已知坑，以及 Docker Hub 不可达时的替代路径。

镜像定义在仓库根目录 [Dockerfile](../../Dockerfile)，不是另一套独立镜像；CPU 网关和 GPU 推理共用同一个多阶段文件。

## 1. 镜像拓扑

| target | 基础镜像 | 内容 | 用途 |
|---|---|---|---|
| `cpu`（默认，最后一个 stage） | `python:3.12-slim` | `cosyvoice-common` + `cosyvoice-runtime`、ffmpeg | mock 引擎或远程模型服务网关；`docker build .` 和 CI 继续可用 |
| `gpu` | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` | conda torch 2.10.0+cu128、Python 3.12 venv、锁定后的推理依赖 | 单卡进程内 CosyVoice3，一张 GPU 一个 Pod |

GPU 镜像分三段构建：

1. `wheelbuilder`：只构建 `packages/common`、`packages/runtime` 两个 wheel，保证 CPU/GPU 镜像安装同一份代码。
2. `gpu-torch`：用 micromamba 创建 `python=3.12 + pytorch=2.10.0 + torchaudio=2.10.0 + pytorch-cuda=12.8` 的 conda 环境，再创建 `--system-site-packages` 的 venv；torch 只来自 conda，pip 只装锁定后的运行库。构建期会断言 torch/CUDA/torchaudio/onnxruntime/transformers 版本，并用 `ctypes` 实际加载 `libcudart.so.12`、`libcublas.so.12`、`libcublasLt.so.12`、`libcufft.so.11`、`libcurand.so.10`、`libcudnn.so.9`。
3. `gpu`：只复制 conda 环境、venv、配置和 entrypoint，不复制模型、不复制上游 CosyVoice 代码。

相关文件：

| 文件 | 作用 |
|---|---|
| [Dockerfile](../../Dockerfile) | 多阶段 CPU/GPU 镜像；默认 target 为 CPU |
| [requirements-runtime.lock](../../requirements-runtime.lock) | 网关依赖，版本来自 2026-11-12 的仓库 dev venv `pip freeze` |
| [requirements-cosyvoice.lock](../../requirements-cosyvoice.lock) | 推理引擎依赖；已验证项写死，未记录项标 TODO |
| [deploy/docker/conda-torch.yaml](../docker/conda-torch.yaml) | conda torch 环境规格 |
| [deploy/docker/entrypoint.sh](../docker/entrypoint.sh) | source GPU 环境、拼 `PYTHONPATH`、exec 进程 |
| [scripts/gpu_env.sh](../../scripts/gpu_env.sh) | 把 torch/conda 的 CUDA/cuDNN 目录导出到 `LD_LIBRARY_PATH` |

## 2. 版本锁定

已在 A100 + CUDA 12.8 上验证、并写入锁文件的版本：

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.12 | conda 环境提供；镜像里是 venv on conda |
| CUDA | 12.8 | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` + conda `pytorch-cuda=12.8` |
| torch | 2.10.0+cu128 | **只用 conda 安装**，pip 不碰 torch |
| torchaudio | 2.10.0 | 与 torch 同批 conda 安装 |
| transformers | 4.51.3 | 锁定 |
| onnxruntime-gpu | 1.20.2 | 不要升到 1.31.x（需要 CUDA 13） |
| protobuf | 4.25.8 | 已验证下界为 `>=4.25.8`；锁文件取下界 |
| flatbuffers | 24.3.25 | 版本未在验证记录中逐字保存，标 TODO |
| soundfile | 0.13.1 | 版本未逐字保存，标 TODO；torchaudio 2.10 的 wav 垫片依赖它 |
| numpy | 1.26.4 | 上游 CosyVoice / scipy 1.11.4 约束；GPU 锁文件在 runtime 锁之后安装，会把 numpy 从 2.x 降回 1.26.4 |

`requirements-cosyvoice.lock` 里每个未逐字确认的版本都带 `TODO` 注释。构建前应在一台联网的 GPU 主机上执行：

```bash
python -m pip freeze > requirements-cosyvoice.freeze.txt
# 或
pip install pip-tools
pip-compile --generate-hashes -o requirements-cosyvoice.lock requirements-cosyvoice.in
```

把 freeze 结果合并回锁文件后，才算真正可复现的 hash lock。当前文件是“直接依赖锁定 + pip 解析传递依赖”的中间状态。

`deploy/docker/conda-torch.yaml` 也只锁了版本号，没有锁 conda build string。生产上建议补成 `pytorch=2.10.0=cuda128_py312*` 这类精确 build，或提交 `conda-lock` 产物。

## 3. 构建

CPU 网关（默认 target，CI 用的就是这条）：

```bash
docker build -t cosyvoice-runtime:0.1.0 .
```

GPU 镜像：

```bash
docker build \
  --target gpu \
  -t registry.example.com/cosyvoice-runtime:0.1.0-gpu .
```

生产构建应把基础镜像钉到 digest，并显式传参：

```bash
docker build --target gpu \
  --build-arg CUDA_IMAGE='nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:<digest>' \
  --build-arg PYTHON_IMAGE='python:3.12-slim@sha256:<digest>' \
  --build-arg MICROMAMBA_VERSION=1.5.10 \
  -t registry.example.com/cosyvoice-runtime:0.1.0-gpu .
```

基础镜像 tag 和 digest 需要在可访问 NVIDIA/内部镜像源的主机上确认；当前环境 Docker Hub 不可达，未实际 build。标签 `12.8.1-cudnn-runtime-ubuntu22.04` 是 NVIDIA 的常规命名，但上线前必须用 `docker manifest inspect` 或内部 registry 核对。

### 3.1 加速栈镜像（可选）

vLLM（LLM 连续 batching）和 TensorRT（Flow 解码器）不打进基线镜像。需要时用 build-arg：

```bash
docker build --target gpu \
  --build-arg WITH_ACCEL=1 \
  -t registry.example.com/cosyvoice-runtime:0.1.0-gpu-accel .
```

该参数在 venv 里安装 `requirements-accel.lock`（`vllm==0.19.1` + `tensorrt-cu12==10.13.3.9`），
torch 仍是 conda 的 2.10.0+cu128（满足 vLLM 的 pin，不会被 pip 替换），`transformers` 会从
4.51.3 升到 4.57.1。构建时只校验包元数据，不 import（构建机通常没有 GPU）；真实可用性
必须在目标 GPU 上用 runbook 7.3 的 `gpu_smoke.py --vllm/--trt` 验证。

加速栈需要在模型目录里生成 `<artifact>/vllm` 和 `flow.decoder.estimator.fp32.mygpu.plan`，
而模型卷通常只读挂载；**先在一台可写主机上用同一镜像预生成，再作为只读卷挂载**，
否则启动会直接返回 503（`<artifact> is not writable`）。plan 与 GPU 架构绑定，不能跨卡复用；
文件名里的 `mygpu` 是上游的固定字面量，换卡后**看不出** plan 是否过期，只能删掉重编或看加载报错。

## 4. 启动与模型资产挂载

镜像**不包含**模型权重和 CosyVoice 上游代码。GPU 容器需要三个只读挂载：

| 挂载点 | 内容 | 说明 |
|---|---|---|
| `/data/models` | `Fun-CosyVoice3-0.5B/` 模型目录 | 与 `configs/runtime.gpu.yaml` 的 `artifact` 对齐 |
| `/data/cvtest` | CosyVoice 上游 checkout（含 `third_party/Matcha-TTS`） | 或挂到 `/opt/CosyVoice` 并设 `COSYVOICE_REPO` |
| `/data/prompts` | 零样本 prompt wav | 路径写进 runtime 配置的 `voices.<id>.prompt_wav` |

模型目录至少包含（以 `docs/ops/cosyvoice-runtime-runbook.md` 为准）：

```text
Fun-CosyVoice3-0.5B/
  cosyvoice3.yaml
  llm.pt
  flow.pt
  hift.pt
  campplus.onnx
  speech_tokenizer_v3.onnx
  CosyVoice-BlankEN/
```

上游代码固定到 `docs/design/cosyvoice-training.md` 记录的 commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`，并初始化 `third_party/Matcha-TTS`。entrypoint 会把 `$COSYVOICE_REPO` 和 `$COSYVOICE_REPO/third_party/Matcha-TTS` 加进 `PYTHONPATH`，不需要在容器里做 editable install。

示例（与仓库模板路径完全一致）：

```bash
docker run --rm --gpus all -p 8000:8000 \
  --read-only --tmpfs /tmp:rw,size=2g \
  -e COSYVOICE_API_KEYS='replace-with-a-long-random-key' \
  -e COSYVOICE_METRICS_TOKEN='replace-with-a-metrics-token' \
  -e COSYVOICE_LOG_FORMAT=json \
  -e COSYVOICE_REPO=/data/cvtest \
  -v /data/models:/data/models:ro \
  -v /data/cvtest:/data/cvtest:ro \
  -v /data/prompts:/data/prompts:ro \
  registry.example.com/cosyvoice-runtime:0.1.0-gpu
```

如果挂到 `/opt/CosyVoice`：

```bash
-v /data/cvtest:/opt/CosyVoice:ro
-e COSYVOICE_REPO=/opt/CosyVoice
```

此时 `configs/runtime.gpu.yaml` 里的 `prompt_wav: /data/cvtest/asset/...` 需要改成实际挂载路径，或者把 prompt 目录也挂到 `/data/cvtest/asset`。模型路径同理：要么挂到 `/data/models`，要么改 `models.<name>.artifact`。

### 4.1 关键环境变量

| 变量 | 默认/建议 | 作用 |
|---|---|---|
| `COSYVOICE_RUNTIME_CONFIG` | GPU 镜像默认 `/app/configs/runtime.gpu.yaml` | runtime YAML；生产建议用自己的 ConfigMap 覆盖 |
| `COSYVOICE_REPO` | `/opt/CosyVoice` | 上游 CosyVoice checkout；entrypoint 据此拼 `PYTHONPATH` |
| `COSYVOICE_MATCHA_REPO` | `$COSYVOICE_REPO/third_party/Matcha-TTS` | Matcha-TTS 路径 |
| `COSYVOICE_GPU_ENV` | `/opt/cosyvoice/gpu_env.sh` | entrypoint source 的库路径脚本 |
| `COSYVOICE_EXTRA_LIB_DIRS` | 空 | 冒号分隔的额外库目录，优先级最高 |
| `COSYVOICE_ENGINE_TYPE` | 由 ConfigMap 决定 | 覆盖 `engine.type`；设为 `cosyvoice` 时 entrypoint 会检查 repo |
| `COSYVOICE_API_KEYS` | 生产必填 | 逗号分隔的 Bearer key；来自 Secret，不要写进镜像 |
| `COSYVOICE_METRICS_TOKEN` | 生产必填 | `/metrics` 独立 token |
| `COSYVOICE_STORAGE_SIGNING_SECRET` | `storage.type=local` 时必填 | 签名 URL |
| `COSYVOICE_LOG_FORMAT` / `COSYVOICE_LOG_LEVEL` | `json` / `INFO` | 日志 |
| `CUDA_VISIBLE_DEVICES` | 由 K8s device plugin 分配 | 单进程独占一张 GPU |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` | 禁止运行时访问 Hugging Face Hub；调试时可设 0 |
| `COSYVOICE_HEALTHCHECK_URL` | `http://127.0.0.1:8000/readyz` | 覆盖端口/路径时同步修改 |

安全提醒：`configs/runtime.gpu.yaml` 是单机验收模板，`auth.enabled: false` 且 `metrics_public: true`，**不能直接用于生产**。生产用 Helm values 或自己的 ConfigMap 打开鉴权、关闭公开 metrics。

### 4.2 健康检查

- `/healthz`（别名 `/livez`）：只表示进程和事件循环存活，不检查模型。用于 liveness。
- `/readyz`：未 drain、引擎已 ready（`preload=true` 时模型已加载）才返回 200；否则 503。用于 readiness 和 Docker `HEALTHCHECK`。
- GPU 冷启动模型加载可能几十秒到几分钟。Dockerfile 的 `start-period=600s` 只是兜底；Kubernetes 应使用 chart 的 `startupProbe`（`failureThreshold: 60`、`periodSeconds: 10`，最多 10 分钟）。
- 容器内检查：

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
```

### 4.3 Kubernetes / Helm

chart 已提供 `/healthz`、`/readyz` 探针和 `gpu.enabled`。模型/代码挂载用 `extraVolumes` / `extraVolumeMounts`：

```yaml
extraVolumes:
  - name: models
    persistentVolumeClaim:
      claimName: cosyvoice-models
  - name: cosyvoice-src
    persistentVolumeClaim:
      claimName: cosyvoice-upstream
extraVolumeMounts:
  - name: models
    mountPath: /data/models
    readOnly: true
  - name: cosyvoice-src
    mountPath: /opt/CosyVoice
    readOnly: true
```

`readOnlyRootFilesystem=true` 已由 chart 设置；镜像把 numba/torch/matplotlib 缓存都指向 `/tmp`，chart 已挂 `emptyDir` 到 `/tmp`。模型文件需要对 UID 10001 可读。

## 5. 已知坑

### 5.1 onnxruntime-gpu 的 CUDA 版本

`onnxruntime-gpu 1.31.x` 需要 CUDA 13，在 CUDA 12.8 环境会报：

```text
ImportError: libcublasLt.so.13: cannot open shared object file
```

本镜像锁 `1.20.2`。不要用 `onnxruntime-gpu>=1.17` 这种无上界约束，也不要在 `packages/runtime[cosyvoice]` 的 extra 上直接 `pip install`——extras 只写了 `onnxruntime>=1.17`，会解析到 CPU 版或 CUDA 13 版。Dockerfile 使用 `--no-deps` 安装本地 wheel，并用锁文件控制引擎依赖。

### 5.2 torchaudio 2.10 移除 soundfile backend

torchaudio 2.10 把 I/O 统一到 TorchCodec，不再有 soundfile backend。直接依赖 `torchaudio.load()` 会在 prompt wav 上失败。运行时已通过 [cosyvoice_compat.py](../../packages/runtime/src/cosyvoice_runtime/engines/cosyvoice_compat.py) 的 `install_wav_loader()` 把 `cosyvoice.utils.file_utils.load_wav` 和 `cosyvoice.cli.frontend.load_wav` 替换为 soundfile 实现，不需要改上游 checkpoint 或业务代码。

镜像因此必须安装 `soundfile` 和系统 `libsndfile1`；不要设置 `torchaudio.set_audio_backend("soundfile")`，也不要为了“修 TorchCodec”去 pip 安装 TorchCodec——它和 torch/CUDA/FFmpeg 的 ABI 绑定很紧。

### 5.3 LD_LIBRARY_PATH

onnxruntime-gpu 自己不链接 CUDA/cuDNN，运行时 `dlopen`。没有 `gpu_env.sh` 时典型报错：

```text
libcublasLt.so.12: cannot open shared object file
libcudnn.so.9: cannot open shared object file
```

`gpu_env.sh` 的处理顺序：

1. `COSYVOICE_EXTRA_LIB_DIRS`
2. `torch/lib`
3. torch 所在 site-packages 的 `nvidia/*/lib`（pip torch wheel 布局）
4. torch 所在 conda/venv prefix 的 `lib`（conda torch 布局）
5. venv purelib 的 `nvidia/*/lib`
6. `/usr/local/cuda/lib64`、`/usr/local/nvidia/lib64`

`gpu_env.sh` 还会在 `COSYVOICE_REPO` 指向一个含 `cosyvoice/` 的 checkout 时，把该目录和 `third_party/Matcha-TTS` 追加进 `PYTHONPATH`。容器里 entrypoint 做同样的拼接；裸机部署因此可以复用同一条启动命令，不会报 `No module named 'cosyvoice'`。

排查：

```bash
COSYVOICE_GPU_ENV_VERBOSE=1 . /opt/cosyvoice/gpu_env.sh
python -c "import ctypes; ctypes.CDLL('libcublasLt.so.12'); ctypes.CDLL('libcudnn.so.9')"
ldd "$(python -c 'import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))')/capi/libonnxruntime_providers_cuda.so" | grep 'not found'
```

如果在 Kubernetes 里自定义了 `NVIDIA_DRIVER_CAPABILITIES`，必须保留 `compute`（通常 `compute,utility`）；只给 `utility` 会导致 `libcuda.so` 不可用。

### 5.4 fp16 在 A100 上更慢，默认关闭

`configs/runtime.gpu.yaml` 和 `packages/runtime` 的默认值都是 `fp16: false`。在 A100 上，这个规模的 TTS 模型用 TF32（`tf32: true`，默认开启）通常比 fp16 更快：fp16 需要额外的 dtype 转换，且小矩阵/自回归解码对 HBM 带宽不敏感，收益很小；同时 fp16 还可能带来音质回归。

不要因为“A100 支持 fp16”就打开它。要开必须用 `scripts/gpu_smoke.py --fp16` 和真实文本分布做 A/B，记录首包、RTF、显存和音质，再决定。

### 5.5 numpy / protobuf / flatbuffers

- GPU 锁文件把 numpy 固定为 1.26.4，因为 scipy 1.11.4 要求 `numpy<1.28`，上游 CosyVoice 也按 1.26.x 验证。runtime 锁文件先装 numpy 2.4.6，GPU 锁文件随后降级；这是有意为之，不要“统一升级”。
- `protobuf==4.25.8` 是已验证下界。上游 CosyVoice 历史上写 `protobuf==3.20.2`，与 onnx/onnxruntime 冲突；本镜像按 4.25.8 验证。
- `flatbuffers` 是 onnxruntime 的依赖，版本未逐字记录，锁文件标 TODO。

### 5.6 Hugging Face 离线

GPU 镜像默认 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。模型目录里的 `CosyVoice-BlankEN/` 必须自带 config/tokenizer，否则启动时不会去 Hub 拉取，会直接失败。需要临时调试时：

```bash
-e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0
```

生产不建议打开：运行时访问外网既不可复现，也会把首包延迟变成网络延迟。

### 5.7 只读根文件系统

镜像以 UID 10001 运行，`PYTHONDONTWRITEBYTECODE=1`，并把 `NUMBA_CACHE_DIR`、`MPLCONFIGDIR`、`TORCHINDUCTOR_CACHE_DIR`、`TORCH_HOME` 都指向 `/tmp`。K8s 必须给 `/tmp` 挂可写卷（chart 已做）；否则 numba/torch JIT 会在只读层写缓存失败。

### 5.8 模型权限

模型和 prompt wav 通常由 root 或 CI 用户写入。容器以 10001 读取，挂载前确认目录和文件至少 `o+r`（目录 `o+x`）：

```bash
chmod -R a+rX /data/models/Fun-CosyVoice3-0.5B
```

否则 `/readyz` 会一直 503，日志里是 `model artifact not found` 或 permission denied。

## 6. Docker Hub 不可达时的替代方案

当前环境无法访问 Docker Hub，因此没有执行 `docker build`。可选路径按推荐顺序：

### 6.1 私有 registry / 镜像代理

把基础镜像同步到内部 registry，构建时用 build-arg 指向它：

```bash
# 在有网机器上
docker pull nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
docker tag  nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 \
            registry.internal/mirror/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
docker push registry.internal/mirror/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# 构建机
docker build --target gpu \
  --build-arg CUDA_IMAGE=registry.internal/mirror/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 \
  --build-arg PYTHON_IMAGE=registry.internal/mirror/library/python:3.12-slim \
  -t registry.internal/cosyvoice-runtime:0.1.0-gpu .
```

K8s 拉取私有 registry 用 `imagePullSecrets`（chart 已支持 `imagePullSecrets`）。也可以配置 containerd/daemon 的 registry mirror，让 `nvidia/cuda` 透明走内部源。

### 6.2 预拉基础镜像并离线传输

```bash
# 有网机器
docker pull nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
docker save nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 | gzip > cuda-12.8.1.tar.gz
sha256sum cuda-12.8.1.tar.gz > cuda-12.8.1.tar.gz.sha256

# 构建机
sha256sum -c cuda-12.8.1.tar.gz.sha256
docker load < cuda-12.8.1.tar.gz
docker build --target gpu -t cosyvoice-runtime:0.1.0-gpu .
```

也可以用 `skopeo copy docker://nvidia/cuda:... dir:/media/usb/cuda` 或 `crane export`，适合跨 air-gap 传输。注意 `docker save` 不保留多架构 manifest，只保留当前平台。

### 6.3 离线 wheelhouse

在联网的 Linux x86_64 / Python 3.12 主机上准备 pip wheel（不要包含 torch，torch 来自 conda）：

```bash
pip download --only-binary=:all: \
  --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 \
  --python-version 3.12 --implementation cp --abi cp312 \
  -d wheelhouse \
  -r requirements-runtime.lock -r requirements-cosyvoice.lock

# antlr4-python3-runtime 4.9.3 只有 sdist，需要单独构建：
pip wheel --no-deps -w wheelhouse antlr4-python3-runtime==4.9.3
```

> 平台标签说明：`numpy 2.4.6`、`onnxruntime-gpu 1.20.2` 使用 `manylinux_2_27/2_28`，只传 `manylinux2014_x86_64` 会解析失败。Ubuntu 22.04（glibc 2.35）和 `python:3.12-slim`（Debian bookworm，glibc 2.36）都兼容 `manylinux_2_28`，所以两个标签都要传。

conda 侧用同一份 `deploy/docker/conda-torch.yaml` 在联网机器上创建环境，再打包 pkgs 缓存或整个 `/opt/conda/envs/torch`：

```bash
micromamba create -y -p /tmp/torchenv -f deploy/docker/conda-torch.yaml
tar -C /tmp -czf torchenv.tar.gz torchenv
# 构建机解压到 /opt/conda/envs/torch，Dockerfile 里改为 COPY + 跳过 create
```

离线构建时，把 Dockerfile 中的 pip 安装改成：

```dockerfile
COPY wheelhouse /wheelhouse
RUN pip install --no-index --find-links=/wheelhouse -r /tmp/requirements-runtime.lock
RUN pip install --no-index --find-links=/wheelhouse -r /tmp/requirements-cosyvoice.lock
```

并把 micromamba/conda create 改为 `--offline` 或直接 `COPY` 预建环境。当前 Dockerfile 没有内置离线分支，需要在离线构建流水线里做上述替换，或用内部 PyPI/conda 镜像（`PIP_INDEX_URL`、`PIP_TRUSTED_HOST`、`.condarc`）让原有 `RUN` 直接工作。

另注意 `wheelbuilder` stage：`pip wheel ./packages/common ./packages/runtime` 默认启用 build isolation，会去索引下载 setuptools；`python:3.12-slim` 的 ensurepip 不再自带 setuptools。完全离线时需要先在 wheelhouse 里准备 setuptools/wheel，并把该 stage 改成 `pip wheel --no-build-isolation`（同时预装构建后端），或者直接在联网机器上构建好两个本地 wheel 再 `COPY` 进镜像。

### 6.4 内部 PyPI / conda 镜像

如果只能访问内部镜像源，最小改动是给构建传代理：

```bash
docker build --target gpu \
  --build-arg PIP_INDEX_URL=https://pypi.internal/simple \
  --build-arg PIP_TRUSTED_HOST=pypi.internal \
  ...
```

当前 Dockerfile 没有声明这些 ARG；使用 `--build-arg` 前需要在 `gpu-torch`/`cpu` stage 加 `ARG PIP_INDEX_URL` 并 `ENV PIP_INDEX_URL=...`，或直接在构建机 `~/.config/pip/pip.conf` 配好。conda 用 `micromamba config` 或 `.condarc` 指向内部 channel。

## 7. 验证清单

镜像 build 成功后，至少执行：

```bash
# 1) 版本与动态库（不需要 GPU）
docker run --rm cosyvoice-runtime:0.1.0-gpu python -c \
  "import torch, onnxruntime as ort, transformers; \
   print(torch.__version__, torch.version.cuda, ort.__version__, transformers.__version__); \
   print(ort.get_available_providers())"
docker run --rm cosyvoice-runtime:0.1.0-gpu sh -c \
  '. /opt/cosyvoice/gpu_env.sh && for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcufft.so.11 libcurand.so.10 libcudnn.so.9; do \
     python -c "import ctypes, sys; ctypes.CDLL(sys.argv[1])" "$lib" || exit 1; \
   done && echo "all CUDA sonames resolved"'

# 2) GPU 可见
docker run --rm --gpus all cosyvoice-runtime:0.1.0-gpu python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3) 脱离网关的模型冒烟
docker run --rm --gpus all \
  -e COSYVOICE_REPO=/data/cvtest \
  -v /data/models:/data/models:ro -v /data/cvtest:/data/cvtest:ro \
  cosyvoice-runtime:0.1.0-gpu \
  python /app/scripts/gpu_smoke.py ...   # 镜像未打包 scripts/；用挂载或本地执行

# 4) 服务探针
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz

# 5) 真实请求 + 压测
python scripts/api_smoke.py http://127.0.0.1:8000 cosyvoice3-0.5b assistant
python scripts/loadtest.py --mode ws --concurrency 4 --tasks 16 ...
```

注意：镜像只打包 `configs/` 和运行库，不打包 `scripts/`。`gpu_smoke.py` 应在宿主机跑，或临时把仓库 `scripts/` 挂进容器。服务端的真实容量验收仍按 `docs/ops/cosyvoice-runtime-runbook.md` 执行。

## 8. 当前无法验证的点

本环境 Docker Hub 不可达，未执行 `docker build`，以下内容需要在有网/有 GPU 的构建机上确认：

1. `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`、`python:3.12-slim`、`micromamba 1.5.10` 的 tag/digest 是否仍可拉取。
2. conda 是否能解析 `pytorch=2.10.0 + torchaudio=2.10.0 + pytorch-cuda=12.8 + python=3.12`；是否需要补 build string 或 channel priority。
3. `requirements-cosyvoice.lock` 的传递依赖（librosa → numba/llvmlite/scikit-learn 等）在 Python 3.12 上的实际解析结果；当前只锁了直接依赖。
4. `antlr4-python3-runtime==4.9.3` 只有 sdist，离线构建需要预构建 wheel。
5. `libcudnn.so.9` 的确切 soname 和路径：conda torch 与 nvidia/cuda cudnn runtime 两套来源，构建期断言已验证 soname，但未在真实 GPU 上跑推理。
6. `transformers 4.51.3` 加载 `CosyVoice-BlankEN` 是否需要额外的 `sentencepiece`/`tiktoken`；当前按 Qwen2 tokenizer.json 处理，未在容器内实测。
7. CosyVoice 上游代码固定 commit 的 `requirements.txt` 是否还包含锁文件未列出的 import（`onnx`、`openai-whisper`、`ttsfrd` 等）。以 `python -c "from cosyvoice.cli.cosyvoice import CosyVoice3"` 的实际报错为准补依赖。
8. fp16 在 A100 上更慢是已验证结论；具体 RTF/首包数字需用 `scripts/gpu_smoke.py` / `scripts/loadtest.py` 在本机复测。
9. `WITH_ACCEL=1` 的镜像未构建过：vLLM 0.19.1 wheel 与 `nvidia/cuda:12.8.1` 运行时的 CUDA 兼容性、TensorRT 10.13 在容器内的 libnvinfer 加载、以及 `import vllm` 在驱动上的行为，都要在目标构建机/GPU 上确认。

## 9. 本轮依赖核查记录（2026-11-12）

- PyPI 可访问；已用 PyPI JSON API 核对锁文件中每个 pin 是否存在，并确认关键包有 Linux cp312 或 pure-Python wheel：`onnxruntime-gpu 1.20.2`、`transformers 4.51.3`、`numpy 1.26.4`、`scipy 1.11.4`、`pyarrow 18.1.0`、`soundfile 0.13.1`、`tokenizers 0.21.4`、`Pillow 11.0.0` 等。
- 网关依赖版本来自仓库 dev venv 的 `pip freeze`（见 `requirements-runtime.lock` 头部）。
- 未下载完整 Linux wheelhouse：当前机器是 Windows，且 Docker Hub 不可达；wheelhouse 生成命令已写入 §6.3。`antlr4-python3-runtime==4.9.3` 仅有 sdist 已单独标注。
- 未执行 `docker build`。本轮修改了 runtime 引擎（加速栈接线与启动预检）、配置模板、bootstrap 脚本和 Helm values/configmap；这些改动未在容器内验证，`WITH_ACCEL=1` 的构建也未执行。
