# CosyVoice 实时通话技术方案

> 版本：v1.0  
> 适用范围：基于 `Fun-CosyVoice3-0.5B-2512` 的单说话人/少说话人实时语音合成，支持私有化部署，并兼容阿里云百炼 CosyVoice HTTP / WebSocket 接口。  
> 关联文档：
> - [运行时方案](./cosyvoice-runtime.md)
> - [训练方案](./cosyvoice-training.md)

---

## 1. 背景与目标

业务需要一个可用于实时通话的 TTS 服务，核心诉求：

- 音色像目标说话人，支持情绪、语速、停顿、方言等风格。
- 首包低、可流式、可并发，实时通话体验接近真人。
- 训练后的模型可以私有化部署，数据不出域。
- 业务层可以同时接入阿里云百炼 CosyVoice 和私有化服务，**只切换接入地址，不修改请求/响应代码**。

本方案把系统拆成两个平面：

| 平面 | 文档 | 关注点 |
|---|---|---|
| 训练平面 | [cosyvoice-training.md](./cosyvoice-training.md) | 数据清洗、环境准备、SFT、评测、模型交付 |
| 运行时平面 | [cosyvoice-runtime.md](./cosyvoice-runtime.md) | 模型部署、并发优化、流式链路、能力提供、阿里云接口兼容 |

---

## 2. 总体架构

```text
┌─────────────────────────────────────────────────────────────────────┐
│ 业务层（呼叫中心 / 实时对话 Agent / App）                            │
│  只配置 base_url / api_key / model / voice                          │
└───────────────┬─────────────────────────────────────┬───────────────┘
                │ HTTPS / SSE                         │ WSS
                ▼                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ CosyVoice Runtime Gateway                                           │
│  - 阿里云兼容 HTTP API / WebSocket API                              │
│  - 鉴权、租户、限流、任务生命周期                                   │
│  - model/voice 别名映射                                             │
│  - 文本前端、SSML 子集、Markdown 过滤、hot_fix                      │
│  - 流式编排、音频转码、对象存储 URL                                 │
└───────────────┬─────────────────────────────┬───────────────────────┘
                │                             │
                ▼                             ▼
┌───────────────────────────┐   ┌─────────────────────────────────────┐
│ LLM Service               │   │ Token2Wav Service                   │
│ vLLM / TensorRT-LLM       │   │ Triton: Flow (DiT) + HiFT vocoder   │
│ 文本 + speaker + instruct │   │ 动态 batching、流式 chunk            │
│ → speech token            │   │ speech token → mel → waveform       │
└───────────────────────────┘   └─────────────────────────────────────┘
                │                             │
                └──────────────┬──────────────┘
                               ▼
                 ┌─────────────────────────────┐
                 │ 支撑设施                    │
                 │ - 对象存储（音频 URL）      │
                 │ - 模型仓库 / 配置中心       │
                 │ - Prometheus / Loki / OTel  │
                 │ - K8s / GPU Operator        │
                 └─────────────────────────────┘
```

训练平面产出模型 artifact：

```text
data/ + pretrained_models/Fun-CosyVoice3-0.5B
        │
        ▼
LLM SFT → Flow SFT（可选）→ HiFiGAN SFT（可选）
        │
        ▼
exp/cosyvoice3/{llm,flow,hifigan}/.../llm.pt / flow.pt / hifigan.pt
        │
        ▼
模型包：llm.pt + flow.pt + hifigan.pt + spk2info.pt + config + tokenizer/onnx
        │
        ▼
运行时加载
```

---

## 3. 关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 主模型 | `Fun-CosyVoice3-0.5B-2512` base | 音色/相似度/方言/instruct 综合最好，0.5B 成本可控 |
| 微调方式 | 全参 SFT，先 LLM，再按需 Flow | 官方无 LoRA；LLM 管发音/停顿/语速，Flow 管音色/情绪细节 |
| LLM 推理 | vLLM 或 TensorRT-LLM | 连续 batching、吞吐高；TRT-LLM 延迟更低 |
| Token2Wav | Triton + TensorRT/ONNX，FP32 | 官方路径；FP16 有 NaN 风险 |
| 流式 | WebSocket 双工 + HTTP SSE | 同时兼容百炼实时/非实时接口 |
| 接口兼容 | 路径、字段、事件、错误码对齐百炼 | 业务只换 base_url |
| 音频输出 | 对象存储 URL + 流式 base64/binary | 对齐百炼响应；私有化用 MinIO/S3 |
| 部署 | 单机开发 + K8s 生产；LLM/Token2Wav 可分离 | Token2Wav 是并发瓶颈，需要独立扩缩容 |

### 3.1 部署规模基线

| 规模 | 典型场景 | Token2Wav（分散起播，p95 ≤1s） | LLM | Gateway |
|---|---|---|---|---|
| 100 active | 单业务线/单说话人 | 11–20×L20 或 6–10×A100 | 1–2×A100 | 4–6 pods |
| 1000 active | 多业务线/多 Cell | 110–190×L20 或 55–95×A100 | 4–8×A100 | 12–20 pods |

> 表中是**分散起播**的常态值。若 100/1000 路在极短时间内同时起播，Token2Wav 分别约需 36–44 / 360–430×L20，这是外呼集中拨号的最坏值，不应作为常态容量。1000 路连接、30% 活跃、5s 一句 → 约 360 active，Token2Wav 约 40–70×L20。详细公式、Cell 拓扑和压测门槛见 [运行时方案 8.5](./cosyvoice-runtime.md#85-100--1000-并发容量方案)。

---

## 4. 运行时与训练的边界

运行时**不训练模型**，只做：

- 加载训练产出的模型包。
- 管理音色（speaker embedding / reference audio）。
- 把阿里云兼容请求翻译成 CosyVoice 推理调用。
- 流式返回音频，处理并发、限流、降级、观测。

训练侧**不关心**在线协议，只保证：

- 产出可被运行时加载的 `llm.pt` / `flow.pt` / `hifigan.pt` / `spk2info.pt`。
- 产出模型卡：训练数据、评测结果、已知问题、推荐推理参数。
- 模型版本可回滚。

### 4.1 代码与打包边界

同一个代码仓库，但按发行包拆分，避免运行时镜像带入训练依赖：

| 目录 | 发行包 | 依赖 | 说明 |
|---|---|---|---|
| `packages/common` | `cosyvoice-common` | 无内部依赖 | 音频编解码、文本前端、错误模型 |
| `packages/runtime` | `cosyvoice-runtime` | `cosyvoice-common` + FastAPI | 在线服务 |
| `packages/data` | `cosyvoice-data` | `cosyvoice-common` | 清洗、切分、manifest |
| `packages/training` | `cosyvoice-training` | `cosyvoice-common` | SFT；torch 在 `sft` extra 中 |
| `packages/evaluation` | `cosyvoice-evaluation` | `cosyvoice-common` | 客观指标与回归门禁 |

约束：

- `runtime` 与 `training` 互不 import；二者只依赖 `common`。
- 运行时镜像只安装 `cosyvoice-runtime`，训练镜像只安装训练相关包。
- 每个包独立版本、独立 wheel、独立发布；以后拆仓不需要改业务代码。

---

## 5. 环境与版本基线

| 项 | 基线 |
|---|---|
| 代码仓库 | `QwenAudio/CosyVoice`，锁定 commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` |
| 模型 | `Fun-CosyVoice3-0.5B-2512` |
| 采样率 / token 率 | 24 kHz / 25 Hz |
| GPU | 开发 1×24–48GB；生产 1×L20/A100 起，按并发扩 |
| CUDA / PyTorch | CUDA 12.x + PyTorch 2.x（以镜像锁定为准） |
| 运行时 | Python 3.10 + FastAPI/Uvicorn；Triton；vLLM/TRT-LLM |
| 容器 | 训练镜像、运行时镜像分离；模型 artifact 外挂 |

> 具体版本以 [cosyvoice-runtime.md](./cosyvoice-runtime.md) 和 [cosyvoice-training.md](./cosyvoice-training.md) 的锁定清单为准。升级 CUDA/PyTorch/TensorRT 必须重跑兼容性和音质回归。

---

## 6. 交付物清单

| 交付物 | 说明 |
|---|---|
| 模型包 | `llm.pt`、`flow.pt`、`hifigan.pt`、`spk2info.pt`、ONNX tokenizer/embedding、config |
| 运行时镜像 | API Gateway + 编排 + 转码；可独立于模型镜像升级 |
| 模型服务镜像 | LLM 服务、Token2Wav Triton 模型仓库 |
| 接口文档 | 阿里云兼容 HTTP / WebSocket / SSE 契约 |
| 部署清单 | Docker Compose（开发）、K8s Helm/Manifests（生产） |
| 评测报告 | 客观指标 + 主观 MOS + 实时指标 |
| 运维手册 | 健康检查、扩缩容、降级、回滚、常见错误 |

---

## 7. 非目标

- 不在运行时做模型训练/微调。
- 不在 v1 实现百炼全部音色、全部 SSML/LaTeX、AIGC 隐性标识。
- 不承诺单卡无限并发；并发能力以压测和 SLO 为准。
- 不把多说话人零样本平台作为 v1 目标；v1 聚焦单说话人/少说话人实时通话。

---

## 8. 阅读顺序

1. 先读 [cosyvoice-runtime.md](./cosyvoice-runtime.md) 的“阿里云兼容层”和“并发与性能”。
2. 再读 [cosyvoice-training.md](./cosyvoice-training.md) 的“数据清洗”和“评测门禁”。
3. 落地时按两份文档的“实施步骤/验收清单”执行。
