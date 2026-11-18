# 模型与提供方

## 1. 支持的组合

ASR 和 TTS 各自只启动**一个** provider，运行时不会同时加载多个模型。

| 链路 | provider | 模型 | 默认 | 说明 |
|---|---|---|---|---|
| ASR | `paraformer` | FunASR 流式 Paraformer | ✅ | 16 kHz 流式 partial，600 ms 分块，最低延迟 |
| ASR | `qwen3asr` | Qwen3-ASR-0.6B | | 52 语言，transformers 后端，partial 用滚动重转写近似 |
| TTS | `cosyvoice3` | Fun-CosyVoice3-0.5B | ✅ | 本地 GPU，句级流式，首块 0.55–0.85 s |
| TTS | `qwen3tts` | Qwen3-TTS-12Hz-0.6B-Base | | transformers，句级合成后分块，RTF ≈ 1.6–1.7 |
| TTS | `qwen3tts_vllm` | Qwen3-TTS-12Hz-0.6B-Base | | 外部 vLLM-Omni 容器，真流式 PCM |
| TTS | `cosyvoice3_vllm` | Fun-CosyVoice3-0.5B | | 同一个 vLLM-Omni 镜像的第二实例（:8092） |

选择优先级：

```text
命令行 --asr-provider / --tts-provider
  > 环境变量 NV_ASR_PROVIDER / NV_TTS_PROVIDER
  > configs/service.yaml engine.asr_provider / engine.tts_provider
```

默认组合是 `paraformer + cosyvoice3`：延迟最低、显存最省。Qwen 系列适合追求
识别准确率或多语种、或对比音质；真流式且低持续延迟则选 vLLM 版。

## 2. 模型下载

`scripts/fetch_deps.py` 会下载本地运行所需的主要资源：

```bash
uv run --extra gpu python scripts/fetch_deps.py
```

Qwen3-TTS / Qwen3-ASR 的模型需要单独放到 `models/`：

```powershell
.venv\Scripts\modelscope.exe download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base `
  --local_dir models\Qwen3-TTS-12Hz-0.6B-Base
.venv\Scripts\modelscope.exe download --model Qwen/Qwen3-ASR-0.6B `
  --local_dir models\Qwen3-ASR-0.6B
```

或者使用 `scripts/setup_qwen_engines.ps1 -DownloadModels` 一步完成。

## 3. 独立虚拟环境

三组模型的依赖钉法不同，不能共用一个 `.venv`：

| 环境 | Python | transformers | 用途 |
|---|---|---|---|
| `.venv` | 3.10 | 4.51.3 | 网关 + CosyVoice3 / FunASR |
| `envs/qwen3tts` | 3.10 | 4.57.3 | Qwen3-TTS engine |
| `envs/qwen3asr` | 3.10 | 4.57.6 | Qwen3-ASR engine |

`providers` 中的 `python` 指向对应解释器；supervisor 使用该解释器启动
`native_voice.engine.cli <kind> --provider <name>`。切换 provider 时，停掉当前
engine 并用另一个 venv 启动到同一端口（ASR 8101 / TTS 8102），网关和前端
配置不变。同一时刻只有一个 TTS 模型驻留显存。

首次安装（torch 2.5.1+cu121，约 2.5 GB/环境）：

```powershell
scripts\setup_qwen_engines.ps1 -Kind both -DownloadModels
```

```bash
scripts/setup_qwen_engines.sh both --download
```

## 4. 各 provider 的行为与限制

### Paraformer（ASR）

- FunASR 在线模型，原生流式。
- 每个会话独立 cache，共享模型权重，推理串行。
- 默认 `chunk_ms=600`；低于 480 ms 可能出现重复字（如“可可以以”）。
- 句尾 `finalize()` 会立即 flush 尾音，不需要等下一个分块边界。

### Qwen3-ASR

- transformers 后端不支持原生流式；engine 在每个会话中保留累计音频，
  最多每 `partial_interval_ms`（默认 800 ms）在后台重转写一次（累计 ≤ 8 s）。
- `finalize()` 对完整话音频做一次全量转写，最终文本以这次为准。
- 首个 partial 约 1.1 s（Paraformer 约 0.16 s），`finalize` 0.43–0.47 s。
- Linux 生产可改用 vLLM 后端的原生流式接口（尚未接入）。

### CosyVoice3（TTS）

- 25 Hz speech token + DiT，本地 GPU，fp16，CUDA graph。
- 句级流式：LLM 每产出一句就提交合成，首块立即播放。
- 取消/打断只影响当前句，生成器会被静默排空，不泄漏线程。
- 关键优化：音色只提取一次、三个子模型 fp16、采样器向量化、
  每句重置 `token_hop_len`、解码器 StaticCache + CUDA graph。

### Qwen3-TTS（transformers）

- `qwen-tts` 开源包只提供整句生成；`non_streaming_mode=False` 只模拟流式文本
  输入，不是真流式音频。网关会把 LLM 输出切成短句，首句合成完即下发。
- 这是“句级流式”，不是 97 ms token 级流式。
- 音色克隆使用 `create_voice_clone_prompt`，启动时加载 `voices.yaml` 和
  `.data/voices`。
- 取消/打断只能在当前句生成完成后生效；长句会放大打断延迟。
- 实测 RTF 1.6–1.7（Windows/transformers），首音慢于 CosyVoice3。

### Qwen3-TTS vLLM（`qwen3tts_vllm`）

- TTS engine 不加载模型，只把句子代理到 `deploy/vllm-omni` 的 WebSocket。
- 服务端边生成 codec frame 边下发 PCM：实测首帧 80 ms 音频在 0.42 s 到达，
  后续约每 0.85–1.0 s 一块，RTF 0.58–0.60。
- 参考音频必须用 `file:///workspace/...` URI（provider 自动转换），路径需在
  `--allowed-local-media-path /workspace` 挂载内。
- 首次真实请求有约 18 s 的 code2wav/CUDA 懒初始化；`warmup: true` 在 `load()`
  时吸收，`/readyz` 通过后首轮不再付这笔时间。
- 一个 WebSocket 可连续合成多个分句；`idle_reconnect_s`（默认 20 s）主动回收
  旧连接，避免 vLLM 30 s 空闲超时。
- 8 GB 显存下容器约占 4.5 GB，需与 Paraformer（~1.6 GB）和 CPU LLM 同机运行。

### CosyVoice3 vLLM（`cosyvoice3_vllm`）

- 复用同一个 vLLM-Omni 镜像，换成 Fun-CosyVoice3-0.5B checkpoint，
  容器监听 8092。
- 只发送 `ref_audio` / `ref_text`；CosyVoice3 不接受 Qwen3-TTS 的
  `task_type`、`x_vector_only_mode`、`language`，provider 会自动省略。
- 8 GB + `enforce_eager: true` 下首块 0.66–0.88 s，与本地版接近；
  优势是真流式与句间流水，首音想更快需要更大显存或关闭 eager。
- 容器 + Paraformer + 桌面约占 7.6 GB，此时 LLM 必须放 CPU。

### LLM 与工具调用

- 网关用 `--jinja` 启动 llama-server，由 GGUF 内嵌的 chat template 解析工具调用，
  转换成 OpenAI 兼容的 `tool_calls`。
- 原始权重为 ModelScope `Qwen/Qwen3.5-2B`，网关使用 unsloth 的 Q8_0 GGUF
  （`unsloth/Qwen3.5-2B-GGUF`，Apache-2.0）。
- 默认 `models/Qwen3.5-2B-Q8_0.gguf`（unsloth GGUF）具备工具调用能力。实测：
  系统提示词必须明确写“你不知道当前时间”，否则模型倾向直接猜时间；
  加上该约束后，`auto` 模式下“现在几点了”稳定调用 `get_current_time`，
  工具轮 content 为空，回填结果后生成口语回答。当前 llama.cpp b12125（CPU）
  与本地 CUDA 构建都能加载 `qwen35` 架构。
- 联网工具 `web_search`（Bing RSS）和 `web_fetch`（公网网页正文）可用，
  需要网关能访问外网；`web_fetch` 会拒绝内网/回环/云元数据地址。
- 需要强制验证工具链路时用 `tool_choice: required`。
- `required` 会强制首轮调用（连“你好”也会调用），只适合联调，生产保持 `auto`。
- 该 GGUF 是多模态模型，语音链路只用文本部分，不需要下载 `mmproj`；
  原生上下文 262144，网关按 `ctx` / `slots` 切分。
- 显存不足时把 LLM 放在 CPU：Q8_0 权重约 2.0 GB，CPU 推理约 25 tok/s；
  更大模型（4B/8B Q4）触发工具调用更稳，但首 token 与生成延迟会上升。

## 5. 实测性能

以下数据来自 Windows 11 + RTX 4070 Laptop 8 GB，仅作参考。

### 全链路（本地 CosyVoice3 + CPU Qwen3.5-2B）

| 指标 | 数值 |
|---|---|
| TTS 首块 | 0.55–0.85 s |
| TTS 持续合成 RTF | 0.38–0.52 |
| ASR 部分结果 | 每 600 ms；流式 RTF ≈ 0.1 |
| ASR 句尾收尾 | 0.03–0.15 s；hybrid 加 SenseVoice 0.1–0.4 s |
| LLM 首 token | 预热后 ~0.08 s；冷启动 ~0.4 s（CPU，8 线程） |
| LLM 生成速度 | ~25 tok/s（CPU，8 线程，Q8_0） |
| TTS 预热后显存 | ~1.8 GB |
| hybrid 全链路显存 | ~3.7 GB |
| 回答首字到出声 | ~2.1 s |

> 早期版本使用 Qwen3-1.7B，`LLM 首 token` 为预热后 ~0.16 s、冷启动 ~1 s；
> 其余指标与 LLM 型号基本无关。

### 并发（单 TTS engine）

- 4 路 TTS：首块 p50 4.02 s / max 7.11 s。
- 2 路 TTS + 2 路 ASR：TTS 首块 p50 2.27 s，ASR 首结果 0.16 s。
- **8 GB 单卡建议承诺 2 路**；4 路可用但 TTS 会排队。

### Qwen3 provider

- ASR-0.6B：加载 8–10 s，显存 1.6 GB；3.48 s 音频离线转写 0.46 s；
  首个 partial ~1.1 s；`finalize` 0.43–0.47 s。
- TTS-0.6B：加载 9–20 s，显存 2.2 GB；“你好。”首音/总耗时 1.11 s；
  13 字句约 4.9 s；RTF 1.6–1.7。
- 两个 Qwen engine 同时运行：显存约 4.5 GB。
- 端到端：首分句较短时 `tts_first_ms=1.27 s`、`total_ms=2.31 s`。

### vLLM-Omni（Linux 容器）

- 仅 TTS 时显存 4.52 GB，可与 Paraformer + CPU LLM 同机。
- Qwen3-TTS 真流式：15 字句 `audio.start` 0.419 s，首块 0.420 s，
  RTF 0.58。
- 端到端（Paraformer + CPU LLM + vLLM TTS，LLM 为当时的 Qwen3-1.7B）：
  `tts_first_ms=406`、`total_ms=1078`。
- 容器首次到 `/health` 约 90 s；首次合成懒初始化约 18 s（provider warmup 吸收）。

## 6. 音色克隆（zero-shot）

### 使用

网页端点击音色下拉旁「＋ 克隆」，可以：

- 直接录音 1.5–30 s（建议 3–10 s，安静单人）；
- 或上传 wav/flac/mp3/ogg（≤ 20 MB）；
- 填写名称、可选参考文本（留空自动 ASR）、勾选授权后上传。

服务端会裁剪首尾静音、重采样到 24 kHz、归一化，并给参考文本加
`You are a helpful assistant.<|endofprompt|>` 前缀。

存储：参考音频和元数据保存在 `.data/voices/`。网关与 TTS engine 必须共享
该目录（Docker 需挂载同一 volume）。TTS engine 启动时加载全部克隆音色；
运行时通过内部 `POST /voices` 注册，与合成任务串行。

### 质量提示

- zero-shot 克隆不是微调；参考音频干净、单人、3–10 秒通常最稳。
- 参考文本必须与音频一致，否则可能不收敛。
- 商用需确认声音授权，并遵守模型许可。

## 7. 显存建议（8 GB）

| 组合 | 显存 |
|---|---|
| Paraformer + 本地 CosyVoice3 + CPU LLM | ~3.7 GB |
| Paraformer + Qwen3-TTS（transformers）+ CPU LLM | ~3.8 GB |
| Qwen3-ASR + Qwen3-TTS | ~4.5 GB |
| vLLM-Omni TTS + Paraformer + CPU LLM | ~6.1 GB |

不要同时启动 CosyVoice3 和 Qwen3-TTS，也不要同时运行两个 vLLM-Omni 容器。
