# Qwen3-TTS 实时通话场景调研

> 调研日期：2026-11-12  
> 目标场景：实时通话、音色 + 说话风格/情感、拟人度高、同算力下更高并发  
> 调研范围：Qwen3-TTS 官方开源模型、官方微调脚本、已知问题与社区修复、实时流式与并发部署方案  
> 官方仓库基线：`QwenLM/Qwen3-TTS` main 分支，commit `022e286b98fbec7e1e916cb940cdf532cd9f488e`

---

## 0. 结论摘要

1. **Qwen3-TTS 已开源，官方提供微调脚本，但只支持 `12Hz Base` 系列的单说话人全参 SFT。**
   - 官方脚本未提供 LoRA。
   - 官方暂不支持多说话人微调。
   - 官方 Base 微调不提供 instruct/情感控制监督。
2. **音色和情感要分开解决。**
   - 音色：Base 单说话人 SFT，训练后把 speaker embedding 烘进模型，推理不再传 `ref_audio`。
   - 情感：优先“固定人设 + 表达性对话数据 + 文本导演”；动态情绪控制需要额外的 control token/instruct 训练，官方不支持，需 POC。
3. **官方 `qwen-tts` Python 包不适合生产实时并发。**
   - 它没有真正的音频流式输出，`non_streaming_mode=False` 只是模拟文本流式输入。
   - 单实例多线程并发会退化，批量推理优于多线程，但仍不是连续批处理。
4. **实时通话必须走 vLLM-Omni、社区流式 fork 或自研引擎。**
   - 官方把流式和在线服务主要交给 vLLM-Omni 社区推进。
   - 社区已有 PCM 流式 fork、GGUF/ONNX 方案，但都属于实验性，需要锁版本和回归测试。
5. **并发的主要杠杆是模型大小、量化、蒸馏、连续批处理、缓存和首句拆分，而不是继续把 1.7B 训得更好。**
   - 质量档：1.7B Base SFT。
   - 并发档：0.6B Base SFT 或 1.7B 蒸馏到 0.6B。
6. **官方微调脚本存在已知 bug，直接长训有白训风险。**
   - label double-shift、sub-talker 对齐、`text_projection` 缺失、speaker encoder 被 weight decay 等。
   - 相关修复大多仍是 open PR，训练前需要打补丁或使用经过审计的 fork。

---

## 1. 官方模型与能力边界

### 1.1 已发布模型

| 模型 | 能力 | 是否可用于官方微调 | 说明 |
|---|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-Base` | 3 秒快速音色克隆 | 是 | 质量上限，推荐做主训练 |
| `Qwen3-TTS-12Hz-0.6B-Base` | 3 秒快速音色克隆 | 是 | 更省算力，适合并发档/蒸馏目标 |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 个内置音色 + instruct 风格控制 | 官方未支持 | 适合快速验证，但音色不自由 |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 9 个内置音色 + instruct | 官方未支持 | 高并发 MVP 候选 |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 根据文字描述设计音色 | 官方未支持 | 可用于生成参考音色，再交给 Base 克隆 |
| `Qwen3-TTS-Tokenizer-12Hz` | 音频 codec，12Hz | — | 微调前必须用它提取 `audio_codes` |

### 1.2 架构与能力

- 离散多 codebook 的 LM 架构：文本 tokenizer → LLM backbone → speech tokenizer/codec → decoder。
- `Qwen3-TTS-Tokenizer-12Hz`：12 帧/秒，比 25Hz 更适合实时和高并发。
- 支持 10 种语言：中、英、日、韩、德、法、俄、葡、西、意。
- 模型层面支持流式/非流式双轨生成；官方宣称端到端合成延迟可低至 97ms。
  - 注意：97ms 是模型/论文层面的指标，基于 vLLM v0 + `torch.compile` + CUDA graphs，且在 A100/H100 级别硬件、每 4 个 token 发射一次的条件下测得。
  - 实际自托管首包延迟通常在 150–300ms 量级，端到端通话还要叠加 ASR、LLM、网络和抖动缓冲。

### 1.3 官方微调边界

官方 `finetuning/README.md` 明确：

- 当前支持 **单说话人 fine-tuning**。
- 多说话人和其他高级微调能力“future releases”。
- 微调输入是 `12Hz Base`，输出会被转换成 `custom_voice` 模型，并注册一个新的 speaker（默认 id `3000`）。

官方脚本 `sft_12hz.py` 的实际行为：

- 全参训练 `model.talker` 和 sub-talker。
- `speaker_encoder` 输出被 `.detach()`，但参数仍在 `AdamW` 中，会被 weight decay 影响。
- 每个 epoch 保存一个完整 checkpoint：复制初始模型目录，再覆盖 `model.safetensors`。
- 保存时删除 `speaker_encoder.*`，把第一个 batch 的 speaker embedding 写入 `talker.model.codec_embedding.weight[3000]`。
- 修改 `config.json`：`tts_model_type = "custom_voice"`，写入 `spk_id` 和 `spk_is_dialect`。

---

## 2. 官方微调流程（单说话人）

### 2.1 环境

```powershell
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# 先装与本机 CUDA 匹配的 torch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
pip install -e .
```

Windows 上 `flash-attn` 安装困难，官方脚本硬编码 `flash_attention_2`，需要改成 `sdpa`，或参考 PR #373 让 attention 实现可配置。

### 2.2 下载权重到本地

```powershell
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir D:/models/Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz --local_dir D:/models/Qwen3-TTS-Tokenizer-12Hz
```

**必须传本地目录给 `--init_model_path`。** 官方脚本保存 checkpoint 时会 `shutil.copytree(MODEL_PATH, ...)`，如果传 HuggingFace repo id 会报 `FileNotFoundError`。

### 2.3 训练 JSONL

```json
{"audio":"D:/data/spk01/0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"D:/data/spk01/ref.wav"}
{"audio":"D:/data/spk01/0002.wav","text":"She said she would be here by noon.","ref_audio":"D:/data/spk01/ref.wav"}
```

要求：

- `audio`：24kHz、单声道、2–10 秒。
- `text`：逐字转写，包含语气词、重复、停顿对应的标点。
- `ref_audio`：**所有样本使用同一个参考音频**。
  - 官方 README 强烈建议这样做。
  - 官方脚本只在第一个 batch 锁定一次 `target_speaker_embedding` 用于保存，后续 batch 仍使用在线 encoder 输出；ref 不一致会造成训练/推理条件不一致。
- 不同长度的 `ref_audio` 会导致 `collate_fn` 里的 `torch.cat` 维度不匹配；PR #381 修复过，但已关闭且未合入。

### 2.4 提取 audio codes

```powershell
python finetuning/prepare_data.py `
  --device cuda:0 `
  --tokenizer_model_path D:/models/Qwen3-TTS-Tokenizer-12Hz `
  --input_jsonl D:/data/train_raw.jsonl `
  --output_jsonl D:/data/train_with_codes.jsonl
```

`prepare_data.py` 会批量调用 `Qwen3TTSTokenizer.encode()`，给每行追加 `audio_codes`。

### 2.5 训练

```powershell
python finetuning/sft_12hz.py `
  --init_model_path D:/models/Qwen3-TTS-12Hz-1.7B-Base `
  --output_model_path D:/output/qwen3-tts-ft `
  --train_jsonl D:/data/train_with_codes.jsonl `
  --batch_size 2 `
  --lr 1e-7 `
  --num_epochs 3 `
  --speaker_name myvoice
```

脚本默认参数：

- `batch_size=2`
- `lr=2e-5`
- `num_epochs=3`
- `gradient_accumulation_steps=4`
- `mixed_precision="bf16"`
- `AdamW(model.parameters(), lr, weight_decay=0.01)`
- 梯度裁剪 1.0
- 每个 epoch 保存 checkpoint

### 2.6 推理

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "D:/output/qwen3-tts-ft/checkpoint-epoch-2",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

wavs, sr = tts.generate_custom_voice(
    text="今天最高气温三十一度。",
    language="Chinese",
    speaker="myvoice",
)
sf.write("out.wav", wavs[0], sr)
```

微调后的模型不需要再传 `ref_audio`，speaker embedding 已烘进 `codec_embedding`。这是实时场景降低首包延迟、提高并发的重要一步。

---

## 3. 数据要求

### 3.1 音频

- 24kHz、单声道、wav。
- 单条 2–10 秒。
- SNR > 20–25dB，无削波、无混响、无背景音乐。
- 保留自然停顿、呼吸、语气词、笑声、叹气；不要过度 VAD 裁剪或降噪。
- 目标说话人 3–10 小时起步：
  - 仅音色：30 分钟–1 小时可试，但稳定性和泛化有限。
  - 口音/新语言：5–10 小时起步，且不一定够。社区报告 7 小时孟加拉语微调后仍有明显印地语口音（issue #323）。

### 3.2 文本

- 逐字准确，包含语气词、重复、自我纠正。
- 数字、单位、英文、标点先规范化。
- 标点用于表达停顿和语气；不要把所有句子都改成书面语。
- 如果做情感控制，文本需要额外的风格标签/instruction。

### 3.3 数据划分

- 按“通话/会话”划分 train/dev/test，不要按句子随机切。
- 目标域 100–200 句 + 通用回归 100–200 句 + 压力集：
  - 短句、开头 1–2 秒
  - 长句
  - 数字、多音字、中英混
  - 情绪切换
  - 打断/语气词

### 3.4 情感/风格覆盖

建议至少覆盖：

- 中性、温柔、开心、难过、生气、惊讶、疲惫、压低声音/耳语
- 笑、叹气、呼吸
- 短回应：“嗯”“对”“好的”“哈哈”“是这样”
- 情绪过渡：平静 → 关切 → 安慰

每个风格都要有足够样本，不要只靠少量极端样本。

---

## 4. 训练配方与调参

### 4.1 学习率：官方文档不一致，必须自己扫

官方三处参数不一致：

- `sft_12hz.py` 默认 `lr=2e-5`
- `finetuning/README.md` 示例 `lr=2e-6`、`epochs=10`
- 一键脚本示例 `lr=2e-5`、`epochs=3`、`batch_size=2`

社区实测（issue #314）：

- `2e-6` 训完 SIM 反而比 zero-shot 低 0.03–0.06。
- 降到 `1e-8 ~ 1e-7` 后主观音色明显变好。
- 低 LR 下 loss 会停在 11–15 不降，但音频更好；**不要用 loss 选模型**。

推荐：

1. 用 10–20% 数据做 LR sweep：`{1e-8, 3e-8, 1e-7, 3e-7, 1e-6}`。
2. 每个 LR 只跑 1 epoch。
3. 用 holdout 的 SECS + 人耳 A/B 选择。
4. 最佳 LR 确定后，全量跑 2–3 epochs。
5. 每个 epoch 都保存，最后挑最佳 checkpoint。

### 4.2 显存与 batch

社区报告：

- RTX 3090 24GB：1.7B 全参 `batch_size=2` 可跑。
- RTX 5090 32GB：1.7B 全参 `batch_size=4` 可跑（7 小时日语数据，lr 4e-6，2 epochs）。
- 0.6B 可以适当翻倍。

脚本内 `gradient_accumulation_steps=4` 固定，所以有效 batch = `batch_size × 4 × GPU 数`。

### 4.3 冻结策略

- 冻结 `speaker_encoder`，并把它从 `AdamW` 中移除。
  - 官方脚本虽然 `.detach()` 了输出，但参数仍在 optimizer 中，`weight_decay=0.01` 会持续衰减它。
  - 结果是训练越久音色越漂，SIM 下降。
  - 参考 PR #351。
- 对于小数据，建议只训 talker 的上层/适配层，或使用 LoRA，减少灾难性遗忘。
- 官方没有 LoRA 支持；社区 issue #343 用 `r=16`、`lr=2e-6` 做过 LoRA，但短句开头仍有音色不稳。

### 4.4 LoRA 与全参的取舍

| 方式 | 优点 | 缺点 |
|---|---|---|
| 全参 SFT | 官方支持，效果上限高 | 小数据易过拟合/遗忘；显存高 |
| LoRA | 省显存、易迭代、保留基座能力 | 官方不支持，需自己改脚本；合并后才能零推理开销 |
| 蒸馏到 0.6B | 并发最好 | 工程复杂，TTS 蒸馏不是简单 logits 匹配 |

实时通话建议：

- 1.7B 先用全参或 LoRA 做质量上限。
- 部署前合并 LoRA，避免推理额外开销。
- 如果 1.7B 并发不够，用同一批数据训 0.6B，或做 1.7B → 0.6B 蒸馏。

---

## 5. 官方脚本已知问题与社区修复

> 状态截至 2026-11-12。main 分支仍未合入大部分修复，训练前需自行打补丁或使用审计过的 fork。

| 问题 | 影响 | 相关 PR/Issue | 状态 |
|---|---|---|---|
| 文本 embedding 没走 `text_projection` | 0.6B 维度报错；1.7B 训练/推理不一致 | #387, #384, #336 | open |
| `codec_0_labels` 被 shift 两次 | 模型预测 `t+2`，EOS 学不到，生成停不下来 | #387, #384, #278, #371 | open |
| `forward_sub_talker_finetune` hidden state 错一帧 | 15 个 residual codebook 对齐错 | #387, #384 | open |
| `speaker_encoder` 留在 AdamW 被 weight decay | 训练越久音色越漂，SIM 下降 | #351 | open |
| 不同 `ref_audio` 长度 | `collate_fn` 的 `torch.cat` 直接崩 | #381 | closed，未合并 |
| `language` 字段读了没用 | 多语言训练实际未生效 | #361 | open |
| `--init_model_path` 传 HF repo id | 保存 checkpoint 时 `copytree` 报错 | #384 | open |
| checkpoint 保存是整目录 copy + 全量 state_dict | 1.7B 每 epoch 约 3.5GB，主机 RAM 峰值约 2 倍模型 | #365 | 需关注 |
| 硬编码 `flash_attention_2` | Windows/ROCm 装不上 flash-attn 就起不来 | #373 | open |
| 单 micro-batch 超过 8192 帧 | code predictor 出现 NaN/非法访存 | #379 | open |
| `log_with="tensorboard"` 没给 `logging_dir` | 启动报错 | #286, #295 | 需处理 |

### 5.1 最低限度补丁清单

1. 冻结 `speaker_encoder` 并移出 optimizer（#351）。
2. 补 `model.talker.text_projection(...)`（#387）。
3. 修复 label double-shift 和 sub-talker 对齐（#387 最完整）。
4. 把 `attn_implementation` 改成 `sdpa`（#373，Windows 必需）。
5. 所有 JSONL 读写加 `encoding="utf-8"`（Windows 上默认 cp936 可能解码失败）。
6. 权重路径用本地目录，不用 repo id。
7. 如果要用多说话人/多语言/instruct，参考 #384 或社区 fork，但要先合并上面的修复。

### 5.2 训练前 smoke test

跑 1 个 epoch 的小数据，重点检查：

- 生成是否能正常结束（EOS 是否学到）。
- 短句和开头 1–2 秒音色是否正确。
- 长句后半段是否漂。
- 训练 loss 是否正常下降（但不要用 loss 选模型）。
- checkpoint 是否能被 `generate_custom_voice` 正常加载。

---

## 6. 实时通话与并发部署

### 6.1 官方 `qwen-tts` 包的限制

- `non_streaming_mode=False` **只是模拟流式文本输入**，不是真正的音频流式生成。
- `generate_*` 同步返回整段音频，不适合低延迟通话。
- 单实例多线程并发会严重退化；issue #350 有复现和讨论。
- 官方明确：流式/在线服务主要由 vLLM-Omni 社区推进（issue #10、#77）。

### 6.2 可选服务方案

| 方案 | 能力 | 风险/备注 |
|---|---|---|
| **vLLM-Omni** `vllm serve --omni` + stage config | 连续批处理、KV cache、官方生态 | 版本迭代快；Qwen README 仍写“离线为主”；有 batch>1 不稳定的社区报告 |
| [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming) | 真正 PCM 流式、声称约 6x 加速、支持 Windows | 实验性 fork，需审计、锁 commit |
| [rekuenkdr/Qwen3-TTS-streaming](https://github.com/rekuenkdr/Qwen3-TTS-streaming) | 两阶段延迟、Hann crossfade、`torch.compile` + CUDA graphs | 实验性；chunk 边界/重复问题有历史 |
| [HaujetZhao/Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) | llama.cpp/Vulkan/CUDA，RTX5050 上 RTF 约 0.3、首音约 300ms | 量化质量、并发和功能限制 |
| 自研引擎 | 最大并发、最低延迟 | 工程量大：continuous batching、paged KV、CUDA graphs、编译 codec |

推荐顺序：

1. 先在目标 GPU 上压测 vLLM-Omni 和 dffdeeq fork。
2. vLLM-Omni batch>1 稳定则优先 vLLM-Omni。
3. 不稳定则先用“batch=1 + 多实例 + 负载均衡”保底，再考虑自研 batching。
4. 赶时间可先用 DashScope 实时 API 做产品验证，自托管并行推进。

### 6.3 延迟优化

- **首句拆分**：LLM 产出 5–12 个字就送 TTS，不要等整段。
- `emit_every_frames=4`，不要每 8 帧才发。
- `torch.compile` talker forward + CUDA graphs。
  - 社区在 RTX 5090 上把 TTFT 从约 0.8s 降到约 0.17s。
- `compile_codec`：社区报告 batch decode 3–4x 加速（PR #191/#316）。
- FP8/INT8/W4A16 量化；必须测 SECS/UTMOS，0.6B 对量化更敏感。
- 烘 speaker embedding、缓存 voice clone prompt、缓存文本规范化、预合成固定话术。
- 音频解码/Opus 编码放 CPU 或独立线程，GPU 只跑模型。
- 打断时立即 abort 请求、释放 KV cache。

### 6.4 并发优化

- 连续批处理，不要静态 batch。
- 1.7B 用数据并行多副本，不要 tensor parallel；TP 会增加延迟。
- 0.6B 做高并发档；1.7B 做质量档。
- 从 1.7B 蒸馏到 0.6B，或用同一批数据直接训 0.6B，对比质量差距。
- 控制 `max_new_tokens`，按通话句子长度限制；不要让单请求生成几分钟。
- 按 token 长度分桶，避免长音频拖垮整个 batch。
- 监控 KV cache、排队数、TTFT、RTF；设置准入控制。
- 预留 1.5–2x 容量余量。

### 6.5 并发估算

不要用单请求 RTF 直接乘。用聚合吞吐：

```
可服务并发 ≈ GPU 每秒能生成的音频秒数 / 1.0
```

例如压测得到聚合吞吐 12 秒音频/秒，理论上可支撑约 12 路实时通话（RTF≈1）；要保证 RTF<0.7，就按 8–9 路规划，再留余量。同时检查 KV cache 是否先爆。

压测必须用真实通话长度（3–15 秒/句，30–60 秒会话），不要只用短句。

社区报告（未经官方验证）：

- RTX 3090 上自定义引擎，0.6B 约 128 并发、1.7B 约 96 并发，TTFA 约 300ms，RTF<0.7。
- vLLM-Omni 在 A800 上 batch>1 有 IndexError 报告，batch=1 + 多实例更稳。
- 只能作为量级参考，必须自己压测。

### 6.6 流式的坑

- `non_streaming_mode=False` 可能导致语速漂移（issue #239）；官方后续把默认改回 `True`（PR #362）。
- chunk 拼接会有爆音/click，需要 crossfade/overlap。
- 流式下 repetition penalty 可能引发重复朗读，需要单独调，部分 fork 建议流式默认关闭。
- 首包太快但 prosody 不完整，听感会“一顿一顿”；首块 5–12 字比较平衡。

### 6.7 实时链路建议

```
LLM 流式输出
  → 句子/短语切分（首块 5–12 字）
  → 风格导演（标点、语气词、情绪标记）
  → TTS 流式生成
  → Opus 编码
  → 抖动缓冲
  → 客户端
```

- 预合成固定话术：问候、确认、等待、结束语。
- 打断：VAD 检测到用户说话 → abort TTS → 释放 KV → 切回 ASR。
- 抖动缓冲 60–120ms；丢包隐藏。
- 如果最终走 PSTN/8k，在边缘下采样；训练仍建议 24k。
- 端到端 SLA 要把 ASR、LLM、网络都算进去，不能只看 TTS。

---

## 7. 音色与情感控制方案

### 7.1 固定人设（推荐先做）

- 用目标说话人的真实对话数据 SFT。
- 情感来自：
  - 数据本身的表达性
  - LLM 文本的措辞、标点、断句
  - 文本导演模块：拆分长句、加停顿、加语气词
- 优点：延迟最低，最像真人，工程简单。
- 缺点：不能随时切“现在很生气”，情绪范围受数据限制。

### 7.2 动态情绪控制（第二阶段 POC）

- 控制标记：`[温柔]`、`[开心]`、`[生气]`、`[疲惫]`、`[耳语]`、`[笑]`、`[叹气]`。
- 训练数据按风格打标；训练时 10–20% 概率丢弃标记，保证无标记也能说。
- 推理时由 LLM/规则“导演”插入标记。
- 官方 Base SFT 没有 instruct 监督；转成 `custom_voice` 后接口能传 `instruct`，不代表模型学会了。
- 社区 fork 的 `--instruct_model` 在尝试类似能力，但继承了官方对齐 bug，需要验证。

### 7.3 两条路的取舍

| 方案 | 音色 | 情感 | 延迟 | 风险 |
|---|---|---|---|---|
| 固定人设 SFT + 文本导演 | 自由 | 整体风格，有限动态 | 低 | 低 |
| 动态 control token/instruct | 自由 | 可切换 | 中 | 中高，需 POC |
| CustomVoice + instruct | 仅 9 个内置音色 | 好 | 低 | 低，但音色不自由 |
| CustomVoice → 声音转换 | 自由 | 好 | 高 | 转换伪影/额外延迟 |

---

## 8. 评测方案

### 8.1 离线

- SECS 说话人相似度（CAM++/ECAPA）。
- CER/WER（ASR 回测）。
- UTMOS/DNSMOS。
- 情绪分类准确率、F0 方差、停顿分布。
- MOS/CMOS，至少 5 人盲测。

必须单独测：

- 短句和开头 1–2 秒（issue #343 的典型问题）。
- 长句后半段是否漂。
- EOS 是否正常，会不会无限生成。
- 通用集 CER 是否回退 >5%。

### 8.2 在线

- TTFA（首音频包）p50/p95/p99。
- RTF、并发路数、排队时长。
- 打断响应、丢包/抖动。
- “是否以为是真人”的盲测。
- 通话完成率、用户挂断率、投诉率。

### 8.3 验收门槛示例

- 目标域 CER 相对基线下降 ≥20%。
- 通用集 CER 回退 ≤5%。
- SECS ≥0.85（按你的说话人验证模型校准）。
- TTFA p95 <300ms。
- RTF p95 <0.7。
- 盲测“像真人”比例达到产品目标。

---

## 9. 推荐落地顺序

### Phase 0（1 周）：压测基线

- 用 0.6B/1.7B CustomVoice + zero-shot clone，在目标 GPU 上测 TTFA、RTF、并发。
- 用 vLLM-Omni 和 dffdeeq fork 各跑一遍。
- 定 SLO：TTFA p95 <300ms，RTF p95 <0.7，单卡并发 N 路。

### Phase 1（1–2 周）：数据

- 收集 3–10 小时目标说话人真实对话。
- 清洗、切分、转写、按会话划分。
- 保留呼吸/停顿/语气词，不要过度降噪。

### Phase 2（1–2 周）：1.7B 音色 SFT

- 修官方脚本 bug。
- LR sweep，1–3 epochs。
- 评估 SECS/CER/MOS，重点听短句和开头。
- 烘 speaker embedding，测推理不再传 `ref_audio`。

### Phase 3：情感

- 先做文本导演（标点、断句、语气词），A/B 看是否够。
- 不够再上 control token/instruct POC，只在 1.7B 上做。

### Phase 4：并发

- 0.6B 同数据 SFT，或从 1.7B 蒸馏。
- 量化 + vLLM/自研 batching。
- 压测选型：质量档 1.7B，并发档 0.6B。

### Phase 5：上线 A/B

- 真实通话盲测“是否像真人”。
- 收集 badcase：短句、情绪切换、打断、长句、噪音。
- 迭代数据和文本导演，不一定继续加训练。

---

## 10. 风险与合规

1. **官方微调脚本有已知 bug**：不修会白训，尤其是 EOS/对齐/speaker encoder 衰减。
2. **官方没有真正的流式服务**：社区 fork 是实验性的，生产要锁版本、做回归。
3. **vLLM-Omni batching 稳定性**：有 batch>1 报错/变慢的报告，必须自己压测。
4. **1.7B 并发可能不够**：如果单卡并发是硬指标，0.6B/蒸馏/量化是必选项。
5. **动态情绪控制不是官方能力**：Base SFT 只给音色，instruct 需要额外训练。
6. **合规**：真人音色克隆需要授权；合成语音按当地法规标识；保留审计日志。

---

## 11. 待确认信息

- GPU 型号/显存，单机还是多机？
- 目标单卡并发路数、TTFA 和 RTF 的 SLO？
- 音色是固定品牌人设，还是需要多音色？
- 情感是“整体人设风格”还是“随时切换喜怒哀乐”？
- 现有数据量、是否真实通话录音、是否有授权？
- 训练在 Windows 本地，还是远程 Linux？Windows 上 flash-attn/vLLM 支持有限，建议训练放 Linux。

---

## 12. 参考资料

### 官方

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS finetuning README](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)
- [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- [Qwen3-TTS-Tokenizer-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz)
- [Qwen3-TTS 技术报告 PDF](https://github.com/QwenLM/Qwen3-TTS/blob/main/assets/Qwen3_TTS.pdf)

### 微调问题与修复

- [Issue #314 — SFT 后 SIM 反而下降](https://github.com/QwenLM/Qwen3-TTS/issues/314)
- [Issue #343 — 短句/开头音色不稳](https://github.com/QwenLM/Qwen3-TTS/issues/343)
- [Issue #323 — 新语言微调后的口音偏移](https://github.com/QwenLM/Qwen3-TTS/issues/323)
- [PR #351 — speaker_encoder 被 weight decay](https://github.com/QwenLM/Qwen3-TTS/pull/351)
- [PR #384 — 0.6B/对齐/多说话人修复](https://github.com/QwenLM/Qwen3-TTS/pull/384)
- [PR #387 — 微调对齐修复与回归测试](https://github.com/QwenLM/Qwen3-TTS/pull/387)
- [PR #381 — 不同长度 ref_audio 修复](https://github.com/QwenLM/Qwen3-TTS/pull/381)
- [PR #373 — attention 实现可配置](https://github.com/QwenLM/Qwen3-TTS/pull/373)
- [PR #379 — 长 batch NaN 修复](https://github.com/QwenLM/Qwen3-TTS/pull/379)
- [PR #361 — language 字段实际生效](https://github.com/QwenLM/Qwen3-TTS/pull/361)
- [PR #278 — label 对齐修复](https://github.com/QwenLM/Qwen3-TTS/pull/278)

### 流式与并发

- [Issue #10 — Streaming audio](https://github.com/QwenLM/Qwen3-TTS/issues/10)
- [Issue #77 — Streaming support inference](https://github.com/QwenLM/Qwen3-TTS/issues/77)
- [Issue #350 — 多并发推理不支持](https://github.com/QwenLM/Qwen3-TTS/issues/350)
- [Issue #148 — vLLM-Omni 在 A800 上很慢](https://github.com/QwenLM/Qwen3-TTS/issues/148)
- [Issue #171 — Inference is too slow](https://github.com/QwenLM/Qwen3-TTS/issues/171)
- [PR #362 — non_streaming_mode 默认改回 True](https://github.com/QwenLM/Qwen3-TTS/pull/362)
- [PR #380 — batched voice cloning 的 CUDA graph decoding](https://github.com/QwenLM/Qwen3-TTS/pull/380)
- [PR #191 — compile_codec 加速 batch decoding](https://github.com/QwenLM/Qwen3-TTS/pull/191)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)

### 社区方案

- [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming)
- [rekuenkdr/Qwen3-TTS-streaming](https://github.com/rekuenkdr/Qwen3-TTS-streaming)
- [HaujetZhao/Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF)
- [vspeech/Qwen3-TTS-Train](https://github.com/vspeech/Qwen3-TTS-Train)

---

## 13. 版本与时效声明

- 本文基于 2026-11-12 的 GitHub/HuggingFace 信息整理。
- PR 状态、vLLM-Omni 能力和社区 fork 都可能快速变化；落地前请以锁定 commit/版本重新验证。
- 社区性能数字（并发数、RTF、TTFT）均为个人报告或 fork 自述，未经官方基准验证。
