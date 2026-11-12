# CosyVoice 实时通话场景调研

> 调研日期：2026-11-12  
> 目标场景：实时通话、音色 + 说话风格/情感、拟人度高、同算力下更高并发  
> 调研范围：CosyVoice 1/2/3 官方开源模型、官方微调脚本、instruct/情感控制、流式与并发部署、已知问题  
> 官方仓库基线：`QwenAudio/CosyVoice`（原 `FunAudioLLM/CosyVoice`），commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`（2026-05-26）  
> 说明：本文基于该 commit 的 README、`examples/libritts/*`、`cosyvoice/cli/*`、`runtime/triton_trtllm/*` 和公开 issue 整理。版本、issue 状态和模型卡会变化，落地前需要复测。  
> **本项目选型更新：确定采用 CosyVoice3。** v3 专用的训练、推理和补丁步骤见第 15 节；第 9 节的 CosyVoice2 优先顺序仅作对照。

---

## 0. 结论摘要

1. **CosyVoice 是当前开源 TTS 里“微调 + 流式 + 并发部署”链路最完整的一档（至少是之一）。**
   - 官方同时提供 LLM / flow / HiFiGAN 三段式训练脚本、SFT/zero-shot/cross-lingual/instruct 推理、FastAPI/gRPC 服务、vLLM 和 Triton + TensorRT-LLM 运行时。
   - 这比 Qwen3-TTS 目前主要依赖社区补流式和并发要成熟。

2. **通用调研结论：v2 的 SFT 推理更稳；本项目已确定改用 CosyVoice3，v3 需要第 15 节的补丁。**
   - CosyVoice2 的 `inference_sft` 是开箱可用的，`spk2info.pt` 补上即可。
   - CosyVoice3 音质/相似度/方言更好；SFT 训练脚本存在，但原版 `inference_sft` 会断言文本中必须含 `<|endofprompt|>`，且 `frontend_sft` 不注入 instruct；按第 15 节补丁后，可走“v3 SFT + 常量/逐句 instruct”，作为本项目主路线。

3. **官方微调是全参 SFT，不是 LoRA。**
   - 官方示例循环训练 `llm`、`flow`、`hifigan` 三个模块，但脚本注释又写“We only support llm training for now”。
   - 实践上先只训 LLM，发音、停顿、语速和文本域适配基本就能收敛；音色和细腻情绪主要由 speaker embedding + flow 决定，flow 视音质/情绪需要再训；hifigan 一般不动。
   - 没有官方 LoRA/Adapter 脚本；要做 LoRA 需要自己改训练代码和导出逻辑。

4. **情感 instruct 是固定指令集，不是自由文本。**
   - `cosyvoice/utils/common.py` 的 `instruct_list` 只覆盖 17 种中文方言、大小声、快慢、开心/伤心/生气、小猪佩奇风格、机器人风格。
   - “温柔”“疲惫”“耳语”“笑”“叹气”等不在官方清单里，直接写自然语言描述效果不稳定。
   - 要动态情绪，需要自定义数据 + 逐句 instruct 标签 + 改 `prepare_data.py` / 推理前端；这是一条 POC 路线，不是开箱能力。

5. **流式是真的 bi-streaming，但 150ms 不是开箱即得。**
   - README 宣称 bi-streaming 延迟低至 150ms，但没有给出该数字的硬件、并发、首包口径和优化条件。
   - 官方 Python 代码首块要等 `token_hop_len(25) + pre_lookahead_len(3)` 个 token；25Hz 下 28 个 token 约 1.12 秒音频，再叠加 LLM 和 token2wav 时间。
   - 社区实测未优化时首包约 1.1–2 秒；TensorRT 优化后可到约 450–600ms；NVIDIA Triton + TRT-LLM 的 CosyVoice2 DiT 在 L20、并发 1 时首块约 197ms，CosyVoice3 在并发 4 时约 750ms。

6. **并发瓶颈在 token2wav / vocoder，不在 LLM。**
   - vLLM 只加速 LLM；token2wav 仍是逐请求执行，没有连续批处理。
   - Triton + TensorRT-LLM 是官方推荐的高并发路径；CosyVoice2 还可以用 Step-Audio2 的 DiT token2wav 做分离部署。
   - CosyVoice3 官方 offline 数据：LLM 从 batch 1 的 13.21s 降到 batch 16 的 2.12s，但 token2wav 始终 5.7–6.3s，端到端 RTF 只从 0.1091 降到 0.0501。这说明只堆 LLM 批处理收益有限。

7. **必须提前处理的坑：**
   - `spk2info.pt` 不会在训练后自动生成；`inference_sft` 需要它。
   - CosyVoice3 `inference_sft` 的 `<|endofprompt|>` 断言。
   - `zero_shot_spk_id` 会缓存注册时的 instruct，调用 `inference_instruct2` 时传入的新 instruct 会被忽略。
   - 官方 `prepare_data.py` 只支持一个全局 instruct 字符串，不能逐句打情感标签。
   - `examples/libritts/cosyvoice3/run.sh` 的 `average_model.py` 路径写成 `exp/cosyvoice`，与训练输出 `exp/cosyvoice3` 不一致；默认 `stop_stage=3` 也不会跑到训练阶段。
   - CosyVoice3 的 TensorRT `fp16=True` 有 NaN 报告，运行时用 `fp16=False`。
   - vLLM 只支持 0.11.x+ 或 0.9.0，0.10.x 未测试。

8. **给本项目的建议（已确定采用 CosyVoice3）：**
   - 第一阶段：用 `Fun-CosyVoice3-0.5B-2512` 做 LLM-only SFT，并打第 15 节的 `frontend_sft` / `inference_sft` instruct 补丁；CosyVoice2 只作为对照基线，不作为主方案。
   - 第二阶段：固定人设用“补丁版 `inference_sft` + 常量 instruct”，推理不传参考音频，首包最低；需要动态情绪时，训练数据逐句 instruct，推理把情绪 instruct 传给补丁版 `inference_sft`。
   - 第三阶段：上 Triton + TensorRT-LLM 的 CosyVoice3 runtime，`fp16=False`；按真实通话长度压 p95/p99，不要用离线 RTF 直接推算并发。
   - 情感先做“固定人设 + 文本导演 + 表达性数据”；动态情绪 instruct 作为第二阶段能力，不要作为首发依赖。完整步骤见第 15 节。

---

## 1. 官方模型与能力边界

### 1.1 已发布模型

| 模型 | 参数 | 输出/ token 率 | 能力 | 微调/部署现状 | 建议 |
|---|---:|---|---|---|---|
| `Fun-CosyVoice3-0.5B-2512` | 0.5B | 24kHz / 25Hz | 9 语言、18+ 中文方言、zero-shot 克隆、bi-streaming、instruct、发音修复、文本规范化 | 有训练/推理脚本；SFT 推理有坑；Triton/TRT-LLM 支持 | 音质和相似度上限，适合做目标模型 |
| `Fun-CosyVoice3-0.5B-2512-RL` | 0.5B | 24kHz / 25Hz | 同上，RL 后处理版本 | 同上 | 内容准确率更高，情感/音色需自己 A/B |
| `CosyVoice2-0.5B` | 0.5B | 24kHz / 25Hz | zero-shot、cross-lingual、instruct2、流式 | 官方 SFT 最成熟；vLLM、Triton/TRT-LLM、gRPC/FastAPI 都有 | **首选 SFT 基线** |
| `CosyVoice-300M` | 0.3B | 22.05kHz | 基础 zero-shot/克隆 | v1 训练脚本；生态老 | 只建议低成本 POC |
| `CosyVoice-300M-SFT` | 0.3B | 22.05kHz | 内置 SFT 音色 | 可直接 `inference_sft` | 验证流程可用，不适合自定义音色 |
| `CosyVoice-300M-Instruct` | 0.3B | 22.05kHz | 内置音色 + instruct | v1 `inference_instruct` | 只能验证 instruct 交互 |

补充：

- CosyVoice3 和 CosyVoice2 都是 24kHz 输出，25Hz speech token；v1 是 22.05kHz。
- CosyVoice3 还支持中文拼音、英文 CMU 音素的发音修复（pronunciation inpainting），对姓名、专有名词、数字播报很有用。
- CosyVoice3 宣称原生文本规范化，不依赖传统前端；但训练文本和推理文本仍应使用同一套规范化策略，否则会引入 CER 波动。

### 1.2 架构与关键参数

以 CosyVoice3 配置 `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml` 为准：

| 项 | 值 | 说明 |
|---|---:|---|
| `sample_rate` | 24000 | 输出 24kHz |
| `llm_input_size` / `llm_output_size` | 896 / 896 | Qwen2Encoder 0.5B 级别 |
| `spk_embed_dim` | 192 | CAM++ speaker embedding |
| `token_frame_rate` | 25 | 每秒 25 个 speech token |
| `token_mel_ratio` | 2 | 25Hz token 对应 50Hz mel |
| `chunk_size` | 25 | 流式 chunk，25 token = 1 秒音频 |
| `speech_token_size` | 6561 | speech token 词表 |
| flow | 22 层 causal DiT | v3 用因果 DiT flow matching |
| vocoder | CausalHiFT | 因果 HiFiGAN 变体，支持流式 |

CosyVoice2 同样是 24kHz / 25Hz / 896 hidden / 192 speaker embedding / 6561 speech token；flow 是 UNet 类 flow matching，vocoder 是 HiFiGAN。v1 是 22.05kHz / 1024 hidden / 4096 speech token。

训练/推理链路：

```
文本 tokenizer
  → Qwen2Encoder LLM（文本 token + speaker embedding + speech token）
  → flow matching（speech token → mel）
  → CausalHiFT / HiFiGAN（mel → waveform）
```

流式时 LLM 先生成 speech token，token2wav 按 chunk 消费；LLM 和 token2wav 可以流水线并行，但共享同一张 GPU 时会互相抢资源。

### 1.3 官方微调边界

官方示例在 `examples/libritts/cosyvoice3/run.sh`、`examples/libritts/cosyvoice2/run.sh`、`examples/libritts/cosyvoice/run.sh`：

- **全参 SFT**，没有 LoRA/Adapter。
- 数据格式是 LibriTTS 风格：`wav.scp`、`text`、`utt2spk`、`spk2utt`，可选 `instruct`、`utt2embedding.pt`、`spk2embedding.pt`、`utt2speech_token.pt`。
- 训练入口 `cosyvoice/bin/train.py`，`--model` 可选 `llm` / `flow` / `hifigan`。
- CosyVoice3 的 `run.sh` 注释写“We only support llm traning for now”，但代码仍循环三个模型；实际以 LLM 为主，flow/hifigan 是否可训要以你的模型目录和显存为准。
- CosyVoice2 另有 `run_dpo.sh`，可以做 DPO/GRPO 类偏好优化；这不是音色 SFT 的主线，属于后处理调优。
- 官方没有“情感微调”专用脚本；v3 的 instruct 是训练序列的一部分，但官方只给了一个全局 instruct 示例，逐句情感标签要自己准备。

---

## 2. 官方微调流程（以 CosyVoice3 为例）

### 2.1 环境

```bash
git clone --recursive https://github.com/QwenAudio/CosyVoice.git
cd CosyVoice

conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice
pip install -r requirements.txt

# 如果 sox 报错
# Ubuntu: sudo apt-get install sox libsox-dev
# CentOS: sudo yum install sox sox-devel
```

模型下载（ModelScope 或 HuggingFace 二选一）：

```python
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice2-0.5B',
                  local_dir='pretrained_models/CosyVoice2-0.5B')
```

```python
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('FunAudioLLM/CosyVoice2-0.5B',
                  local_dir='pretrained_models/CosyVoice2-0.5B')
```

### 2.2 数据准备

官方 `local/prepare_data.py` 假设 LibriTTS 目录结构（`*/*/*.wav` + `*.normalized.txt`），并生成：

```
data/train/
  wav.scp
  text
  utt2spk
  spk2utt
  instruct          # 可选
```

CosyVoice3 官方示例给所有句子写同一个 instruct：

```bash
python local/prepare_data.py \
  --src_dir /path/to/LibriTTS/train-clean-100 \
  --des_dir data/train \
  --instruct "You are a helpful assistant.<|endofprompt|>"
```

注意：`prepare_data.py` 的实现是“给每条 utterance 写同一行 instruct”，不是逐句读取标签。自定义数据建议直接生成这五个文件，或训练前覆盖 `data/train/instruct`：

```python
# 逐句 instruct：key 是 utt id，value 是完整 instruct
with open('data/train/instruct', 'w', encoding='utf-8') as f:
    for utt, instruct in utt2instruct.items():
        f.write(f'{utt} {instruct}\n')
```

`make_parquet_list.py` 会在存在 `instruct` 文件时写入 parquet 的 `instruct` 列；如果一部分句子有、一部分没有，batch 组装时只有“全部都有 instruct”才会带 `instruct_token`（`processor.py` 里用 `.all()` 判断），所以要么全有、要么全无。

### 2.3 提取 speaker embedding 和 speech token

```bash
# CAM++ speaker embedding
python tools/extract_embedding.py \
  --dir data/train \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx

# speech token
python tools/extract_speech_token.py \
  --dir data/train \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx
```

产出：

```
data/train/utt2embedding.pt
data/train/spk2embedding.pt
data/train/utt2speech_token.pt
```

官方注释说现在支持在线特征提取，不预提取也能训，但训练速度会明显变慢；正式训练建议预提取。

### 2.4 生成 parquet

```bash
python tools/make_parquet_list.py \
  --num_utts_per_parquet 1000 \
  --num_processes 10 \
  --src_dir data/train \
  --des_dir data/train/parquet
```

训练列表：

```bash
cat data/{train-clean-100,train-clean-360,train-other-500}/parquet/data.list > data/train.data.list
cat data/{dev-clean,dev-other}/parquet/data.list > data/dev.data.list
```

### 2.5 训练

CosyVoice3 的官方命令（简化）：

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=4

torchrun --nnodes=1 --nproc_per_node=$num_gpus \
  --rdzv_id=1986 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config conf/cosyvoice3.yaml \
  --train_data data/train.data.list \
  --cv_data data/dev.data.list \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model llm \
  --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/llm.pt \
  --model_dir exp/cosyvoice3/llm/torch_ddp \
  --tensorboard_dir tensorboard/cosyvoice3/llm/torch_ddp \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 100 \
  --pin_memory \
  --use_amp
```

训完 LLM 后，如果确实需要，再把 `--model` 换成 `flow`，checkpoint 用 `flow.pt`。**不建议训 `hifigan`**：官方发布目录通常只有 `hift.pt`（生成器），没有完整的 GAN checkpoint；`run.sh` 写的 `hifigan.pt` 通常不存在，`train.py` 找不到只会 warning，然后从头初始化，等于 GAN 从零训。

### 2.6 模型平均与导出

```bash
python cosyvoice/bin/average_model.py \
  --dst_model exp/cosyvoice3/llm/torch_ddp/llm.pt \
  --src_path exp/cosyvoice3/llm/torch_ddp \
  --num 5 \
  --val_best

python cosyvoice/bin/export_jit.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
python cosyvoice/bin/export_onnx.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
```

补充：`CosyVoice3` 的 Python 推理类没有 `load_jit` 参数（只有 `load_trt` / `load_vllm`），所以 JIT 产物主要给支持 JIT 的自定义 runtime 使用；普通 Python 推理走 `load_trt`、`load_vllm` 或 ONNX。CosyVoice2 才支持 `load_jit`。

注意两个路径问题：

1. 官方 `run.sh` 的 `average_model.py` 写的是 `exp/cosyvoice/$model/...`，但训练输出是 `exp/cosyvoice3/...`；直接跑会找不到文件。
2. `export_jit.py` / `export_onnx.py` 读的是 `--model_dir`，需要先把训练好的 `llm.pt` / `flow.pt` / `hift.pt` 复制进一个完整的模型目录，目录里还要有配置、tokenizer、onnx、`CosyVoice-BlankEN` 等。

### 2.7 推理：`spk2info.pt` 必须自己补

`frontend_sft` 的实现是：

```python
embedding = self.spk2info[spk_id]['embedding']
model_input = {
    'text': tts_text_token,
    'text_len': tts_text_token_len,
    'llm_embedding': embedding,
    'flow_embedding': embedding,
}
```

所以 `inference_sft(tts_text, spk_id)` 要求模型目录下有 `spk2info.pt`，且包含对应 `spk_id` 的 `embedding`。训练不会自动生成它。

最简单的做法是用目标说话人的参考音频注册一次：

```python
from cosyvoice.cli.cosyvoice import CosyVoice2

model = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False, load_trt=False, fp16=False
)

model.add_zero_shot_spk(
    prompt_text='参考音频对应的文本',
    prompt_wav='ref.wav',
    zero_shot_spk_id='my_spk'
)
model.save_spkinfo()   # 写入 model_dir/spk2info.pt
```

也可以从 `spk2embedding.pt` 手工构造：

```python
import torch

spk2emb = torch.load('data/train/spk2embedding.pt', map_location='cpu')
spk2info = {
    spk: {'embedding': torch.tensor(emb, dtype=torch.float32).unsqueeze(0)}
    for spk, emb in spk2emb.items()
}
torch.save(spk2info, 'pretrained_models/CosyVoice2-0.5B/spk2info.pt')
```

`frontend_sft` 只用到 `embedding`，所以手工构造足够；`add_zero_shot_spk` 会额外缓存 prompt text、prompt speech token、prompt feat，适合 zero-shot/instruct2。

---

## 3. 数据要求

### 3.1 音频

| 项 | 建议 |
|---|---|
| 采样率 | 源音频 16kHz 以上；训练会重采样到 24kHz |
| 声道 | 单声道 |
| 时长 | 配置允许 1–60s（`filter.max_length=6000`，按 10ms 帧算 60s）；**建议 3–15s**，和推理句子长度接近 |
| 文本长度 | `token_max_length=200`，约 200 个文本 token；过长会被过滤 |
| 音质 | 低底噪、无混响、无削波、同一麦克风/房间 |
| 情感 | 目标人设的真实表达，不要全是平静朗读 |
| 数据量 | 音色 SFT 约 1–3 小时；风格/情感覆盖建议 5–20 小时；以同说话人、同风格为准 |
| 参考音频 | 3–10s 最佳；`_extract_speech_token` 断言参考音频 ≤30s |
| 划分 | 按录音场次/日期切分，不要把同一段录音随机切碎后同时进 train/dev |

注意：`flow` / `hifigan` 的 GAN 训练有 `truncate_length: 24960`，24kHz 下约 1.04s，会把长音频裁成短片段；LLM 训练用完整句子。flow 微调时不要期待它学长距离韵律。

### 3.2 文本

- 使用和推理一致的文本规范化：数字、单位、日期、英文缩写、专有名词。
- 标点要保留，断句会影响韵律。
- 中文/英文混读要统一规则；CosyVoice3 支持原生规范化，但训练文本仍建议先规范化。
- 姓名、地名、品牌名可以用发音修复（拼音/CMU）在推理时纠正；训练数据里也可以加少量标注样本。
- 如果做逐句 instruct，instruct 文本本身也要规范化，并包含 `<|endofprompt|>`。

### 3.3 情感/风格覆盖

- 固定人设：不需要逐句情感标签，情感来自数据本身的表达 + 文本措辞 + 标点/停顿。
- 动态情绪：每个目标情绪至少 50–100 句，覆盖不同文本长度；同一句话最好有多种情绪版本，帮助模型解耦内容和风格。
- instruct 标签要和官方清单对齐；自定义标签（“温柔”“疲惫”）需要自己扩词表/训练，否则模型没见过。
- 所有样本要么都有 instruct，要么都没有；混合会导致 batch 里 `instruct_token` 整个缺失。

---

## 4. 训练配方与调参

### 4.1 必改配置

`examples/libritts/cosyvoice3/conf/cosyvoice3.yaml` / `cosyvoice2.yaml`：

```yaml
padding:
    use_spk_embedding: True   # 官方注释：change to True during sft

train_conf:
    optim: adam
    optim_conf:
        lr: 1e-5              # 官方注释：change to 1e-5 during sft
    scheduler: constantlr     # 官方注释：change to constantlr during sft
    scheduler_conf:
        warmup_steps: 2500
    max_epoch: 200
    grad_clip: 5
    accum_grad: 2
```

- `use_spk_embedding: True`：训练时用 speaker 级平均 embedding，和 `inference_sft` 的 `spk2info['embedding']` 对齐。若走 zero-shot/instruct2，推理用的是参考音频的 utterance embedding，建议对比 `False` 的效果。
- `lr: 1e-5` + `constantlr`：SFT 的官方推荐；不要用预训练的大学习率。
- `warmup_steps: 2500`：小数据下可能比总步数还大，建议按 `总步数 × 5–10%` 调小。
- `max_epoch: 200`：只是上限；小数据通常 1–5 个 epoch 就过拟合。
- `max_frames_in_batch: 2000`：动态 batch，按帧数而非样本数；显存不足就降这个值或加 `accum_grad`。
- `use_amp`：官方脚本默认开；如果遇到 loss NaN，先关 AMP 做 smoke test。

### 4.2 训哪些模块

| 模块 | 作用 | 建议 |
|---|---|---|
| `llm` | 文本 → speech token，决定音色、韵律、内容 | **先只训这个**；1–3 小时数据即可明显靠近目标音色 |
| `flow` | speech token → mel，决定音质/细节 | 音色已经像但音质/气息不对时再训；小数据容易过拟合 |
| `hifigan` | mel → waveform | 一般不动；官方发布目录只有 `hift.pt`，从零训风险高 |

社区常见做法是只替换 `llm.pt`，flow/hifigan 用预训练。这样推理最快、风险最低，也避免 GAN 训练不稳定。

### 4.3 过拟合与 checkpoint 选择

- **不要只看 validation loss。** TTS 的 val loss 下降不等于听感变好；LLM 很容易在几十步内记住音色，然后开始损失韵律和稳定性。
- 每 1–2 个 epoch 导出 checkpoint，用固定 50–100 句测试集做：
  - CER/WER（ASR 回读）
  - SECS / speaker similarity
  - UTMOS / DNSMOS
  - 人工盲听（自然度、情感、是否像目标人）
  - 重复、漏字、吞字、爆音、长句崩溃
- 选择“听感最好 + SECS 高 + CER 低”的 checkpoint，不要选 val loss 最低的。
- 小数据建议 1–5 epoch；如果 1 epoch 就已经很像，优先早停。
- 学习率扫描：`5e-6 / 1e-5 / 2e-5`，固定其他参数，各训 1–2 epoch 对比。CosyVoice 官方只给 1e-5，不是万能值。

### 4.4 显存与并行

- 0.5B LLM 在 24GB 卡上可以训，但 `max_frames_in_batch=2000` 可能偏大；先降到 1000–1500 试。
- flow / hifigan 的 GAN 训练更吃显存，建议 40GB+；24GB 需要更小 batch/片段。
- 多卡用 `torchrun` + DDP；不要用 tensor parallel 做训练，收益低且增加复杂度。
- `--num_workers`、`--prefetch` 按 CPU 和 IO 调；parquet 在本地 SSD 上更稳。
- 训练前先跑 20–50 step 的 smoke test，确认 loss 有限、能保存、能推理，再长训。

---

## 5. 官方脚本已知问题与补丁

### 5.1 问题清单

| # | 问题 | 证据 | 影响 | 处理 |
|---:|---|---|---|---|
| 1 | CosyVoice3 `inference_sft` 断言文本必须含 `<|endofprompt|>` | `cosyvoice/llm/llm.py`：`assert 151646 in text` | 直接调用 `inference_sft` 会失败；`frontend_sft` 不会自动加 EOP | 生产改用 `inference_instruct2` + 固定参考；或给 v3 的 SFT 前端补 instruct/EOP |
| 2 | CosyVoice3 `frontend_sft` 不注入 instruct | `frontend.py:frontend_sft` 只返回 text + embedding | 训练时如果带了 instruct，推理 SFT 时模型看不到，音色/风格会漂 | 改 `frontend_sft` / `inference_sft` 支持 instruct；否则不要用 SFT 路径 |
| 3 | `zero_shot_spk_id` 会忽略新的 instruct | `frontend_zero_shot` 在 id 非空时直接 `model_input = {**self.spk2info[id]}` | `inference_instruct2(..., zero_shot_spk_id='x')` 永远用注册时的 instruct，动态情绪失效 | 每个 instruct 预注册一个 speaker id；或打补丁覆盖 `prompt_text` |
| 4 | `prepare_data.py` 只支持一个全局 instruct | `prepare_data.py` 对每条 utt 写同一个 `args.instruct` | 不能逐句打情感/风格标签 | 自己生成 `instruct` 文件，或改脚本 |
| 5 | 训练后没有 `spk2info.pt` | `frontend_sft` 直接索引 `self.spk2info[spk_id]` | `inference_sft` KeyError | `add_zero_shot_spk` + `save_spkinfo`，或手工构造 |
| 6 | `run.sh` 默认不训练 | `stage=-1; stop_stage=3`，训练在 stage 5 | 直接 `bash run.sh` 只做数据准备 | 改 `stop_stage` 或分段跑 |
| 7 | `run.sh` average 路径不匹配 | 训练写 `exp/cosyvoice3/...`，average 读 `exp/cosyvoice/...` | 找不到 checkpoint | 改路径 |
| 8 | `run.sh` 的 hifigan checkpoint 是 `hifigan.pt` | 发布目录通常是 `hift.pt` | `train.py` 找不到会 warning，从头初始化 GAN | 确认文件名；不要轻易训 hifigan |
| 9 | CosyVoice2 `run.sh` 用 `speech_tokenizer_v3.onnx`，推理代码加载 `speech_tokenizer_v2.onnx` | `examples/libritts/cosyvoice2/run.sh:40` vs `cosyvoice.py:155`；`run_dpo.sh` 用 v2 | 可能 token 词表/版本不一致 | 训练前核对模型目录实际文件名；优先按推理代码用 v2 tokenizer |
| 10 | CosyVoice3 TensorRT `fp16=True` 出现 NaN | issue #1930：64/64 生成 NaN | 合成全静音/噪声 | 用 `fp16=False`，或等修复 |
| 11 | `inference_instruct` 只支持 v1 | `cosyvoice.py` 断言 `self.__class__.__name__ == 'CosyVoice'` | v2/v3 不能用 `spk_id + instruct` | v2/v3 用 `inference_instruct2`（参考音频 + instruct） |
| 12 | 流式下不能变速 | `model.py`：`assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'` | 实时场景不能动态 `speed` | 变速放文本/后处理，或非流式 |
| 13 | 短文本可能被警告 | `inference_zero_shot`：文本长度 < 0.5 × prompt text 时警告 | “嗯”“好的”这类短句音色/韵律差 | 短句用 SFT 音色，或缩短 prompt text |
| 14 | 参考音频 >30s 会断言 | `frontend._extract_speech_token` | 长参考直接失败 | 参考音频控制在 3–10s |
| 15 | vLLM 版本窗口窄 | README：0.11.x+ / 0.9.0；0.10.x 未测试 | 装错版本直接跑不起来 | 锁版本，独立环境 |
| 16 | `stream_scale_factor` / `token_hop_len` 改动会破坏音质 | `model.py` 注释：must matching training `static_chunk_size` | chunk 边界重复/爆音 | 不要随意改；要改就重训 flow 并回归 |

### 5.2 最小补丁清单

如果要用 CosyVoice3 做“固定音色 + 动态 instruct”的生产路线，至少需要：

1. **`spk2info` 构造**：注册参考音频，保存 `embedding`。
2. **per-call instruct 覆盖**：在 `frontend_instruct2` 里，当 `zero_shot_spk_id` 非空时，仍然用传入的 `instruct_text` 覆盖缓存里的 `prompt_text`：

```python
def frontend_instruct2(self, tts_text, instruct_text, prompt_wav,
                       resample_rate, zero_shot_spk_id):
    model_input = self.frontend_zero_shot(
        tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id
    )
    if zero_shot_spk_id != '' and instruct_text:
        instruct_token, instruct_len = self._extract_text_token(instruct_text)
        model_input['prompt_text'] = instruct_token
        model_input['prompt_text_len'] = instruct_len
    del model_input['llm_prompt_speech_token']
    del model_input['llm_prompt_speech_token_len']
    return model_input
```

这样复用缓存的参考 speech token / feat / embedding，只替换 instruct 文本，避免每次重算参考音频。

3. **v3 SFT instruct 支持**：如果坚持用 `inference_sft`，需要：
   - `frontend_sft` 接受 `instruct_text`，把 instruct token 放到 text 前面；
   - `inference_sft` 把 instruct 传进去；
   - 保证 EOP token 的位置和训练一致；
   - 训练时 `use_spk_embedding: True`，推理用同一 speaker 的 `spk2info['embedding']`。

4. **`prepare_data.py` 逐句 instruct**：改成读取 `utt2instruct`，或训练前覆盖 `data/train/instruct`。

5. **`run.sh` 路径修复**：`exp/cosyvoice3`、`hift.pt`、`stop_stage`。

### 5.3 训练前 smoke test

- 20–50 step：loss 有限、不 NaN、显存稳定。
- 保存一个 checkpoint，能加载、能推理。
- 同一句话跑 10 次，检查是否有重复/漏字/爆音。
- 用训练集内和训练集外各 5 句，确认不是“只会背训练集”。
- 流式 30 秒长文本，检查 chunk 边界是否咔哒、是否语速漂移。

---

## 6. 实时通话与并发部署

### 6.1 官方流式到底怎么工作

`inference_sft` / `inference_zero_shot` / `inference_instruct2` 都支持 `stream=True`，返回 generator，每次 yield 一个 `{'tts_speech': tensor}`。内部是：

```
LLM 线程：逐 token 生成 speech token，写入 tts_speech_token_dict
主线程：每 100ms 检查 token 数，够了就 token2wav 出一个 chunk
```

CosyVoice2/3 的关键参数：

```python
token_hop_len = 25          # 25 token = 1 秒音频
token_max_hop_len = 100     # 后续 chunk 逐步增大，降低重复推理
stream_scale_factor = 2
pre_lookahead_len = 3       # flow 需要看未来 3 个 token
```

首块条件：

```
已生成 token 数 >= token_hop_len + prompt_token_pad + pre_lookahead_len
```

`prompt_token_pad` 来自参考音频的 flow prompt token 长度对 25 取整：

```python
prompt_token_pad = ceil(prompt_len / 25) * 25 - prompt_len
```

因此：

- 纯 SFT、无 prompt：首块约等 25 + 3 = 28 token ≈ 1.12 秒音频。
- zero-shot/instruct2：还要加 0–24 个 prompt padding token，首块更慢。
- 25Hz 下，LLM 要先生成约 1.1 秒音频对应的 token，token2wav 才能出第一块。这就是官方 Python 首包难低于 1 秒的根本原因。

`stream_scale_factor=2` 让后续 chunk 从 25 → 50 → 100 token，减少 token2wav 调用次数，但 chunk 变大、首包之后延迟抖动也更明显。

`inference_bistream` 支持文本流式输入（LLM 边收文本边生成），要求 `prompt_text` 含 `<|endofprompt|>`，内部用 `mix_ratio=[5, 15]` 做文本/语音 token 交错。这是更接近“边想边说”的模式，但官方示例和工程化程度不如普通流式。

### 6.2 150ms 的现实差距

官方 README 写“bi-streaming ... latency as low as 150ms”，但没有给测试条件。结合代码和社区数据：

- 未优化 Python 路径：首包约 1.1–2 秒（issue #1798 等）。
- 社区反馈（issue #1811 评论）：不开 TensorRT 首包 2–4 秒；开 TensorRT 在 A6000 约 600ms、4090 约 450ms。
- NVIDIA Triton + TRT-LLM：
  - CosyVoice2 UNet：L20 并发 1 首块 220ms，并发 2 477ms，并发 4 1107ms。
  - CosyVoice2 DiT：L20 并发 1 首块 197.5ms，并发 2 281ms，并发 4 511ms，并发 10 1215ms。
  - CosyVoice3：L20 并发 4 首块平均 750ms，p95 978ms，p99 1002ms。

所以 150ms 更像是“内部优化版本 + 特定硬件 + 特定并发/文本长度”的指标，不是官方 Python 代码的开箱结果。实时通话要把 150ms 当作目标，不要当作基线。

### 6.3 服务方案对比

| 方案 | 能力 | 并发真相 | 适用 |
|---|---|---|---|
| 直接 Python `inference_*` | 最简单，支持流式 generator | 单进程；多线程会抢 GPU/GIL；无 batching | 离线批处理、POC |
| FastAPI server | HTTP/SSE，官方示例 | 全局单模型；无 `--max_conc`；`async def` 里调阻塞推理会卡事件循环 | 低并发 demo |
| gRPC server | 官方示例，支持 `--max_conc` | `ThreadPoolExecutor(max_workers=max_conc)` + `maximum_concurrent_rpcs`，默认 4；只是并发上限，不是吞吐优化 | 中小规模服务 |
| vLLM | 加速 LLM，支持连续批处理 | token2wav 仍逐请求；0.11.x/0.9.0；`gpu_memory_utilization=0.2` | LLM 是瓶颈时 |
| Triton + TRT-LLM | 官方推荐生产路径；LLM/ token2wav/ vocoder 分模块 | 真正的多实例、动态 batch、分离部署；需要构建 TRT engine | 高并发、低首包 |
| FlashCosyVoice | CosyVoice2 token2wav batching | 主要面向 v2；v3 支持要看分支 | v2 高吞吐 |
| 自研引擎 | 最大自由度 | 需要 continuous batching、paged KV、CUDA graphs、token2wav 池 | 大规模生产 |

关键结论：

- **`--max_conc` 不是吞吐。** gRPC 的 `max_conc=4` 只是允许 4 个请求同时进来，GPU 仍然串行/竞争；真正的吞吐要靠 batching 和 token2wav 并行。
- **vLLM 只解决 LLM。** CosyVoice3 的 offline 数据里，LLM batch 16 从 13.21s 降到 2.12s，但 token2wav 仍是 6.27s，端到端只快 2.2 倍。
- **token2wav 是主瓶颈。** 要提高并发，必须让 token2wav 支持 batch、多实例、或多 GPU 分离部署。
- **FastAPI 示例不适合生产。** 它的 `async def` 直接调用同步推理，会阻塞事件循环；至少改成 `run_in_threadpool` 或同步 worker + 多进程。

### 6.4 官方 benchmark 数据

#### CosyVoice3 + Triton/TRT-LLM（L20）

流式，并发 4，首块：

| 并发 | 平均 (ms) | p50 | p90 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|
| 4 | 750.42 | 740.31 | 941.05 | 977.55 | 1002.37 |

离线，TRT-LLM：

| LLM batch | LLM 时间 (s) | token2wav 时间 (s) | pipeline (s) | RTF |
|---:|---:|---:|---:|---:|
| 1 | 13.21 | 5.72 | 19.48 | 0.1091 |
| 2 | 8.46 | 6.02 | 14.91 | 0.0822 |
| 4 | 5.07 | 5.95 | 11.43 | 0.0630 |
| 8 | 2.98 | 6.11 | 9.53 | 0.0562 |
| 16 | 2.12 | 6.27 | 8.83 | 0.0501 |

读法：LLM 批处理收益很大，但 token2wav 几乎不随 batch 降；总 RTF 只从 0.109 降到 0.050。

#### CosyVoice2 + UNet（L20，26 条 prompt，约 170 秒音频）

流式首块：

| 并发 | 平均 (ms) | p50 (ms) | RTF |
|---:|---:|---:|---:|
| 1 | 220.43 | 218.07 | 0.1237 |
| 2 | 476.97 | 369.25 | 0.1022 |
| 4 | 1107.34 | 1243.75 | 0.0922 |
| 1（spk2info 缓存） | 189.88 | 184.81 | 0.1155 |
| 2（spk2info 缓存） | 323.04 | 316.83 | 0.0905 |
| 4（spk2info 缓存） | 977.68 | 903.68 | 0.0733 |

离线，HF vs TRT-LLM：

| 后端 | batch | LLM (s) | total (s) | RTF |
|---|---:|---:|---:|---:|
| HF | 1 | 39.26 | 44.31 | 0.2494 |
| HF | 16 | 8.42 | 13.78 | 0.0821 |
| TRT-LLM | 1 | 12.46 | 17.31 | 0.0987 |
| TRT-LLM | 16 | 2.01 | 6.63 | 0.0386 |

#### CosyVoice2 + Step-Audio2 DiT（L20）

首块：

| 并发 | 平均 (ms) | p50 | p90 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 197.50 | 196.13 | 214.65 | 215.96 | 229.21 |
| 2 | 281.15 | 278.20 | 345.18 | 361.79 | 395.97 |
| 4 | 510.65 | 530.50 | 630.13 | 642.44 | 666.65 |
| 6 | 921.54 | 918.86 | 1079.97 | 1265.22 | 1524.41 |
| 8 | 1019.95 | 1085.26 | 1371.05 | 1402.24 | 1410.66 |
| 10 | 1214.98 | 1293.54 | 1575.36 | 1654.51 | 2161.76 |

总请求延迟：

| 并发 | RTF | 平均 (ms) | p50 | p90 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1228 | 833.66 | 779.98 | 1297.05 | 1555.97 | 1653.02 |
| 2 | 0.0901 | 1166.23 | 1124.69 | 1762.76 | 1900.64 | 2204.14 |
| 4 | 0.0741 | 1849.30 | 1759.42 | 2624.50 | 2822.20 | 3128.42 |
| 6 | 0.0774 | 2936.13 | 3054.64 | 3849.60 | 3900.49 | 4245.79 |
| 8 | 0.0691 | 3408.56 | 3434.98 | 4547.13 | 5047.76 | 5346.53 |
| 10 | 0.0707 | 4306.56 | 4343.44 | 5769.64 | 5876.09 | 5939.79 |

DiT 分离部署（LLM 独占 1 张 L20，token2wav 放其他 GPU）能把首块压到约 200–300ms（低并发），但并发上来后仍然线性变差；token2wav GPU 数量比 LLM batch 更决定首包。

### 6.5 并发估算

不要用单请求 RTF 直接乘。用聚合吞吐：

```
可服务实时路数 ≈ GPU 每秒能产出的音频秒数 / 1.0
```

再按目标 RTF 打折。例如 CosyVoice3 在 L20、batch 16 的 offline RTF 0.0501，理论聚合吞吐约 20 秒音频/秒；但这是离线、无首包 SLA 的数字，实时流式还要扣除首块、chunk 边界、p99 抖动，实际规划可能只有 5–10 路/卡。必须用自己的文本长度和并发压测。

压测要求：

- 文本长度用真实通话分布（3–15 秒/句），不要只用 1 秒短句。
- 会话长度 30–60 秒，包含多句、打断、静音。
- 记录 TTFA（首包）、inter-chunk gap、p50/p95/p99、RTF、GPU 利用率、显存、队列长度。
- 逐步加并发，直到 p95 首包超过 SLA 或出现 OOM/排队。
- 预留 1.5–2 倍容量，避免突发流量把 p99 打穿。

### 6.6 实时链路建议

```
LLM 流式输出
  → 句子/短语切分（首块 5–12 字，尽快出第一声）
  → 文本规范化 / 风格导演（标点、语气词、情绪标记）
  → CosyVoice 流式生成
  → Opus 编码
  → 抖动缓冲
  → 客户端
```

- 预合成固定话术：问候、确认、等待、结束语、错误提示。
- 首句尽量短，让首包尽快出来；后续句子可以更长。
- 参考音频和 `spk2info` 启动时预加载；不要每通电话重新提取。
- 打断：VAD 检测到用户说话 → abort 当前 TTS → 释放请求/缓存 → 切回 ASR。
- 抖动缓冲 60–120ms；丢包隐藏。
- 最终走 PSTN/8k 时在边缘下采样；训练和内部链路仍用 24k。
- 端到端 SLA 要把 ASR、LLM、网络、抖动缓冲都算进去，不能只看 TTS 首包。
- GPU 上只跑模型；音频编解码、Opus、重采样放 CPU 或独立线程/进程。

---

## 7. 音色与情感控制方案

### 7.1 固定人设（推荐先做）

- 用目标说话人的真实对话数据做 SFT。
- 推理用 `inference_sft(tts_text, spk_id)`，不需要每次传参考音频。
- 情感来源：
  - 训练数据本身的表达性（笑声、叹气、重音、语气词）
  - 文本措辞、标点、断句
  - 文本导演模块：拆长句、加停顿、加语气词、调整语序
- 优点：延迟最低，音色最稳，工程最简单。
- 缺点：不能随时切“现在很生气”，情绪范围受数据限制。

### 7.2 Instruct 路线

| 模型 | 接口 | 参考音频 | instruct | 备注 |
|---|---|---|---|---|
| CosyVoice v1 | `inference_instruct(tts_text, spk_id, instruct_text)` | 不需要 | 需要 | 只能用内置 speaker |
| CosyVoice2 | `inference_instruct2(tts_text, instruct_text, prompt_wav, zero_shot_spk_id='')` | 需要 | 需要 | 固定音色 + 风格 |
| CosyVoice3 | 同上 | 需要 | 需要 | 支持语言/方言/情绪/语速/音量 |

官方 instruct 清单（`cosyvoice/utils/common.py`）包括：

- 方言：广东、东北、甘肃、贵州、河南、湖北、湖南、江西、闽南、宁夏、山西、陕西、山东、上海、四川、天津、云南。
- 音量：loud / soft。
- 语速：slow / fast。
- 情绪：开心、伤心、生气。
- 风格：小猪佩奇、机器人。

示例：

```python
instruct = "You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>"
for out in model.inference_instruct2(
        '你好，今天想了解点什么？',
        instruct,
        'ref.wav',
        zero_shot_spk_id='',
        stream=True):
    ...
```

限制：

- 清单外的自然语言描述（“温柔一点”“像老朋友聊天”“疲惫但礼貌”）没有训练保证，效果不稳定。
- `inference_instruct2` 会删掉 LLM 的 prompt speech token，但保留 speaker embedding 和 flow 的 prompt feat/embedding；音色靠参考音频的 speaker embedding，风格靠 instruct 文本。
- 用 `zero_shot_spk_id` 缓存后，新的 `instruct_text` 会被忽略（见 5.1 #3）。动态情绪要么每次传参考音频（重算特征），要么每个 instruct 预注册一个 id：

```python
# 同一条参考音频，不同 instruct 预注册成不同 id
for spk_id, instruct in {
    'spk_neutral': 'You are a helpful assistant.<|endofprompt|>',
    'spk_happy':   'You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>',
    'spk_sad':     'You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>',
}.items():
    model.add_zero_shot_spk(instruct, 'ref.wav', spk_id)

model.save_spkinfo()
```

调用时切 `zero_shot_spk_id` 即可。每个 id 会缓存一份 prompt speech token/feat，内存开销不大，但注册时要提一次特征。

- 流式下 `speed` 不支持动态改；`model.py` 明确断言。变速放文本层或后处理。

### 7.3 动态情绪 POC

如果业务必须“同一通电话里温柔→开心→道歉→严肃”，固定 instruct 清单不够，需要：

1. 数据：每个情绪 50–100 句，同一文本多情绪版本；标注情绪强度。
2. 标签：逐句 instruct，或控制 token（如 `[温柔]`、`[开心]`、`[严肃]`）。
3. 训练：CosyVoice3 的 `instruct_token` 已经在训练序列里，可以把情绪标签放进 instruct；但官方前端不支持逐句/动态推理，需要改前端。
4. 推理：文本导演根据对话状态选 instruct；或每情绪一个 speaker id。
5. 评估：情绪识别准确率、SECS、CER、自然度 MOS；重点看情绪切换是否突兀、音色是否漂。

风险：

- 小数据下模型可能把“情绪”和“说话人”纠缠，切情绪时音色也变。
- 控制 token 可能被文本规范化/分词破坏，需要允许 special token。
- 流式首块延迟会因 instruct 变长而增加。
- 没有官方 LoRA，全参多情绪训练成本高；可以尝试冻结大部分 LLM、只训少量层/Adapter，但要自己实现。

### 7.4 两条路取舍

| 方案 | 音色 | 情感 | 首包 | 风险 |
|---|---|---|---|---|
| 固定人设 SFT + 文本导演 | 最稳 | 整体风格，有限动态 | 最低 | 低 |
| instruct2 + 多 speaker id | 稳 | 固定清单内可切 | 中 | 中 |
| 逐句 instruct SFT / 控制 token | 可能漂 | 动态 | 中高 | 高，需 POC |

建议：首发用“固定人设 SFT + 文本导演”；instruct 只做粗粒度风格；动态情绪作为第二阶段 POC，不要阻塞主线。

---

## 8. 评测方案

### 8.1 离线

| 指标 | 工具/做法 | 关注点 |
|---|---|---|
| CER/WER | Paraformer / Whisper / SenseVoice 回读 | 漏字、重复、数字/英文读法 |
| Speaker similarity | CAM++ / WavLM / SECS | 目标音色一致性 |
| UTMOS / DNSMOS | 自动 MOS | 自然度、噪声、金属音 |
| MOS / CMOS | 人工盲听 | 情感、韵律、拟人度 |
| RTF | 生成音频时长 / 耗时 | 单请求效率 |
| 首包 TTFA | 从请求到第一个音频 chunk | 实时核心指标 |
| p50/p95/p99 | 压测 | 尾延迟 |
| 重复率 | 人工 + 自动检测 | 流式 chunk 边界常见 |
| 长文本稳定性 | 30–60 秒连续文本 | 语速漂移、漏句、崩溃 |
| 方言/多语言 | 分语种测试集 | 发音、语调 |
| 发音修复 | 姓名/地名/缩写 | 拼音/CMU 是否生效 |

### 8.2 在线

- TTFA、inter-chunk gap、抖动、丢包。
- 30–60 秒会话的完成率、打断恢复时间。
- 并发路数、GPU 利用率、显存、队列长度。
- 错误率、超时率、fallback 比例。
- 用户主观评分：像不像真人、情绪是否合适、是否愿意继续听。

### 8.3 验收门槛示例

以下只是示例，按业务调整：

- 首包 p95 < 500ms（Triton/TRT-LLM，目标 GPU 实测）。
- 流式 RTF < 0.5，p99 不出现 > 2 秒卡顿。
- CER < 3%（中文通用文本），专有名词单独测试。
- SECS 不低于基线模型，盲听“像目标人”比例 > 80%。
- MOS ≥ 4.0（5 分制），无明显机械音/爆音/重复。
- 30 秒长文本无漏句、无重复、无中途变声。
- 打断后 300ms 内停止出声。

评测集要固定：同一批文本、同一参考音频、同一 GPU、同一并发，才能和 Qwen3-TTS 或基线对比。

---

## 9. 推荐落地顺序

> 如果确定用 CosyVoice3，直接按第 15 节的 v3 专用方案执行；本节保留 CosyVoice2 的基线顺序，仅作为对照和回退方案。

### Phase 0（约 1 周）：基线压测

- 部署 CosyVoice2-0.5B 和 CosyVoice3-0.5B 的 zero-shot / instruct2。
- 用 50–100 句真实通话文本，测首包、RTF、SECS、CER、MOS。
- 在目标 GPU 上跑 gRPC `max_conc=1/2/4`，记录 p95/p99。
- 确认 150ms 宣称和实际差距，建立自己的基线。

### Phase 1（1–2 周）：数据

- 收集 1–3 小时目标说话人干净音频，统一麦克风/环境。
- 按 3–15 秒切句，文本规范化，保留标点。
- 切 train/dev/test；dev/test 按场次留出，不要随机切。
- 选 3–10 秒参考音频，准备 prompt text。

### Phase 2（1–2 周）：CosyVoice2 SFT

- 只训 LLM，`use_spk_embedding: True`，`lr=1e-5`，`constantlr`，1–5 epoch。
- 补 `spk2info.pt`，用 `inference_sft` 流式推理。
- 每 epoch 导出，按 SECS/CER/盲听选 checkpoint。
- 如果音质不够，再考虑 flow；hifigan 不动。

### Phase 3（1–2 周）：CosyVoice3 对比

- 用同一批数据做 zero-shot / instruct2，缓存参考音频。
- 如果要做 v3 SFT，先打 5.2 的补丁：EOP、instruct、`spk2info`。
- 同文本 A/B：CosyVoice2 SFT vs CosyVoice3 zero-shot/instruct2 vs v3 SFT。
- 选“音色最像 + 情感最合适 + 首包可接受”的方案，不要只看公开 CER。

### Phase 4（2–4 周）：流式与并发

- 先 vLLM 加速 LLM，测 token2wav 是否成为瓶颈。
- 再上 Triton + TRT-LLM；v3 用官方 pipeline，v2 可试 UNet/DiT。
- 参考音频长度对齐 25 token 的整数倍，减少 `prompt_token_pad`。
- 固定话术预合成；首句拆短；缓存 `spk2info`。
- 压测到 p95 首包和 p99 抖动达标，留 1.5–2 倍余量。

### Phase 5（2–4 周）：情感

- 先做“文本导演 + 固定人设”，验证情绪表达是否够用。
- 不够再做逐句 instruct SFT POC；每情绪 50–100 句。
- 用多 speaker id 或前端补丁支持动态 instruct。
- 评估情绪切换时的音色漂移和首包变化。

### Phase 6：上线

- 灰度、A/B、fallback 到预合成/备用模型。
- 监控首包、RTF、错误率、GPU、队列。
- 建立数据回流：失败案例、重复、漏字、情绪不合适的样本。

---

## 10. 与 Qwen3-TTS 的对比

> 公开指标口径不同，不能直接横比；下表是工程能力对比，不是音质排名。

| 维度 | CosyVoice | Qwen3-TTS | 对本项目的影响 |
|---|---|---|---|
| 模型规模 | 0.3B / 0.5B | 0.6B / 1.7B | CosyVoice 更省算力；Qwen 1.7B 质量上限可能更高 |
| Speech token 率 | 25Hz（v2/v3） | 12Hz | CosyVoice 每秒 token 更多，LLM/token2wav 压力更大；Qwen 更适合高并发 |
| 官方 SFT | LLM/flow/hifigan 全参，脚本完整 | 单说话人全参 SFT，脚本有已知 bug | CosyVoice 训练链路更成熟 |
| SFT 推理 | v2 可用；v3 有 EOP/instruct 坑 | 转 custom_voice，接口较顺 | v2 更适合先打通 |
| LoRA | 无官方 | 无官方 | 都要自己改 |
| Instruct/情感 | 固定清单 + instruct2 | Base 无 instruct；CustomVoice 有 | CosyVoice 的 instruct 更明确，但仍是固定集 |
| 流式 | 官方 bi-streaming，首包 0.2–2s（看优化） | 官方包无真正音频流式，靠社区/vLLM-Omni | CosyVoice 流式更可用 |
| 并发 | vLLM + Triton/TRT-LLM + 分离部署 | vLLM-Omni/社区 fork | CosyVoice 生产路径更清晰 |
| 生态 | 官方 runtime、NVIDIA 贡献、gRPC/FastAPI | Qwen 生态大，TTS 运行时较新 | CosyVoice 部署资料更全 |
| License | Apache 2.0 | 以模型卡为准（通常 Apache 2.0） | 商用都要法务复核 + 参考音频授权 |
| 适合 | 自定义音色 + 流式 + 高并发 | Qwen 生态、1.7B 质量档 | 建议两套都压测，按数据选 |

粗略算力直觉：

- CosyVoice 0.5B × 25Hz ≈ 12.5B token-参数/秒。
- Qwen 1.7B × 12Hz ≈ 20.4B token-参数/秒；Qwen 0.6B × 12Hz ≈ 7.2B。
- 所以 CosyVoice 的 LLM 负担低于 Qwen 1.7B，和 Qwen 0.6B 同一量级；但 CosyVoice 多一个 25Hz token2wav，实际并发要看 token2wav 优化。
- 这只是量级直觉，不是 benchmark；真正对比必须在同一 GPU、同一文本、同一首包口径下压测。

对本项目的建议：

1. **不要先押注单一模型。** 用同一批 50–100 句、同一参考音频，跑 CosyVoice2 SFT、CosyVoice3 zero-shot/instruct2、Qwen3-TTS Base SFT 三条线。
2. **首包和并发优先用 CosyVoice 验证**，因为它的 Triton/TRT-LLM 和流式代码更完整；把 Qwen 作为质量/生态备选。
3. **如果 CosyVoice2 SFT 的音色/情感已经达标，优先它**：0.5B、25Hz、官方 SFT、官方 runtime，工程风险最低。
4. **如果 CosyVoice3 的音质明显更好但 SFT 推理不稳**，用 zero-shot/instruct2 + 缓存参考先上线，SFT 作为二期。
5. **如果 Qwen3-TTS 1.7B 在盲听中明显更拟人**，再评估它的流式/并发补丁成本；不要为了生态牺牲首包。

---

## 11. 风险与合规

- **代码 License**：仓库根目录 Apache 2.0；商用一般允许，但需要保留 LICENSE/NOTICE，并检查第三方子模块（如 Matcha-TTS）的许可证。
- **模型权重 License**：以 HuggingFace/ModelScope 模型卡为准；下载前确认是否允许商用、是否要求署名、是否有限制。
- **参考音频权利**：必须获得说话人授权，明确用途、期限、地域、是否允许合成；不要用抓取的公开音频做商用音色。
- **声音克隆合规**：遵守当地深度合成/人格权/反诈骗法规；建议加显式标识、水印、日志和滥用检测。
- **数据隐私**：通话录音、文本、说话人 embedding 都是敏感数据；脱敏、加密、最小留存。
- **README disclaimer**：官方写“for academic purposes only ... examples sourced from the internet”，这不是商业授权；商用前让法务复核。
- **License 咨询**：社区 issue #1945 只有 bot 回复，没有维护者法律确认；不要把它当作商用许可。
- **出口/行业合规**：金融、医疗、政务通话可能有额外录音/合成标识要求。

---

## 12. 待确认信息

1. CosyVoice3 官方是否计划修复 `inference_sft` 的 EOP 断言和 instruct 前端？
2. `zero_shot_spk_id` 缓存忽略新 instruct 是设计还是 bug？是否有官方推荐的多情绪方案？
3. 150ms 首包的具体测试条件（GPU、并发、文本长度、是否 TensorRT、是否内部版本）？
4. CosyVoice3 TensorRT `fp16=True` 的 NaN（issue #1930）是否已修复？
5. vLLM 0.10.x 是否会在后续版本支持？
6. CosyVoice3 的 token2wav 是否有官方 batching/多实例方案（类似 v2 的 FlashCosyVoice）？
7. CosyVoice3 SFT 是否支持多说话人 + 每说话人 instruct 的官方推理接口？
8. 商用授权的正式口径（代码 Apache 2.0 与模型权重、第三方子模块、参考音频）。
9. `examples/libritts/cosyvoice2/run.sh` 使用 `speech_tokenizer_v3.onnx` 是否为笔误；v2 训练应使用 v2 还是 v3 tokenizer。
10. `run.sh` 的 `hifigan.pt` / `exp/cosyvoice` 路径问题是否会在上游修复。

---

## 13. 参考资料

### 官方

- 仓库：<https://github.com/QwenAudio/CosyVoice>（原 `FunAudioLLM/CosyVoice`）
- CosyVoice3 模型：<https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512>
- CosyVoice2 模型：<https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B>
- CosyVoice 1.0 模型：<https://huggingface.co/FunAudioLLM/CosyVoice-300M>
- CosyVoice3 论文：<https://arxiv.org/pdf/2505.17589>
- CosyVoice2 论文：<https://arxiv.org/pdf/2412.10117>
- CosyVoice3 demo：<https://funaudiollm.github.io/cosyvoice3/>
- CosyVoice2 demo：<https://funaudiollm.github.io/cosyvoice2/>
- CV3-Eval：<https://github.com/FunAudioLLM/CV3-Eval>

### 代码与配置

- CosyVoice3 训练脚本：`examples/libritts/cosyvoice3/run.sh`
- CosyVoice2 训练脚本：`examples/libritts/cosyvoice2/run.sh`
- CosyVoice3 配置：`examples/libritts/cosyvoice3/conf/cosyvoice3.yaml`
- CosyVoice2 配置：`examples/libritts/cosyvoice2/conf/cosyvoice2.yaml`
- 推理入口：`cosyvoice/cli/cosyvoice.py`
- 前端：`cosyvoice/cli/frontend.py`
- 模型/流式：`cosyvoice/cli/model.py`
- LLM：`cosyvoice/llm/llm.py`
- 数据处理：`cosyvoice/dataset/processor.py`
- instruct 清单：`cosyvoice/utils/common.py`
- 特征提取：`tools/extract_embedding.py`、`tools/extract_speech_token.py`
- parquet：`tools/make_parquet_list.py`

### 运行时

- Triton/TRT-LLM 总览：`runtime/triton_trtllm/README.md`
- CosyVoice3 runtime：`runtime/triton_trtllm/README.Cosyvoice3.md`
- CosyVoice2 UNet runtime：`runtime/triton_trtllm/README.Cosyvoice2.Unet.md`
- CosyVoice2 DiT runtime：`runtime/triton_trtllm/README.Cosyvoice2.DiT.md`
- FastAPI：`runtime/python/fastapi/server.py`
- gRPC：`runtime/python/grpc/server.py`

### 相关 issue

- 首包延迟讨论：#1798
- TensorRT 首包数据：#1811
- CosyVoice3 TRT fp16 NaN：#1930
- 商用 License 咨询：#1945

> issue 链接：`https://github.com/QwenAudio/CosyVoice/issues/<编号>`。旧仓库链接会自动跳转。

### 对比文档

- 本项目 Qwen3-TTS 调研：`docs/research/qwen3-tts-realtime-finetuning.md`

---

## 14. 版本与时效声明

- 本文基于 2026-05-26 的 `QwenAudio/CosyVoice` commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`。
- CosyVoice3 仍在快速迭代；`inference_sft`、instruct 缓存、TRT fp16、vLLM 版本支持都可能变化。
- 所有性能数字来自官方 README/runtime 文档或公开 issue，硬件、数据、并发、文本长度不同，不能直接作为你环境的 SLA。
- 落地前请在目标 GPU 上用真实通话文本复测首包、RTF、SECS、CER 和 p99。

---

## 15. 附录：CosyVoice3 专用落地方案（本项目采用）

> 结论：用 `Fun-CosyVoice3-0.5B-2512`，LLM-only SFT；固定人设走“常量 instruct + 补丁版 `inference_sft`”，动态情绪走“逐句 instruct 训练 + 补丁版 `inference_sft`”。不要直接依赖官方原版 `inference_sft`。

### 15.1 为什么 v3 不能直接照搬 v2 的 SFT

CosyVoice3 的代码和 v2 有几处关键差异：

1. **v3 训练时每个样本都必须有 instruct。**
   - `CosyVoice3LM.forward` 无条件读取 `batch['instruct_token']` 和 `batch['instruct_token_len']`。
   - 如果数据没有 `instruct` 文件，batch 里没有这两个 key，训练直接 `KeyError`。
   - 所以即使是固定人设，也要给所有句子写一个常量 instruct，例如：
     `You are a helpful assistant.<|endofprompt|>`

2. **v3 的 `inference_sft` 会断言文本含 `<|endofprompt|>`。**
   - `CosyVoice3LM.inference` 里：
     ```python
     assert 151646 in text, '<|endofprompt|> not detected ...'
     ```
   - 原版 `frontend_sft` 只返回 text + embedding，不会加 EOP，所以直接调用会失败。

3. **v3 的 `frontend_sft` 不传 instruct。**
   - 训练时模型见过 instruct，推理时看不到，属于训练/推理不一致；即使绕过断言，音色和风格也会漂。

4. **`spk2info.pt` 仍然要自己补。**
   - `frontend_sft` 需要 `self.spk2info[spk_id]['embedding']`。
   - 训练不会自动生成 `spk2info.pt`。

5. **`CosyVoice3` 没有 `load_jit`。**
   - 只有 `load_trt` / `load_vllm`；JIT 产物用于支持 JIT 的自定义 runtime，不是 v3 Python 推理的默认路径。

6. **v3 的 TensorRT `fp16=True` 有 NaN 报告。**
   - issue #1930：64/64 生成 NaN；运行时用 `fp16=False`。

因此，v3 的推荐做法不是“直接 `inference_sft`”，而是**给 `frontend_sft` / `inference_sft` 补上 instruct 参数**；补丁很小，但必须做。

### 15.2 两条 v3 路线

| 路线 | 训练 | 推理 | 参考音频 | 动态 instruct | 首包 | 适用 |
|---|---|---|---|---|---|---|
| **A. v3 SFT + 补丁版 `inference_sft`** | LLM-only；常量或逐句 instruct；`use_spk_embedding: True` | 补丁版 `inference_sft` | **不需要** | 支持（训练覆盖到的 instruct） | 最低 | **推荐，生产主路线** |
| B. v3 `inference_instruct2` | LLM-only 或直接用预训练；逐句 instruct；`use_spk_embedding: False` 更贴近推理 | `inference_instruct2` | 需要，或缓存 `spk2info` | 需要补丁或多 speaker id | 中 | 快速 POC / 回退 |

推荐 A：推理不带参考音频，没有 prompt token padding，首包最低；动态情绪也可以通过“逐句 instruct 训练 + 推理传 instruct”实现。

### 15.3 路线 A：v3 SFT + 补丁版 `inference_sft`

#### 15.3.1 数据

- 目标说话人 1–3 小时干净音频；3–15 秒/句；统一麦克风和环境。
- 文本规范化；保留标点；姓名/专有名词单独测试。
- 所有句子都要有 instruct，否则 v3 训练会缺 key。
- 固定人设：全部用同一个常量 instruct。
- 动态情绪：逐句 instruct，例如：
  - `You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>`
  - `You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>`
  - `You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>`
- 训练和推理的 instruct 字符串必须**逐字一致**（空格、标点、EOP 都要一致）。

官方 `prepare_data.py` 只支持一个全局 instruct：

```bash
python local/prepare_data.py \
  --src_dir /path/to/your_wavs \
  --des_dir data/train \
  --instruct "You are a helpful assistant.<|endofprompt|>"
```

如果自定义数据不是 LibriTTS 的 `*/*/*.wav + *.normalized.txt` 结构，直接准备 `wav.scp`、`text`、`utt2spk`、`spk2utt`、`instruct` 五个文件，再跑 `make_parquet_list.py`。

动态情绪则在 `prepare_data.py` 之后覆盖 `data/train/instruct`：

```python
# utt2instruct: {utt_id: 完整 instruct 字符串}
with open('data/train/instruct', 'w', encoding='utf-8') as f:
    for utt, instruct in utt2instruct.items():
        f.write(f'{utt} {instruct}\n')
```

注意：`make_parquet_list.py` 只有在所有样本都有 `instruct` 时才稳定；v3 训练本身也要求每个 batch 都有 `instruct_token`，所以不要混入没有 instruct 的句子。

#### 15.3.2 特征和 parquet

```bash
python tools/extract_embedding.py \
  --dir data/train \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx

python tools/extract_speech_token.py \
  --dir data/train \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx

python tools/make_parquet_list.py \
  --num_utts_per_parquet 1000 \
  --num_processes 10 \
  --src_dir data/train \
  --des_dir data/train/parquet
```

`use_spk_embedding: True` 需要 `spk2embedding.pt`；上面的 `extract_embedding.py` 会生成。

#### 15.3.3 配置

修改 `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml`：

```yaml
padding:
    use_spk_embedding: True   # 和补丁版 inference_sft 的 spk2info embedding 对齐

train_conf:
    optim: adam
    optim_conf:
        lr: 1e-5
    scheduler: constantlr
    scheduler_conf:
        warmup_steps: 500     # 小数据建议 5%-10% 总步数，不要盲目 2500
    max_epoch: 200            # 上限，实际 1-5 epoch 早停
    grad_clip: 5
    accum_grad: 2
```

`max_frames_in_batch: 2000` 保持默认；24GB 显存可降到 1000–1500。

#### 15.3.4 只训 LLM

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
torchrun --nnodes=1 --nproc_per_node=4 \
  --rdzv_id=1986 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config conf/cosyvoice3.yaml \
  --train_data data/train.data.list \
  --cv_data data/dev.data.list \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model llm \
  --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/llm.pt \
  --model_dir exp/cosyvoice3/llm/torch_ddp \
  --tensorboard_dir tensorboard/cosyvoice3/llm/torch_ddp \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 100 \
  --pin_memory \
  --use_amp
```

先只训 LLM；flow/hifigan 保持预训练。训练完把 `exp/cosyvoice3/llm/torch_ddp/llm.pt` 复制到完整模型目录，替换原 `llm.pt`。

#### 15.3.5 构造 `spk2info.pt`

```python
from cosyvoice.cli.cosyvoice import CosyVoice3

model = CosyVoice3(
    'pretrained_models/Fun-CosyVoice3-0.5B',
    load_trt=False, load_vllm=False, fp16=False
)
model.add_zero_shot_spk('参考音频对应的文本', 'ref.wav', 'my_spk')
model.save_spkinfo()
```

也可以从 `spk2embedding.pt` 手工构造：

```python
import torch
spk2emb = torch.load('data/train/spk2embedding.pt', map_location='cpu')
spk2info = {
    spk: {'embedding': torch.tensor(emb, dtype=torch.float32).unsqueeze(0)}
    for spk, emb in spk2emb.items()
}
torch.save(spk2info, 'pretrained_models/Fun-CosyVoice3-0.5B/spk2info.pt')
```

补丁版 `inference_sft` 只使用 `embedding`；`add_zero_shot_spk` 额外缓存的 prompt 字段对 SFT 不影响。

#### 15.3.6 最小补丁

修改 `cosyvoice/cli/frontend.py`：

```python
def frontend_sft(self, tts_text, spk_id, instruct_text=''):
    tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
    embedding = self.spk2info[spk_id]['embedding']
    model_input = {
        'text': tts_text_token,
        'text_len': tts_text_token_len,
        'llm_embedding': embedding,
        'flow_embedding': embedding,
    }
    if instruct_text:
        instruct_token, instruct_token_len = self._extract_text_token(instruct_text)
    else:
        instruct_token = torch.zeros(1, 0, dtype=torch.int32)
        instruct_token_len = torch.zeros(1, dtype=torch.int32)
    model_input['prompt_text'] = instruct_token
    model_input['prompt_text_len'] = instruct_token_len
    return model_input
```

修改 `cosyvoice/cli/cosyvoice.py` 的 `inference_sft`：

```python
def inference_sft(self, tts_text, spk_id, instruct_text='',
                  stream=False, speed=1.0, text_frontend=True):
    for i in tqdm(self.frontend.text_normalize(
            tts_text, split=True, text_frontend=text_frontend)):
        model_input = self.frontend.frontend_sft(i, spk_id, instruct_text)
        start_time = time.time()
        logging.info('synthesis text {}'.format(i))
        for model_output in self.model.tts(
                **model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(
                speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()
```

这个补丁的作用：

- `frontend_sft` 把 instruct 放进 `prompt_text`；
- `CosyVoice3LM.inference` 会把它和正文拼在一起，`assert 151646` 自然通过；
- `min_len` / `max_len` 会减掉 `prompt_text_len`，正文长度计算不受影响；
- 推理不需要参考音频，也没有 flow prompt token padding。

#### 15.3.7 推理

固定人设：

```python
INSTRUCT = "You are a helpful assistant.<|endofprompt|>"

for out in model.inference_sft(
        '你好，今天想了解点什么？',
        'my_spk',
        instruct_text=INSTRUCT,
        stream=True):
    ...
```

动态情绪：

```python
INSTRUCT_HAPPY = "You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>"

for out in model.inference_sft(
        '太好了，终于等到你！',
        'my_spk',
        instruct_text=INSTRUCT_HAPPY,
        stream=True):
    ...
```

前提：训练数据里确实有 `INSTRUCT_HAPPY` 这个 instruct，且所有样本都有 instruct。训练没见过的 instruct 不要指望泛化。

### 15.4 路线 B：v3 `inference_instruct2` + 参考音频

如果不想改 `frontend_sft` / `inference_sft`，可以用 `inference_instruct2`：

```python
model = CosyVoice3(
    'pretrained_models/Fun-CosyVoice3-0.5B',
    load_trt=False, load_vllm=False, fp16=False
)

INSTRUCT = "You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>"
for out in model.inference_instruct2(
        '你好，今天想了解点什么？',
        INSTRUCT,
        'ref.wav',
        zero_shot_spk_id='',
        stream=True):
    ...
```

特点：

- 不需要改训练代码；可以直接用预训练 v3，也可以加载 SFT 后的 LLM。
- 每次传参考音频时，会重新提取 prompt speech token / feat / embedding，首包更慢。
- 用 `add_zero_shot_spk` 缓存后，`zero_shot_spk_id` 会复用缓存；但**新的 `instruct_text` 会被忽略**，因为 `frontend_zero_shot` 在 id 非空时直接返回缓存的 `prompt_text`。
- 要动态情绪，二选一：
  1. 每个 instruct 预注册一个 speaker id（同一条参考音频）；
  2. 打 5.2 的 `frontend_instruct2` 补丁，用传入的 `instruct_text` 覆盖缓存的 `prompt_text`。

路线 B 的训练建议：

- 如果加载 SFT 后的 LLM，训练时用 `use_spk_embedding: False` 更贴近 `inference_instruct2`（推理用参考音频的 utterance embedding）。
- 如果直接用预训练 v3，不需要训练，先压测首包和并发。
- 参考音频 3–10 秒；长度尽量让 speech token 数接近 25 的整数倍，减少 `prompt_token_pad`。

### 15.5 v3 流式与并发

- 路线 A 的 `inference_sft(stream=True)` 没有 flow prompt token，`prompt_token_pad=0`，首块只等 `25 + 3 = 28` 个 token；比 `inference_instruct2` 少 0–24 个 token 的等待。
- 路线 B 有参考音频，`prompt_token_pad = ceil(prompt_len / 25) * 25 - prompt_len`，首块更慢。
- v3 生产运行时优先 Triton + TensorRT-LLM：
  - `runtime/triton_trtllm/README.Cosyvoice3.md`
  - `load_trt=True` 时用 `fp16=False`，避开 issue #1930 的 NaN。
  - `trt_concurrent` 设成目标并发；默认 1 会把 token2wav 串行化。
- vLLM 只加速 LLM；v3 的 token2wav 仍是瓶颈。官方 offline batch 16 时 LLM 2.12s、token2wav 6.27s、RTF 0.0501。
- 官方 L20、并发 4 的 v3 首块：平均 750ms，p95 978ms，p99 1002ms。不要把 150ms 当基线；先在目标 GPU 上复测。
- 并发规划：
  - 路线 A 减少 prompt padding，首包更好；
  - token2wav 仍要 batch/多实例/分离部署；
  - 按真实 3–15 秒句子压 p95/p99，不要用离线 RTF 直接乘。

### 15.6 v3 验收清单

- 音色：SECS、盲听“像目标人”比例。
- 内容：CER/WER、数字/英文/姓名、方言。
- 情感：instruct 依从率、情绪识别准确率、切换时音色漂移。
- 自然度：MOS/CMOS、UTMOS/DNSMOS、爆音/金属音。
- 流式：首包 p50/p95/p99、inter-chunk gap、30 秒长文本、重复/漏字。
- 并发：目标 GPU 上 1/2/4/8 路，GPU 利用率、显存、队列。
- 回归：每次改 instruct、chunk、TRT engine 都跑同一套固定测试集。

### 15.7 v3 风险与回退

| 风险 | 影响 | 回退 |
|---|---|---|
| 补丁随上游变化失效 | `inference_sft` 报错或音色漂 | 锁 commit；补丁加单元测试 |
| 训练数据 instruct 不统一 | 训练 KeyError 或推理风格漂 | 所有样本强制同一格式；训练前校验 |
| `spk2info` 缺失/维度不对 | `KeyError` / shape error | 启动时校验；用 `add_zero_shot_spk` 生成 |
| 动态 instruct 训练不足 | 情绪不生效或音色漂 | 每情绪 50–100 句；先做固定人设 |
| TRT fp16 NaN | 全静音/噪声 | `fp16=False` |
| token2wav 瓶颈 | 并发上不去、p99 高 | Triton/TRT-LLM、多实例、分离部署 |
| 150ms 预期落空 | SLA 不达标 | 按实测 500ms–1s 规划；首句拆短 |
| 上游 `run.sh` 路径/文件问题 | 训练/平均失败 | 按第 5 节修复；不用 `hifigan.pt` |

**回退顺序**：路线 A → 路线 B（`inference_instruct2` + 参考音频）→ CosyVoice2 SFT → Qwen3-TTS。每一步都用同一套测试集对比，不要凭感觉切换。

### 15.8 v3 最小闭环（按顺序执行）

1. 下载 `Fun-CosyVoice3-0.5B-2512`，确认 `llm.pt`、`flow.pt`、`hift.pt`、`campplus.onnx`、`speech_tokenizer_v3.onnx`、`CosyVoice-BlankEN` 齐全。
2. 准备 1–3 小时目标说话人音频，3–15 秒切句，文本规范化。
3. `prepare_data.py` 写常量 instruct；动态情绪覆盖 `instruct` 文件。
4. `extract_embedding.py` + `extract_speech_token.py` + `make_parquet_list.py`。
5. 改 `use_spk_embedding: True`、`lr=1e-5`、`constantlr`、`warmup_steps`。
6. 只训 LLM，1–5 epoch，每 epoch 导出。
7. 打 `frontend_sft` / `inference_sft` 补丁。
8. 用 `add_zero_shot_spk` 生成 `spk2info.pt`，把训练好的 `llm.pt` 放进完整模型目录。
9. 用固定 50–100 句做 SECS/CER/MOS/首包评测，选 checkpoint。
10. 上 Triton + TRT-LLM，`fp16=False`，压 1/2/4/8 路并发，按 p95/p99 定容量。

---

## 16. 基准模型选型：长期演进用哪个

> 这里的“基准”指**长期主模型 / 演进底座**，不是单次评测的 baseline。

### 16.1 结论

- **主底座：`Fun-CosyVoice3-0.5B-2512`（base，不是 RL 版）。**
- **稳定对照 / 回退：`CosyVoice2-0.5B`。**
- **质量上限挑战者：`Qwen3-TTS-12Hz-1.7B-Base`、`IndexTTS2-1.5B`。**
- **并发挑战者：`Qwen3-TTS-12Hz-0.6B-Base`。**
- **不建议：CosyVoice 1.0 的 300M 作为新项目主底座。**

理由：

1. **音色/情感**
   - CosyVoice3 在 CosyVoice 家族内 CER/SS 最好：test-zh CER 1.21 / SS 78.0，test-en WER 2.24 / SS 71.8，test-hard CER 6.71 / SS 75.8。
   - 支持 9 语言、18+ 方言、instruct、发音修复、文本规范化，适合“单一人设 + 情感/方言”的通话场景。
   - 0.5B 对单说话人 SFT 通常够用；但多说话人、强情感、零样本泛化会先撞到容量上限。CosyVoice3 没有官方更大模型，这是长期风险。

2. **性能**
   - 25Hz token 率是结构性劣势：每秒 token 数是 Qwen 12Hz 的 2 倍；token2wav 又是瓶颈。
   - 但 0.5B 比 Qwen 1.7B 小很多，LLM 绝对算力反而低；官方 Triton + TRT-LLM、vLLM、gRPC/FastAPI 链路最完整。
   - Qwen 12Hz + 0.6B 在理论上更适合高并发，但官方流式/并发工具链不如 CosyVoice 成熟。

3. **生态**
   - CosyVoice 有官方 SFT、inference、vLLM、Triton/TRT-LLM、gRPC/FastAPI，Apache 2.0，问题可定位、可改代码。
   - Qwen 生态更大，但 Qwen3-TTS 的 TTS 运行时较新，流式主要靠社区；IndexTTS2 质量强，但训练/流式/部署资料少。
   - 如果团队已经重度使用 Qwen/vLLM，Qwen3-TTS 的生态协同是加分项；否则 CosyVoice 的 TTS 专用工具链更省事。

### 16.2 对比矩阵

| 模型 | 音色/情感上限 | 单说话人 SFT 可控性 | 首包/流式 | 并发/算力 | 官方训练/部署 | 生态/社区 | 主要风险 | 定位 |
|---|---|---|---|---|---|---|---|---|
| `Fun-CosyVoice3-0.5B-2512` | 强 | 强（需第 15 节补丁） | 中强（Triton 750ms@4） | 中（25Hz，token2wav 瓶颈） | 强 | 强 | SFT 推理需补丁；0.5B 容量上限 | **主底座** |
| `CosyVoice2-0.5B` | 中强 | 最强（`inference_sft` 成熟） | 强（DiT 197ms@1） | 中（25Hz） | 强 | 强 | 音质/方言弱于 v3 | 稳定对照/回退 |
| `Qwen3-TTS-1.7B-Base` | 很强（待自有数据验证） | 中（官方 SFT 有 bug） | 中弱（官方包无真流式） | 中弱（1.7B，12Hz） | 中 | 中强（Qwen 生态） | 流式/并发要补工程；SFT 脚本坑 | 质量挑战者 |
| `Qwen3-TTS-0.6B-Base` | 中强 | 中 | 中弱（同上） | 强（0.6B，12Hz） | 中 | 中强 | 音质上限低于 1.7B；工具链新 | 并发挑战者 |
| `IndexTTS2-1.5B` | 很强（公开 CER/SS 好） | 中弱 | 弱（实时资料少） | 弱中（1.5B） | 中弱 | 中弱 | 训练/流式/部署生态不成熟 | 质量参照，不作主底座 |

> 评级是工程判断，不是客观分数；公开 CER/SS 的测试集和口径不同，不能直接横比。

### 16.3 决策规则

按你的优先级选：

1. **单说话人实时通话，质量/并发/生态要平衡**
   - 选 **CosyVoice3-0.5B**。
   - 这是当前最均衡的底座：音色/方言/instruct 强，官方 runtime 完整，0.5B 训练和推理成本可控。

2. **并发是硬指标，质量可以略让**
   - 先压 **Qwen3-TTS-0.6B** 或其它 12Hz 模型。
   - 25Hz → 12Hz 直接少一半 token；但要把官方缺失的流式/并发工程补上。
   - 如果 Qwen 0.6B 的首包/稳定性不达标，回到 v3 + Triton 优化。

3. **音色/情感是硬指标，算力可以加**
   - 先 A/B **Qwen3-TTS-1.7B** 和 **IndexTTS2-1.5B**。
   - 如果 1.7B 在自有数据上 MOS/SECS 明显更好，接受更低的并发和更多的工程投入；后续再考虑蒸馏到 0.6B。
   - 如果差距不到一个可感知档位，不要为 3 倍参数买单。

4. **需要 18+ 方言、instruct、发音修复**
   - 选 **CosyVoice3**。
   - 这是 v3 相对 Qwen/IndexTTS2 的明确差异点。

5. **要最快上线、最稳 SFT**
   - 先用 **CosyVoice2-0.5B** 打通，再切 v3。
   - v2 的 `inference_sft` 不需要补丁，适合做回退和对照。

6. **已有 Qwen/vLLM 基础设施**
   - Qwen3-TTS 1.7B/0.6B 的生态协同更好。
   - 但要预留流式/并发工程预算，不要假设官方包能直接上生产。

### 16.4 三周 bake-off（选型必须做）

用同一批数据、同一张 GPU、同一套文本，跑：

| 项 | 内容 |
|---|---|
| 数据 | 100–200 句真实通话文本；目标说话人 1–3 小时 |
| 模型 | CosyVoice3-0.5B、CosyVoice2-0.5B、Qwen3-TTS-1.7B、Qwen3-TTS-0.6B（可选 IndexTTS2） |
| 训练 | 各自官方 SFT 或 zero-shot；记录数据量、步数、显存、耗时 |
| 质量 | SECS、CER/WER、UTMOS/DNSMOS、5 人盲测 MOS/CMOS、情感依从率 |
| 实时 | 首包 p50/p95/p99、RTF、inter-chunk gap、30 秒长文本 |
| 并发 | 1/2/4/8 路；GPU 利用率、显存、队列、错误率 |
| 决策 | 按业务权重打分；质量 40%、性能 35%、生态 25%（示例） |

不要只看公开 CER/SS；公开集和你的通话文本、音色、情感分布可能完全不同。

### 16.5 演进路线建议

- **主线：CosyVoice3-0.5B**
  - 数据飞轮：真实通话失败案例回流。
  - instruct 体系：固定人设 → 粗粒度情绪 → 细粒度情绪。
  - 性能：token2wav batching、Triton/TRT-LLM、分离部署、首句拆分。
  - 保持 `frontend_sft` / `inference_sft` 补丁的可维护性，锁 commit。

- **质量挑战线：Qwen3-TTS-1.7B / IndexTTS2-1.5B**
  - 每季度用同一套测试集复测。
  - 如果明显胜出，评估蒸馏到 0.6B 或直接切换。

- **并发挑战线：Qwen3-TTS-0.6B 或其它 12Hz 模型**
  - 如果单卡并发是硬指标，这条线可能比继续优化 25Hz token2wav 更划算。

- **架构上保留可替换层**
  - 文本前端、模型接口、流式协议、评测集解耦；不要把所有业务逻辑绑死在 CosyVoice 的 `inference_*` 上。
  - 这样从 v3 切 Qwen/IndexTTS2 时，数据管线和评测可以复用。

**一句话结论：** 以 **CosyVoice3-0.5B 为主底座**，CosyVoice2 做稳定回退，Qwen3-TTS 1.7B/0.6B 分别做质量/并发挑战者；先用三周 bake-off 在自有数据上验证，再决定是否把主底座迁到更大或更低 token 率的模型。

---

## 17. CosyVoice3-0.5B SFT 算力与时间估算

> 下面是工程估算，不是官方 benchmark；实际值受 GPU、数据长度、是否预计算特征、batch、epoch 影响，误差按 ±50% 看。

### 17.1 先看结论

| 目标 | 数据量 | 推荐算力 | 单次训练时间（LLM-only，10 epochs） | 项目周期（含数据/评测/迭代） |
|---|---|---|---|---|
| 最小验证 | 1–5 h | 1×24–48 GB | 5–60 min | 半天到 1 天 |
| 单说话人生产 | 10–30 h | 1×A100 80GB 或 4×A100 | 20 min–2 h | 2–4 天 |
| 高质量/强情感 | 50–100 h | 4–8×A100 | 1–7 h | 1–2 周 |

关键判断：

- **单说话人 SFT，1–5 小时干净数据就能看到发音/节奏适配；10–30 小时主要用于覆盖情绪/语速/文本域。**
- **先只训 LLM**，flow / hifigan 冻结。LLM SFT 最直接改变发音、停顿、语速和文本域；音色和细腻情绪主要由 speaker embedding + flow 决定，若这两项不够再训 flow。
- 官方配置的 `warmup_steps: 2500`、`max_epoch: 200` 是通用配置，小数据 SFT 要改小，否则 warmup 就要跑很多 epoch。

### 17.2 显存需求

| 训练阶段 | 参数量（约） | 最低显存 | 推荐显存 | 说明 |
|---|---:|---:|---:|---|
| LLM SFT | 0.5B | 24 GB | 40–48 GB | 全参数 + Adam；24GB 需 micro-batch 1、`max_frames_in_batch` 降到 1000 左右 |
| Flow SFT | ~0.3B | 24 GB | 48–80 GB | DiT 全注意力，序列 2000 帧时显存随 T² 增长；24GB 要把帧数降到 500–1000 |
| HiFiGAN SFT | ~20M G + D | 24 GB | 24–48 GB | 只训 1s 左右片段；GAN 训练慢，通常最后才做 |
| 全流程 | ~0.8B | 48 GB | 80 GB 或多卡 | 分阶段训，不是三个模型同时放显存 |

注意：

- 仓库的 Qwen2Encoder / DiT **没有内置 gradient checkpointing**；24GB 全参数 SFT 会比较紧，靠减小 `max_frames_in_batch`、`accum_grad` 和序列长度。
- 多卡 DDP 是数据并行，**每张卡都要放一份完整模型和优化器状态**，不会降低单卡显存；要降显存用 DeepSpeed（`--train_engine deepspeed`）。
- 如果 24GB 不够，优先：冻结 LLM 底部若干层 → 只训顶部 + `llm_decoder`/`speech_embedding` → LoRA（需自己加）→ 换 48GB 卡。

### 17.3 时间估算：LLM-only SFT

假设：

- 全参数 SFT、AMP、`max_frames_in_batch=2000`、`accum_grad=2`。
- 一个 optimizer step 处理 4000 个 speech token ≈ 160 秒音频。
- 10 epochs；数据已预计算 token/embedding，不做在线特征提取。
- 1 小时音频 ≈ 90,000 个 25Hz speech token。

| 数据量 | 1×4090 24GB | 1×L20 48GB | 1×A100 80GB | 4×A100 80GB |
|---:|---:|---:|---:|---:|
| 1 h | 5–15 min | 3–10 min | 2–5 min | 1–2 min |
| 5 h | 25–60 min | 15–35 min | 8–20 min | 3–7 min |
| 10 h | 1–2 h | 30–70 min | 20–40 min | 6–12 min |
| 30 h | 3–6 h | 1.5–3.5 h | 1–2 h | 20–40 min |
| 100 h | 10–20 h | 5–12 h | 3–7 h | 1–2.5 h |

换算：

- **20 epochs 翻倍，50 epochs 乘 5。**
- 4×A100 的扩展效率按 0.7–0.8 算，不是线性 4×。
- 4090 受 24GB 显存限制，batch 小，吞吐不一定比 L20 高；L20 的 48GB 能吃更大 batch。

### 17.4 加上 flow / hifigan 的倍数

| 训练范围 | 相对 LLM-only 时间 | 何时需要 |
|---|---:|---|
| 只训 LLM | 1× | 首选；发音/停顿/语速/文本域基本够；音色/情绪细节看 flow |
| LLM + Flow | 2–3× | LLM 训完后音质、音色稳定性、长句韵律不够 |
| LLM + Flow + HiFiGAN | 3–6× | 有明显 vocoder 伪影、金属音、气息不自然 |

实际项目里不要一上来三阶段全训；先 LLM，听评测，再决定是否加 flow。

### 17.5 数据准备与存储

| 数据量 | 预处理时间（8–16 线程/单 GPU） | 存储（WAV 24k mono 16bit + parquet 特征） |
|---:|---:|---:|
| 1 h | 5–15 min | 0.2–0.5 GB |
| 10 h | 1–2 h | 2–5 GB |
| 100 h | 8–24 h | 20–50 GB |

预处理包括：

- `prepare_data.py`：生成 `wav.scp/text/utt2spk/spk2utt`，v3 必须带 instruct。
- `extract_embedding.py`：CPU ONNX，提取 CAM++ speaker embedding。
- `extract_speech_token.py`：GPU ONNX，提取 25Hz speech token。
- `make_parquet_list.py`：打包 parquet。
- 可选预计算 `speech_feat` / `whisper_feat` / `f0`；官方支持在线提取，但会明显拖慢训练。

### 17.6 小数据 SFT 的配置修正

官方 `cosyvoice3.yaml` 是通用配置，单说话人小数据建议：

| 参数 | 官方值 | 小数据建议 |
|---|---:|---:|
| `use_spk_embedding` | False | **True** |
| `lr` | 1e-5 | 1e-5（不要更大） |
| `scheduler` | constantlr | constantlr |
| `warmup_steps` | 2500 | **100–500** |
| `max_epoch` | 200 | **10–30**，按验证集早停 |
| `max_frames_in_batch` | 2000 | 24GB 用 1000–1500；48GB+ 用 2000 |
| `accum_grad` | 2 | 1–4，按显存调 |
| `grad_clip` | 5 | 5 |

另外：

- 每条样本都必须有 instruct；固定人设可以用常量 `"You are a helpful assistant.<|endofprompt|>"`。
- 动态情绪则按句写 instruct，例如 `"用温柔、稍慢的语气说。<|endofprompt|>"`。
- 验证集不要只用 CER；加 SECS 和盲听 MOS，否则容易“字准但不像”。

### 17.7 用你自己的 GPU 估算

先跑 100 个 optimizer step，测 `s/step`，再算：

```text
steps_per_epoch ≈ 数据小时数 × 3600 / 160
total_hours ≈ steps_per_epoch × epochs × s_per_step / 3600
```

例：10 小时数据，`steps_per_epoch ≈ 225`；10 epochs、`0.5 s/step` → `225 × 10 × 0.5 / 3600 ≈ 0.31 h`。

实际再加 30–50% 的数据加载、验证、checkpoint 开销。

### 17.8 预算建议

- **POC**：1–5 小时数据，1×24–48GB，半天到 1 天；先验证数据和 patch 是否跑通。
- **生产单说话人**：10–30 小时数据，1×A100 或 4×A100，2–4 天；含 2–3 轮参数/数据迭代。
- **高质量**：50–100 小时数据，4–8×A100，1–2 周；含 flow 微调、盲测、失败案例回流。
- **算力预算按单次训练的 3–5 倍准备**，因为选型、调参、重训、评测都会重复消耗。



