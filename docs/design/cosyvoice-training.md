# CosyVoice 训练技术方案

> 目标：基于 `Fun-CosyVoice3-0.5B-2512` 做单说话人/少说话人 SFT，产出可交付给运行时的模型包；覆盖数据清洗、环境准备、训练、评测和发布。  
> 关联文档：[总体方案](./cosyvoice-overview.md) · [运行时方案](./cosyvoice-runtime.md)

---

## 1. 目标与原则

### 1.1 目标

- 让模型学会目标说话人的**音色、发音习惯、语速、停顿和常见情绪**。
- 在真实通话文本上稳定、自然，不出现明显复读、漏字、怪停顿。
- 训练过程可复现、可回滚，模型包可被运行时直接加载。
- 用尽量少的数据和算力达到可上线质量：**1–5 小时可验证，10–30 小时可生产，50–100 小时做高质量**。

### 1.2 原则

1. **数据质量 > 数据数量**。1 小时干净数据通常优于 10 小时脏数据。
2. **先 LLM，后 Flow，最后 HiFiGAN**。不要一上来三阶段全训。
3. **每一条样本都必须有 instruct**。CosyVoice3 的 `CosyVoice3LM.forward` 无条件读取 `instruct_token`。
4. **训练集/验证集/测试集按会话或说话人切分**，避免同句泄漏。
5. **评测必须包含主观盲听**。CER 低不代表像目标说话人，也不代表有情绪。
6. **模型版本与训练数据、配置、评测报告绑定**，运行时可回滚。

---

## 2. 训练流水线

```text
数据采集/授权
   │
   ▼
清洗：解码 → 单声道 24k → 降噪/去混响 → VAD → 切分 → 响度归一
   │
   ▼
转写：ASR → 人工校对 → 标点/填充词 → instruct 标注
   │
   ▼
质量门禁：SNR/削波/时长/说话人一致性/转写 CER
   │
   ▼
数据格式：wav.scp/text/utt2spk/spk2utt/instruct
   │
   ▼
特征：CAM++ embedding + speech token + parquet
   │
   ▼
Stage 1：LLM SFT（必做）
   │
   ├── 评测：CER/SECS/MOS/节奏
   │
   ▼
Stage 2：Flow SFT（按需）
   │
   ├── 评测：音质/音色/情绪/长句
   │
   ▼
Stage 3：HiFiGAN SFT（按需）
   │
   ▼
平均 checkpoint → 导出 → 模型包 → 运行时
```

---

## 3. 数据规范

### 3.1 来源与授权

- 只使用有明确授权/合规来源的说话人数据。
- 声音复刻/克隆需留存授权声明、用途、有效期。
- 不使用未授权的公众人物声音。
- 数据脱敏：去除姓名、电话、地址等敏感信息；转写文本同样脱敏。

### 3.2 采集要求

| 项 | 要求 | 说明 |
|---|---|---|
| 采样率 | ≥24 kHz | 16 kHz 上采样不会增加带宽；电话 8 kHz 音质上限低 |
| 位深 | 16/24 bit PCM | 避免有损压缩反复转码 |
| 声道 | 单声道 | 多人对话先做分离/分段 |
| 环境 | 安静、低混响 | 避免音乐、键盘、空调、回声 |
| 麦克风 | 固定 | 同说话人尽量同一设备/距离 |
| 内容 | 覆盖通话场景 | 陈述、疑问、感叹、犹豫、快慢、长句、数字、专名 |
| 情绪 | 有分布 | 不要全是中性朗读；也不要全是夸张表演 |
| 时长 | 单段 2–15 s，目标 3–10 s | 太短缺上下文，太长难对齐 |

### 3.3 清洗流水线

| 步骤 | 操作 | 工具/阈值 | 失败处理 |
|---|---|---|---|
| 解码 | 统一转 WAV/PCM | ffmpeg | 解码失败丢弃 |
| 重采样 | 单声道 24 kHz | ffmpeg/sox | — |
| 去直流/去爆音 | high-pass 60–80 Hz | ffmpeg | — |
| 降噪/去混响 | 轻量降噪，避免过度 | DNS/RNNoise（谨慎） | 过度降噪会伤音色 |
| 响度归一 | 集成响度 -23 LUFS，真峰值 ≤ -1 dBTP | ffmpeg loudnorm | 削波丢弃 |
| VAD | 去除首尾长静音，保留自然停顿 | Silero VAD | — |
| 切分 | 按停顿切 2–15 s | 静音检测/强制对齐 | 不从词中间切 |
| 噪声评估 | SNR ≥ 20 dB 优先 | 估算 | <15 dB 丢弃 |
| 削波检测 | 连续满幅样本 <0.1% | 波形统计 | 超标丢弃 |
| 说话人一致性 | CAM++ 余弦 ≥0.75 | 与目标音色比对 | 低分丢弃/复核 |
| 去重 | 文本/音频近重复 | hash + embedding | 保留质量更高的一条 |
| 转写 | ASR + 人工校对 | Paraformer/Whisper | CER >5% 重转 |
| 标点/填充词 | 保留 `，。？！……`、`嗯/那个` | 人工/规则 | 不要过度规范化 |
| instruct | 逐句情绪/风格标签 | 模板 | 每条必须有 |

注意：

- **降噪不是越干净越好**。过度降噪会损失气息、齿音和音色细节，导致 SFT 后声音“塑料”。
- **不要把所有停顿都切掉**。自然停顿是节奏信息；只去首尾长静音。
- **转写必须和音频对齐**。如果音频有“嗯……”而文本没有，模型会学到错位的停顿。
- **电话窄带数据要单独评估**。8 kHz 数据可以训，但高频和齿音上限明显。

### 3.4 转写与 instruct

#### 转写

- ASR 首选中文强模型（Paraformer/Whisper large），再做人工校对。
- 保留标点：逗号、句号、问号、感叹号、省略号。
- 保留语气词/填充词：`嗯`、`那个`、`就是`、`让我想想`。
- 数字/专名：按实际读法转写，或统一规范后加 `hot_fix`。
- 中英混读：保留英文原词，必要时加空格。

#### instruct

CosyVoice3 每条样本必须有 instruct。推荐两种策略：

| 策略 | instruct | 适用 |
|---|---|---|
| 固定人设 | `"你是一个温柔、自然的客服助手。<|endofprompt|>"` | 单一人设，稳定 |
| 逐句风格 | `"用温柔、稍慢的语气说。<|endofprompt|>"` | 情绪/语速可控 |
| 组合 | `"你是XX客服，用耐心、稍慢的语气说。<|endofprompt|>"` | 生产推荐 |

要求：

- 训练和推理的 instruct 格式一致；不要训练用一套、推理用另一套。
- 情绪标签要平衡；中性占比过高会把情绪“平均掉”。
- 不要用模型没见过的抽象标签（如“第 7 号情绪”）；用自然语言。
- 若只做固定人设，推理时也用同一句 instruct，避免分布漂移。

### 3.5 数据质量门禁

| 指标 | 建议阈值 | 动作 |
|---|---:|---|
| 总时长 | ≥1 h（验证）；≥10 h（生产） | 不足则补数据 |
| 单段时长 | 2–15 s | 超范围切/丢 |
| SNR | ≥20 dB 优先，≥15 dB 可用 | 低分复核 |
| 削波比例 | <0.1% | 超标丢弃 |
| 转写 CER | <5% | 重转/人工校对 |
| 说话人相似度 | ≥0.75 | 低分丢弃 |
| 静音/噪声占比 | <20% | 重新切分 |
| 情绪覆盖 | 至少 3 类，单类 ≤60% | 补数据/调采样 |
| 重复率 | 近重复 <5% | 去重 |
| instruct 覆盖率 | 100% | 缺的补 |

### 3.6 数据格式

LibriTTS 风格：

```text
data/train/
  wav.scp
  text
  utt2spk
  spk2utt
  instruct
  utt2embedding.pt
  spk2embedding.pt
  utt2speech_token.pt
  parquet/
    data.list
```

示例：

```text
# wav.scp
utt001 /data/wav/utt001.wav

# text
utt001 你好，今天想了解点什么？

# utt2spk
utt001 spk001

# instruct
utt001 你是一个温柔、自然的客服助手。<|endofprompt|>
```

### 3.7 数据切分

- 按**会话/日期/说话人**切分，不按句子随机切。
- 验证集 5–10%；测试集 5–10%。
- 测试集覆盖：陈述、疑问、感叹、快/慢、犹豫、长句、数字、专名、方言/中英混读。
- 训练集和测试集不能有同一句或近重复句。

---

## 4. 环境准备

### 4.1 硬件

| 场景 | GPU | 说明 |
|---|---|---|
| 数据预处理 | CPU 16 核+ / 1×GPU | 特征提取可用 GPU ONNX |
| 最小验证 | 1×24 GB | LLM-only，`max_frames` 降到 1000 |
| 推荐生产 | 1×A100 80GB 或 4×A100 | 10–30 h 数据 |
| 高质量 | 4–8×A100 | 50–100 h + Flow SFT |

### 4.2 软件基线

| 项 | 建议 |
|---|---|
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| CUDA | 12.x（与 PyTorch/TensorRT 匹配） |
| PyTorch | 2.x（锁版本） |
| 依赖 | `requirements.txt` + `third_party/Matcha-TTS` |
| ONNX | `onnxruntime-gpu` |
| 分布式 | `torchrun` / DeepSpeed（可选） |
| 容器 | 训练镜像固定 CUDA/PyTorch/依赖 |

### 4.3 代码与模型

```bash
git clone https://github.com/QwenAudio/CosyVoice.git
cd CosyVoice
git checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc

pip install -r requirements.txt
cd third_party/Matcha-TTS && pip install -e . && cd -
```

模型目录：

```text
pretrained_models/Fun-CosyVoice3-0.5B/
  llm.pt
  flow.pt
  hifigan.pt
  campplus.onnx
  speech_tokenizer_v3.onnx
  CosyVoice-BlankEN/
```

### 4.4 存储

| 数据量 | 原始 WAV | 特征/parquet | checkpoint | 建议磁盘 |
|---:|---:|---:|---:|---:|
| 10 h | 2 GB | 5 GB | 20 GB | 100 GB |
| 100 h | 20 GB | 50 GB | 50 GB | 500 GB |

### 4.5 配置修改

`examples/libritts/cosyvoice3/conf/cosyvoice3.yaml`：

```yaml
padding:
  use_spk_embedding: True   # SFT 必须打开

train_conf:
  optim: adam
  optim_conf:
    lr: 1e-5
  scheduler: constantlr
  scheduler_conf:
    warmup_steps: 300      # 小数据 100–500；官方 2500 太大
  max_epoch: 20            # 10–30；按验证集早停
  grad_clip: 5
  accum_grad: 2

filter:
  max_length: 6000         # 25Hz 下 240s；SFT 建议自定义 2–15s
  min_length: 25           # 1s；官方 100=4s，会过滤短句
  token_max_length: 200
```

### 4.6 v3 SFT 必要补丁

#### 问题

- `CosyVoice3LM.forward` 无条件读取 `batch['instruct_token']`，缺 instruct 会 `KeyError`。
- `CosyVoice3LM.inference` 断言文本含 `<|endofprompt|>`（token 151646）。
- 原版 `frontend_sft` 不注入 instruct，`inference_sft` 也没有 `instruct_text` 参数。

#### 补丁：frontend_sft

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

#### 补丁：inference_sft

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

#### spk2info 构造

```python
import torch

spk2emb = torch.load('data/train/spk2embedding.pt', map_location='cpu')
spk2info = {
    spk: {'embedding': torch.tensor(emb, dtype=torch.float32).unsqueeze(0)}
    for spk, emb in spk2emb.items()
}
torch.save(spk2info, 'pretrained_models/Fun-CosyVoice3-0.5B/spk2info.pt')
```

或：

```python
from cosyvoice.cli.cosyvoice import CosyVoice3
model = CosyVoice3(
    'pretrained_models/Fun-CosyVoice3-0.5B',
    load_trt=False, load_vllm=False, fp16=False
)
model.add_zero_shot_spk('参考音频对应的文本', 'ref.wav', 'my_spk')
model.save_spkinfo()
```

---

## 5. 训练

### 5.1 Stage 0：数据准备

```bash
# 1) 生成 wav.scp/text/utt2spk/spk2utt + instruct
python examples/libritts/cosyvoice3/local/prepare_data.py \
  --src_dir data/raw/my_speaker \
  --des_dir data/train \
  --instruct "你是一个温柔、自然的客服助手。<|endofprompt|>"

# 2) CAM++ speaker embedding
python tools/extract_embedding.py \
  --dir data/train \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx

# 3) speech token
python tools/extract_speech_token.py \
  --dir data/train \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx

# 4) parquet
python tools/make_parquet_list.py \
  --num_utts_per_parquet 1000 \
  --num_processes 10 \
  --src_dir data/train \
  --des_dir data/train/parquet
```

校验：

```bash
wc -l data/train/wav.scp data/train/text data/train/instruct
ls data/train/parquet/data.list
```

### 5.2 Stage 1：LLM SFT（必做）

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nnodes=1 --nproc_per_node=4 \
  --rdzv_id=1986 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config examples/libritts/cosyvoice3/conf/cosyvoice3.yaml \
  --train_data data/train/parquet/data.list \
  --cv_data data/dev/parquet/data.list \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model llm \
  --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/llm.pt \
  --model_dir exp/cosyvoice3/llm/torch_ddp \
  --tensorboard_dir tensorboard/cosyvoice3/llm/torch_ddp \
  --ddp.dist_backend nccl \
  --num_workers 2 --prefetch 100 --pin_memory --use_amp
```

建议：

- 先 1–5 epoch 看 loss/验证集，再决定是否继续。
- 10–30 epoch 足够；不要盲目跑到 200。
- 每 epoch 导出/保存；保留 best 和 last。
- 监控：train loss、cv loss、CER、SECS、显存、吞吐。

### 5.3 Stage 2：Flow SFT（按需）

只在以下情况做：

- LLM 训完后，音色/音质仍不够像。
- 情绪、气息、重音不自然。
- 长句韵律、音色稳定性差。
- 有轻微 vocoder 伪影但 hifigan 不想动。

```bash
torchrun --nnodes=1 --nproc_per_node=4 \
  --rdzv_id=1987 --rdzv_backend=c10d --rdzv_endpoint=localhost:1235 \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config examples/libritts/cosyvoice3/conf/cosyvoice3.yaml \
  --train_data data/train/parquet/data.list \
  --cv_data data/dev/parquet/data.list \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model flow \
  --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/flow.pt \
  --model_dir exp/cosyvoice3/flow/torch_ddp \
  --tensorboard_dir tensorboard/cosyvoice3/flow/torch_ddp \
  --ddp.dist_backend nccl \
  --num_workers 2 --prefetch 100 --pin_memory --use_amp
```

显存不足：

- `max_frames_in_batch` 降到 500–1000。
- `accum_grad` 加到 4。
- 缩短训练片段（2–8s）。
- 用 48GB/80GB 卡。

### 5.4 Stage 3：HiFiGAN SFT（最后手段）

只在明显 vocoder 伪影、金属音、气息不自然时做：

```bash
torchrun --nnodes=1 --nproc_per_node=4 \
  --rdzv_id=1988 --rdzv_backend=c10d --rdzv_endpoint=localhost:1236 \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config examples/libritts/cosyvoice3/conf/cosyvoice3.yaml \
  --train_data data/train/parquet/data.list \
  --cv_data data/dev/parquet/data.list \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model hifigan \
  --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/hifigan.pt \
  --model_dir exp/cosyvoice3/hifigan/torch_ddp \
  --tensorboard_dir tensorboard/cosyvoice3/hifigan/torch_ddp \
  --ddp.dist_backend nccl \
  --num_workers 2 --prefetch 100 --pin_memory --use_amp
```

注意：

- GAN 训练 `accum_grad` 必须为 1。
- `truncate_length=24960`（约 1.04s），片段太短，训练慢。
- 容易不稳定；没有明显收益就回退。

### 5.5 模型平均与导出

```bash
python cosyvoice/bin/average_model.py \
  --dst_model exp/cosyvoice3/llm/torch_ddp/llm.pt \
  --src_path exp/cosyvoice3/llm/torch_ddp \
  --num 5 --val_best

python cosyvoice/bin/export_jit.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
python cosyvoice/bin/export_onnx.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
```

把训练产出的 `llm.pt`/`flow.pt`/`hifigan.pt` 放回模型目录，和 `spk2info.pt` 一起打包。

### 5.6 超参数建议

| 参数 | LLM | Flow | HiFiGAN |
|---|---:|---:|---:|
| `lr` | 1e-5 | 1e-5 | 2e-4 |
| `scheduler` | constantlr | constantlr | constantlr |
| `warmup_steps` | 100–500 | 100–500 | 100–500 |
| `max_epoch` | 10–30 | 5–15 | 5–20 |
| `max_frames_in_batch` | 1000–2000 | 500–1000 | — |
| `accum_grad` | 1–4 | 1–4 | 1 |
| `grad_clip` | 5 | 5 | 5 |

### 5.7 算力与时间

LLM-only，10 epochs，预计算特征：

| 数据量 | 1×4090 24GB | 1×L20 48GB | 1×A100 80GB | 4×A100 |
|---:|---:|---:|---:|---:|
| 1 h | 5–15 min | 3–10 min | 2–5 min | 1–2 min |
| 5 h | 25–60 min | 15–35 min | 8–20 min | 3–7 min |
| 10 h | 1–2 h | 30–70 min | 20–40 min | 6–12 min |
| 30 h | 3–6 h | 1.5–3.5 h | 1–2 h | 20–40 min |
| 100 h | 10–20 h | 5–12 h | 3–7 h | 1–2.5 h |

加上 Flow：约 2–3×；再加 HiFiGAN：约 3–6×。项目预算按单次训练的 3–5 倍准备。

显存：

| 阶段 | 最低 | 推荐 |
|---|---:|---:|
| LLM SFT | 24 GB | 40–48 GB |
| Flow SFT | 24 GB | 48–80 GB |
| HiFiGAN | 24 GB | 24–48 GB |

### 5.8 常见问题

| 问题 | 原因 | 处理 |
|---|---|---|
| `KeyError: instruct_token` | 样本缺 instruct | 每条补 instruct |
| `assert 151646 in text` | `frontend_sft` 没注入 EOP | 用 4.6 补丁 |
| `spk2info` KeyError | 没生成/没加载 | 构造 `spk2info.pt` |
| 训练不开始 | `run.sh` 默认 `stop_stage=3` | 改 `stop_stage` 或显式命令 |
| 平均模型路径错 | `run.sh` 用 `exp/cosyvoice` | 改成 `exp/cosyvoice3` |
| 导出文件名错 | `hifigan.pt` vs `hift.pt` | 以实际文件为准 |
| Loss NaN | FP16/学习率过大 | AMP 检查、降 lr、grad clip |
| 音色不像 | 只训 LLM / 数据差 | Flow SFT；检查 embedding 和参考音频 |
| 情绪平 | 数据中性/无 instruct | 补情绪数据 + 逐句 instruct |
| 复读/漏字 | 数据转写错位/过拟合 | 校对、降 epoch、加数据 |
| 长句崩 | 训练片段太短 | 加长句数据；Flow SFT |

---

## 6. 评测

### 6.1 客观指标

| 指标 | 工具 | 说明 |
|---|---|---|
| CER/WER | Paraformer/Whisper | 字准不准；中文 CER，英文 WER |
| SECS | CAM++/WavLM | 说话人相似度；音色像不像 |
| UTMOS/DNSMOS | 预训练评分器 | 自然度/音质参考 |
| F0 RMSE | 对参考音频 | 音高轮廓相似度 |
| 时长/语速 | 对齐统计 | 每字时长、整体语速 |
| 停顿分布 | VAD/对齐 | 停顿位置、次数、时长 |
| 情绪分类 | 情绪分类器 | 情绪可识别性 |
| 字级时间戳误差 | 强制对齐 | 若运行时提供该能力 |

### 6.2 主观评测

| 项 | 方法 | 样本 |
|---|---|---|
| MOS | 5 人+，1–5 分 | 50–100 句 |
| CMOS | 与 base/上一版对比 | 30–50 句 |
| ABX 节奏 | 同文本，哪个更像目标说话人节奏 | 30 句 |
| 情绪识别 | 听者说出情绪，统计准确率 | 每类 10–20 句 |
| 长文本 | 30–60s 连续听 | 10 段 |

盲测要求：随机顺序、同一音量、同一播放设备；不要让评测者知道版本。

### 6.3 测试集设计

至少 200–500 句，覆盖：

- 陈述、疑问、感叹、命令。
- 快语速、慢语速、长句、短句。
- 犹豫/思考：`嗯……那个……让我想想`。
- 情绪：温柔、兴奋、严肃、抱歉、耐心。
- 数字、电话、地址、金额、专名、英文缩写。
- 方言/中英混读（如业务需要）。
- 真实通话日志中的失败案例。

### 6.4 实时指标

训练侧只做模型相关；运行时另测：

- TTFC p50/p95/p99。
- RTF、inter-chunk gap。
- 1/2/4/8 路并发。
- 长文本稳定性。

### 6.5 验收门禁

| 指标 | 门槛 |
|---|---|
| CER | ≤ base + 0.5%（或绝对值 ≤2%） |
| SECS | ≥ base + 0.02 |
| MOS | ≥ base，且 ≥4.0（5 分制） |
| 情绪识别 | ≥80%（目标情绪） |
| 节奏 ABX | ≥60% 偏好 SFT |
| 长文本 | 无漏字/复读/崩溃 |
| TTFC p95 | ≤1.0s（4 路/L20 基线） |
| RTF | ≤0.10 |

任何一项不达标，不进入发布。

### 6.6 评测报告模板

```markdown
# 模型评测报告

- model_id:
- base_model:
- 训练数据: 小时数 / 说话人 / 情绪分布
- 训练配置: LLM/Flow/HiFiGAN, lr, epochs
- 测试集: 句数 / 场景

## 客观
| 指标 | base | SFT | 变化 |
|---|---:|---:|---:|
| CER | | | |
| SECS | | | |
| UTMOS | | | |
| F0 RMSE | | | |
| 停顿误差 | | | |

## 主观
| 项 | base | SFT |
|---|---:|---:|
| MOS | | |
| CMOS | | |
| 节奏 ABX | | |
| 情绪准确率 | | |

## 实时
| 并发 | TTFC p95 | RTF | gap p95 |
|---:|---:|---:|---:|

## 结论
- 通过/不通过
- 已知问题
- 推荐推理参数
```

---

## 7. 模型交付与版本管理

### 7.1 模型包

```text
cosyvoice3-sft-20261112/
  llm.pt
  flow.pt
  hifigan.pt
  spk2info.pt
  campplus.onnx
  speech_tokenizer_v3.onnx
  CosyVoice-BlankEN/
  config.yaml
  manifest.json
  model-card.md
  eval-report.md
```

### 7.2 版本规则

```text
cosyvoice3-sft-YYYYMMDD-<data>-<rev>
```

`manifest.json` 记录：

- base model、repo commit。
- 训练数据版本、小时数、说话人。
- 训练配置、GPU、耗时。
- 评测指标、已知问题。
- 推荐推理参数（`token_hop_len`、`pre_lookahead_len`、`fp32`）。

### 7.3 发布流程

1. 训练产出模型包。
2. 离线评测 + 盲听。
3. 运行时灰度：1% → 10% → 50% → 100%。
4. 对比 TTFC、错误率、SECS、CER、用户投诉。
5. 不达标回滚上一版 `model_id`。

---

## 8. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 数据脏/转写错 | 复读、漏字、怪停顿 | 质量门禁 + 人工校对 |
| 数据少 | 音色/情绪不稳 | 1–5h 验证；10–30h 生产 |
| 过拟合 | 机械、单调 | 早停、数据增强、保留多样性 |
| 0.5B 容量上限 | 多说话人/强情感不够 | 单说话人优先；必要时换更大模型 |
| Flow 未训 | 情绪/音色细节不足 | 按需 Flow SFT |
| 评测主观 | 结论不稳 | 盲测 + 多人 + 固定测试集 |
| 合规 | 声音侵权 | 授权、审计、脱敏 |
| 运行时瓶颈 | 训练好但上线并发低 | 见 [运行时方案 8.5](./cosyvoice-runtime.md#85-100--1000-并发容量方案) 的容量规划与压测 |

---

## 附录 A：数据目录建议

```text
data/
  raw/
    speaker01/
      session01/*.wav
  cleaned/
    speaker01/*.wav
  train/
    wav.scp text utt2spk spk2utt instruct
    utt2embedding.pt spk2embedding.pt utt2speech_token.pt
    parquet/data.list
  dev/
  test/
```

## 附录 B：训练前检查清单

- [ ] 数据授权/合规确认。
- [ ] 采样率、声道、响度、SNR、削波检查。
- [ ] 转写人工校对，标点/填充词保留。
- [ ] 每条样本有 instruct。
- [ ] `use_spk_embedding: True`。
- [ ] `spk2info.pt` 已生成。
- [ ] v3 `frontend_sft` / `inference_sft` 补丁已打。
- [ ] 训练/验证/测试按会话切分，无泄漏。
- [ ] 显存/磁盘/checkpoint 路径确认。
- [ ] 先跑 100 step，确认 loss/吞吐正常。
