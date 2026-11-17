# 调优与基准脚本

这些脚本记录了 CosyVoice3 / FunASR 优化过程中的对比实验，**不是**稳定的公共
接口，也不参与 CI。修改模型代码或复现性能数据时使用。

## 前置条件

- 在仓库根目录运行（脚本通过 `__file__` 推导根目录）；
- 已安装 conda `cosyvoice` 环境；
- 已下载 `models/`、`third_party/CosyVoice-main/`；
- 需要样例音频的脚本要求 `audio/asr_test.wav` 存在。

## 脚本

| 脚本 | 用途 |
|---|---|
| `bench_cpu.py` | CPU 合成基线 |
| `bench_gpu.py` | GPU 合成 + ASR 基线 |
| `bench_core.py` | 核心模型加载/推理耗时 |
| `bench_graph.py` | CUDA graph 对 TTS 解码的影响 |
| `bench_llm.py` | LLM + TTS 流水线 |
| `bench_profile.py` / `bench_prof2.py` | 采样器与子模型 profiling |
| `bench_step.py` / `bench_step2.py` | 分步延迟拆解 |
| `bench_tts2.py` / `bench_tts3.py` | TTS 首块/RTF 对比 |

```bash
conda activate cosyvoice
python benchmarks/bench_gpu.py
```

脚本会把音频、日志写到 `audio/`（已被 `.gitignore` 忽略）。
实测结论已整理到 [`docs/models.md`](../docs/models.md)。
