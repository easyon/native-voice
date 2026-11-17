# 手工检查（GPU / 声卡）

这些脚本需要本地模型、`third_party/`、`bin/` 和样例音频，**不参与 CI**。
命名使用 `check_*`，不会被 `pytest` 自动收集。

## 前置条件

- 已运行 `python scripts/fetch_deps.py`；
- conda `cosyvoice` 环境（或安装了全部 GPU 依赖的 `.venv`）；
- `audio/asr_test.wav` 等样例音频存在；
- 在仓库根目录运行。

## 脚本

| 脚本 | 检查内容 |
|---|---|
| `check_pipeline.py` | wav → VAD → ASR → LLM → TTS → ASR 回读，打印各阶段耗时与显存 |
| `check_stream_asr.py` | 流式 Paraformer 的 partial 与 final |
| `check_barge.py` | 以文字模式驱动 `desktop/voice_call.py`，验证打断 |
| `check_tts.py` | 直接加载 CosyVoice3，对比 stream / non-stream |

```bash
python tests/manual/check_pipeline.py
python tests/manual/check_stream_asr.py
python tests/manual/check_barge.py
python tests/manual/check_tts.py
```

`check_pipeline.py` 会把合成音频写到 `audio/selftest/`。

## 与自动化测试的区别

- `tests/unit/` 与 `tests/protocol/` 使用 fake 后端，无需 GPU，CI 会运行。
- `tests/manual/` 依赖真实模型和声卡，结果受硬件/驱动/网络影响。
