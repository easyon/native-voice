# 第三方组件与模型

本仓库的源码采用 Apache License 2.0。以下组件由脚本下载或在运行时使用，
**不包含在本仓库中**，请以各自上游仓库的许可为准：

| 组件 | 用途 | 上游 |
|---|---|---|
| FunASR | 流式 Paraformer、SenseVoice | https://github.com/modelscope/FunASR |
| CosyVoice | CosyVoice3 TTS 源码与模型 | https://github.com/FunAudioLLM/CosyVoice |
| Matcha-TTS | CosyVoice 依赖 | https://github.com/shivammehta25/Matcha-TTS |
| Qwen3.5-2B | LLM（GGUF，unsloth 量化） | https://www.modelscope.cn/models/Qwen/Qwen3.5-2B |
| Qwen3-ASR | ASR 模型 | https://github.com/QwenLM/Qwen3-ASR |
| Qwen3-TTS | TTS 模型 | https://github.com/QwenLM/Qwen3-TTS |
| llama.cpp | LLM 推理二进制 | https://github.com/ggml-org/llama.cpp |
| vLLM / vLLM-Omni | 真流式 TTS 服务 | https://github.com/vllm-project/vllm |
| Silero VAD | 语音概率门控 | https://github.com/snakers4/silero-vad |

## 模型权重

- 模型权重不随仓库分发，由 `scripts/fetch_deps.py` 或用户手动下载。
- 模型的商用许可、再分发条件和地域限制由模型发布方决定。
- 声音克隆功能仅用于获得授权的声音；使用者需自行确认授权与合规性。

## 第三方源码

`third_party/` 由脚本克隆到本地，并被 `.gitignore` 排除。修改或分发这些代码时，
请保留其原始版权与许可声明。
