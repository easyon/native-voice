# cosyvoice-common

共享的音频与文本工具，供 `cosyvoice-runtime`、`cosyvoice-data`、
`cosyvoice-training`、`cosyvoice-evaluation` 使用。

- `audio.py`：PCM/WAV/MP3/Opus 编码、增益、重采样、变调、格式/采样率校验。
- `text.py`：Markdown 过滤、hot_fix、SSML 子集、分句、流式 `SentenceBuffer`。
- `errors.py`：百炼兼容错误码与异常模型。

依赖方向：本包**不依赖**任何其他 CosyVoice 包。
