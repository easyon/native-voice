# 脚本

| 脚本 | 用途 |
|---|---|
| `fetch_deps.py` | 下载模型、llama.cpp、CosyVoice 源码等本地依赖 |
| `serve_dev.ps1` / `serve_dev.sh` | 开发模式启动网关（可自动拉起 engine 和 LLM） |
| `setup_qwen_engines.ps1` / `.sh` | 创建 Qwen3-TTS / Qwen3-ASR 独立 venv 并下载模型 |
| `create_key.py` | 创建 API Key 的便捷包装 |
| `load_test.py` | ASR/TTS 并发压测 |

## 常用

```bash
# 检查缺件
uv run --extra gpu python scripts/fetch_deps.py --check

# 启动开发服务
scripts/serve_dev.ps1 -Backend local
BACKEND=worker ASR_PROVIDER=paraformer TTS_PROVIDER=cosyvoice3 scripts/serve_dev.sh

# 独立 Qwen engine 环境
scripts/setup_qwen_engines.ps1 -Kind both -DownloadModels

# 创建 Key
python scripts/create_key.py --name demo --scopes asr,tts,conversation

# 压测
python scripts/load_test.py --key "$NV_KEY" tts -n 4
```

环境变量与 provider 组合见 [`docs/configuration.md`](../docs/configuration.md)
和 [`docs/models.md`](../docs/models.md)。
