# 配置参考

服务配置使用 YAML（默认 `configs/service.yaml`），由
`src/native_voice/config/schema.py` 中的 dataclass 校验。所有相对路径都以
**配置文件所在目录的上一级**（即仓库根目录）为基准解析。

```bash
python -m native_voice.cli.main serve --config configs/service.yaml
# 或设置 NV_CONFIG
```

## 1. 配置节总览

| 节 | 作用 |
|---|---|
| `server` | 监听地址、CORS、消息大小、WebSocket 空闲超时 |
| `auth` | Key 数据库、开发 Key、短票 |
| `quotas` | 全局/单 Key 并发、RPM、每日额度 |
| `models` | 默认模型路径与推理参数 |
| `tts` / `asr` | 对外默认音频格式与识别行为 |
| `engine` | ASR/TTS engine 拓扑与 provider 选择 |
| `providers` | 每种 ASR/TTS 实现的解释器、模型与选项 |
| `llm` | llama-server 启动参数与生成参数 |
| `conversation` | VAD、端点、打断、附和 |
| `voice_store` | 克隆音色存储 |
| `logging` | 日志级别与 JSON 格式 |

## 2. `server`

| 字段 | 默认 | 说明 |
|---|---|---|
| `host` | `127.0.0.1` | 监听地址；对外服务时通常放在反向代理后 |
| `port` | `8090` | HTTP/WS 端口 |
| `cors_origins` | `[]` | 允许的浏览器来源；同源部署可留空 |
| `max_message_bytes` | `65536` | 单帧上限，防止内存滥用 |
| `ws_idle_timeout_s` | `60` | WebSocket 空闲超时 |
| `graceful_shutdown_s` | `10` | 退出时等待连接关闭的时间 |

## 3. `auth` 与 `quotas`

| 字段 | 默认 | 说明 |
|---|---|---|
| `auth.db_path` | `.data/native_voice.db` | SQLite Key 数据库 |
| `auth.dev_key` | `false` | 数据库为空时是否创建并打印开发 Key |
| `auth.ticket_ttl_s` | `60` | 对话短票有效期 |
| `auth.ticket_secret` | `""` | HMAC 密钥；空 = 每进程随机（仅单副本） |
| `quotas.max_concurrent_tasks` | `4` | 全局并发任务上限 |
| `quotas.max_concurrent_per_key` | `2` | 单 Key 并发上限 |
| `quotas.rpm_per_key` | `120` | 单 Key 每分钟请求数 |
| `quotas.tts_chars_per_day` | `200000` | 单 Key 每日 TTS 字符数 |
| `quotas.asr_seconds_per_day` | `14400` | 单 Key 每日 ASR 秒数 |

多副本部署时 `ticket_secret` 必须一致，并且所有副本需要共享 Key 数据库
（当前 SQLite 仅适合单节点，生产可替换为 Postgres，见 [deployment.md](deployment.md)）。

## 4. `models` / `tts` / `asr`

| 字段 | 默认 | 说明 |
|---|---|---|
| `models.asr_stream_model` | `iic/speech_paraformer-...-online` | FunASR 流式模型 ID |
| `models.asr_device` | `cuda:0` | ASR 设备 |
| `models.asr_chunk_ms` | `600` | 流式 ASR 分块刷新粒度 |
| `models.tts_model` | `models/Fun-CosyVoice3-0.5B` | 本地 CosyVoice 路径 |
| `models.tts_prompt_wav` | `third_party/CosyVoice-main/asset/zero_shot_prompt.wav` | 默认音色参考音频 |
| `models.tts_prompt_text` | 见配置 | 参考音频对应文本 |
| `models.tts_hop` | `15` | 首块语音 token 数 |
| `models.use_cuda_graph` | `true` | TTS 解码器 CUDA graph |
| `models.warmup` | `true` | 启动时预热模型 |
| `tts.default_format` | `pcm` | 默认输出格式：`pcm/wav/mp3/opus` |
| `tts.default_sample_rate` | `24000` | 默认输出采样率 |
| `asr.max_sentence_silence_ms` | `1300` | 百炼兼容 ASR 的分句静音阈值 |
| `asr.final_pass` | `streaming` | 定稿方式：`streaming` 或 `sensevoice` |

## 5. `engine` 与 `providers`

`engine.mode`：

- `spawn`：网关启动 ASR/TTS 子进程，适合开发/单机。
- `external`：engine 已由 Docker/systemd 启动，网关只连接，适合生产。

| 字段 | 默认 | 说明 |
|---|---|---|
| `engine.asr_provider` | `paraformer` | 当前 ASR 实现 |
| `engine.tts_provider` | `cosyvoice3` | 当前 TTS 实现 |
| `engine.asr_http/ws` | `:8101` | ASR engine 内部地址 |
| `engine.tts_http/ws` | `:8102` | TTS engine 内部地址 |
| `engine.token` | `""` | 内部 engine 共享 token |

**选择优先级**：命令行 `--asr-provider/--tts-provider` > 环境变量
`NV_ASR_PROVIDER/NV_TTS_PROVIDER` > YAML。

`providers` 为每种实现声明独立解释器和选项。不同模型族的
`transformers/torch` 版本可能冲突，因此可以用不同 venv：

```yaml
providers:
  asr:
    qwen3asr:
      python: envs/qwen3asr/Scripts/python.exe
      model: models/Qwen3-ASR-0.6B
      options:
        device: cuda:0
        dtype: bfloat16
  tts:
    cosyvoice3:
      python: .venv/Scripts/python.exe
      model: models/Fun-CosyVoice3-0.5B
```

Linux 下 `Scripts/python.exe` 不存在时，supervisor 会自动映射到同目录的
`bin/python`。各 provider 的选项与限制见 [models.md](models.md)。

## 6. `llm`

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 是否启用 LLM |
| `mode` | `spawn` | `spawn` 由网关启动；`external` 连接已有服务 |
| `server` | `bin/llama/llama-server.exe` | llama-server 二进制 |
| `model` | `models/Qwen3.5-2B-Q8_0.gguf` | GGUF 模型 |
| `port` | `8081` | llama-server 端口 |
| `threads` / `batch_threads` | `8` / `16` | CPU 线程 |
| `ctx` | `16384` | 总上下文，按 `slots` 平分 |
| `slots` | `2` | 并发槽位 |
| `n_gpu_layers` | `0` | GPU 层数；`99` = 全部卸载 |
| `kv_cache_type` | `f16` | `f16/q8_0/q4_0` |
| `batch_size` / `ubatch_size` | `2048` / `512` | llama.cpp batch |
| `max_tokens` | `160` | 每轮回复上限 |
| `prompt_char_budget` | `6000` | system + 历史字符预算 |
| `temperature` / `top_p` | `0.7` / `0.9` | 采样 |
| `history_turns` | `8` | 保留的历史轮数 |
| `tools` | `[get_current_time, web_search, web_fetch]` | 启用的内置工具；空列表关闭工具调用 |
| `tool_choice` | `auto` | 首轮工具选择：`auto` / `required` / `none` |
| `max_tool_rounds` | `3` | 单轮用户输入最多执行几轮工具（`0` 关闭） |
| `tool_filler` | `我查一下。` | 工具执行前先说的话，避免静默；空串关闭 |
| `system_prompt` | 见配置 | 系统提示词 |

工具调用说明：

- 网关会给 llama-server 加 `--jinja`，由 GGUF 内嵌的 chat template 解析工具调用；
  老版本或未带模板的模型无法使用。
- `tool_choice: required` 只作用于用户回合的第一轮，工具结果返回后自动切回 `auto`，
  避免模型反复调用。它适合调试，不适合作为生产默认值。
- 默认的 Qwen3.5-2B 在默认系统提示词下实测可正确触发工具调用；模型越小或提示词
  越复杂，触发越不稳定，必要时换用更大的 GGUF 或精简 `system_prompt`。
- 工具执行在网关进程内完成，需保持轻量；具体实现与新增方式见
  [development.md](development.md#新增内置工具)。
- `web_search` 通过 Bing 的 RSS 接口（`?format=rss`）搜索，默认
  `https://cn.bing.com/search`，可用环境变量 `NV_WEB_SEARCH_URL` 指向兼容端点
  （中文查询会自动附加 `mkt=zh-CN`）；`web_fetch` 只允许公网 http/https 地址，
  拒绝内网、回环、链接本地（含云元数据地址）和带用户名的 URL，并限制重定向、
  响应大小与超时。
- 两个联网工具默认都启用。离线部署或不需要联网时，把 `tools` 改成
  `[get_current_time]` 或空列表。

8 GB 显存场景通常把 LLM 放在 CPU：

```powershell
$env:LLM_DEVICE = "cpu"
scripts\serve_dev.ps1 -Backend worker
```

或手动：

```bash
python -m native_voice.cli.main serve --llm-gpu-layers 0 \
  --llm-server bin/llama/llama-server.exe
```

## 7. `conversation`

| 字段 | 默认 | 说明 |
|---|---|---|
| `endpoint_silence_ms` | `300` | 能量静音判停 |
| `noise_override_ms` | `500` | 高能量低语音概率时的兜底判停 |
| `merge_window_s` | `1.6` | 未出声前碎片合并窗口 |
| `barge_db` | `-30.0` | 打断能量阈值（dBFS） |
| `barge_sustain_ms` | `400` | 持续多久才认为在说话 |
| `barge_content_wait_ms` | `900` | 无 ASR partial 时的兜底打断等待 |
| `min_speech_ms` | `240` | 最短语音段 |
| `max_utterance_s` | `20` | 单句最长时长 |
| `preroll_ms` | `450` | 打断时保留的句首音频 |
| `voice` | `default` | 默认音色 |
| `speech_gate` | `true` | 启用 Silero 语音概率门控 |
| `speech_gate_model` | `assets/silero_vad.onnx` | Silero ONNX 模型 |
| `backchannel_enabled` | `true`（YAML） | 是否启用“嗯/我在听”附和 |
| `backchannel_min_speech_s` | `1.5` | 用户持续说话多久后附和 |
| `backchannel_interval_s` | `6.0` | 两次附和最小间隔 |
| `backchannel_max_per_utterance` | `2` | 单句最多附和次数 |
| `backchannel_volume` | `0.45` | 附和音量 |

> 代码默认 `backchannel_enabled=false`，仓库 YAML 显式设为 `true`。
> 外放环境可能被麦克风重新采集，建议耳机或关闭该功能。

## 8. `voice_store` 与 `logging`

| 字段 | 默认 | 说明 |
|---|---|---|
| `voice_store.dir` | `.data/voices` | 克隆音色目录；网关与 TTS engine 必须共享 |
| `voice_store.max_upload_mb` | `20` | 上传大小上限 |
| `voice_store.min_seconds` | `1.5` | 参考音频最短时长 |
| `voice_store.max_seconds` | `30` | 参考音频最长时长 |
| `logging.level` | `INFO` | 日志级别 |
| `logging.json_logs` | `true` | JSON 结构化日志；开发时可关 |

## 9. 环境变量

| 变量 | 对应配置 |
|---|---|
| `NV_CONFIG` | 配置文件路径（默认 `configs/service.yaml`） |
| `NV_HOST` / `NV_PORT` | `server.host` / `server.port` |
| `NV_LOG_LEVEL` | `logging.level` |
| `NV_DB_PATH` | `auth.db_path` |
| `NV_DEV_KEY` | `auth.dev_key`（`1/true/yes`） |
| `NV_TICKET_SECRET` | `auth.ticket_secret` |
| `NV_TTS_MODEL` | `models.tts_model` |
| `NV_TTS_PROMPT_WAV` | `models.tts_prompt_wav` |
| `NV_ASR_DEVICE` | `models.asr_device` |
| `NV_ASR_PROVIDER` | `engine.asr_provider` |
| `NV_TTS_PROVIDER` | `engine.tts_provider` |
| `NV_LLM_GPU_LAYERS` | `llm.n_gpu_layers` |
| `NV_LLM_SERVER` | `llm.server` |

脚本层还有：

| 变量 | 说明 |
|---|---|
| `LLM_DEVICE=cpu\|gpu` | `serve_dev` 脚本选择 CPU/CUDA llama.cpp 与 `-ngl` |
| `LLM_SERVER` | 覆盖 llama-server 路径 |
| `NV_HF_ENDPOINT` | HuggingFace 镜像（默认 `https://huggingface.co`） |
| `NV_MS_ENDPOINT` | ModelScope 镜像（默认 `https://www.modelscope.cn`） |

## 10. 音色配置

`configs/voices.yaml` 定义内置音色：

```yaml
voices:
  default:
    description: 本地默认中文女声
    prompt_wav: third_party/CosyVoice-main/asset/zero_shot_prompt.wav
    prompt_text: "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
```

克隆音色保存在 `.data/voices/voices.json`，无需手写配置。参考文本必须与
参考音频内容一致。
