# 故障排查

## 启动与依赖

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: native_voice` | 未设置 `PYTHONPATH=src` 或未安装。用 `scripts/serve_dev.*`，或 `uv sync` 后使用 `.venv` 入口 |
| 缺少模型 / 二进制 | 运行 `python scripts/fetch_deps.py --check`，再按提示下载 |
| `uv run` 后 CUDA 不可用 | `uv run` 默认卸载 `gpu` extra；使用 `uv run --extra gpu ...` |
| `No available memory for the cache blocks` | 显存不足或 vLLM stage 预留过高；降低 `gpu_memory_utilization` / batch，或把 LLM 放 CPU |
| `/readyz` 一直是 503 | 查看 engine 日志；确认 `engine.mode`、端口、provider 解释器路径 |
| llama-server 端口被占用 | 修改 `llm.port`，或 `--port 8082` |
| 首次回答很慢 | 正常：system prompt 首次 prefill；后续走 prompt cache |

## 音频与打断

| 现象 | 处理 |
|---|---|
| 说话没反应 | `--verbose` 查看 VAD/ASR；`--mic N` 换设备；检查系统麦克风权限与音量 |
| 不戴耳机时自己把自己打断 | 戴耳机，或提高 `--barge-db`（如 `-20`）、`conversation.barge_db` |
| 一直不结束 / 结束太慢 | 降低 `--endpoint-ms`（450–500）或 `conversation.endpoint_silence_ms` |
| 说话开头被截掉 | 增大 `conversation.preroll_ms`（默认 450 ms） |
| 短噪声误触发 | 确认 `speech_gate=true` 且 `assets/silero_vad.onnx` 存在 |
| ASR 把 AI 回声当用户 | 戴耳机；确认客户端发送 `playback.started/done` |
| 没有实时字幕 | 默认有；检查 `--no-partials`、`--asr sensevoice` 或 ASR partial 指标 |
| 部分结果重复字 | 不要把 ASR 分块降到 480 ms 以下；保持 600 ms |
| 浏览器没有声音 | 检查浏览器自动播放策略；先点击页面再连接；查看控制台与 `tts.audio.start` |

## 网页与协议

| 现象 | 处理 |
|---|---|
| 页面 404 | 访问 `/ui/`；`web/` 目录必须在仓库根目录 |
| 401 / 握手 1008 | Key 无效、被禁用或缺少 scope；重新创建 Key |
| 短票认证失败 | 短票 60 s 过期；确认多副本共享 `NV_TICKET_SECRET` |
| CORS 错误 | 在 `server.cors_origins` 加入页面来源 |
| WebSocket 立即断开 | 检查反向代理 Upgrade 头、空闲超时、`max_message_bytes` |
| TTS 返回 `InvalidParameter` | 音色名不存在、采样率不在支持列表、`rate/pitch != 1.0` |
| TTS 返回 `UnsupportedFeature` | 使用了 SSML、非 PlainText 或不支持的格式 |

## 模型与性能

| 现象 | 处理 |
|---|---|
| TTS 首块慢 | 确认 `use_cuda_graph=true`、`warmup=true`；8 GB 下 vLLM 版首音不一定更快 |
| TTS RTF > 1 | Qwen3-TTS transformers 版正常在 1.6–1.7；需要低延迟请用 CosyVoice3 或 vLLM |
| ASR partial 很慢 | GPU 被 TTS 占满；降低并发、把 LLM 放 CPU，或使用兜底打断参数 |
| 克隆音色不收敛 / 怪声 | 参考音频需干净、单人、3–10 s；参考文本必须与音频一致 |
| 克隆后重启丢失 | 网关与 TTS engine 是否共享 `.data/voices`？Docker 是否挂载同一 volume？ |
| 显存不足 | 只运行一个 TTS 模型；LLM 用 CPU；不要同时跑两个 vLLM 容器 |

## vLLM-Omni

| 现象 | 处理 |
|---|---|
| 容器启动失败 / z3 ptxas 错误 | 使用 `deploy/vllm-omni/Dockerfile` 构建，不要直接跑 nightly |
| 首次合成 18 s | code2wav/CUDA 懒初始化；保持 `warmup: true`，等 `/readyz` 变绿 |
| 空闲后下一句变慢 | vLLM 30 s 空闲超时；provider 默认 20 s 回收，属预期 |
| 参考音频被拒绝 | 必须位于 `--allowed-local-media-path /workspace` 挂载内，用 `file:///workspace/...` |
| 两个模型都想跑 | 8 GB 显存不允许；一次只保留一个 TTS 模型 |

## 单机 CLI

| 现象 | 处理 |
|---|---|
| `sounddevice` 导入失败 | `pip install sounddevice` |
| 列不出设备 | `python desktop/voice_call.py --list-devices`；检查系统音频驱动 |
| 模型路径错误 | 在仓库根目录运行；`desktop/` 依赖 `models/`、`bin/`、`third_party/` |
| 文字模式验证 | `python desktop/voice_call.py --text --no-play --verbose` |
| 打断测试 | `python tests/manual/check_barge.py` |

## 收集诊断信息

提交 Issue 时请附上：

1. 操作系统、GPU、显存、驱动版本；
2. Python / torch / transformers 版本；
3. `configs/service.yaml` 中与问题相关的片段（隐去密钥）；
4. 启动命令与完整错误日志；
5. `GET /readyz` 和 `GET /v1/metrics/summary` 输出；
6. 如果是音频问题，附上 `--save-audio` 生成的 wav（注意隐私）。

更多已知限制见 [deployment.md](deployment.md#9-已知限制--路线图)。
