# REST 接口

基础地址：`http://127.0.0.1:8090`

## 健康检查

### `GET /healthz`

进程存活即返回 200：

```json
{"status": "ok"}
```

### `GET /readyz`

模型/后端就绪返回 200，否则 503：

```json
{"status": "ready", "tts": true, "asr": true}
```

用于容器 healthcheck、负载均衡摘流。`spawn` 模式下网关会等待 engine 和
llama-server 就绪后再返回 ready。

## 指标

### `GET /metrics`

Prometheus 文本格式。主要指标：

| 指标 | 含义 |
|---|---|
| `nv_ws_connections` | 当前 WebSocket 连接数（按端点） |
| `nv_tasks_total` | 任务数（按类型/结果） |
| `nv_task_duration_seconds` | 任务耗时（按类型） |
| `nv_errors_total` | 错误数（按错误码） |
| `nv_asr_partial_seconds` | ASR partial 间隔 |
| `nv_tts_first_chunk_seconds` | TTS 首块延迟 |
| `nv_tts_rtf` | TTS 实时率 |
| `nv_audio_seconds_total` | 处理音频秒数 |
| `nv_tts_characters_total` | 合成字符数 |
| `nv_quota_rejections_total` | 准入拒绝数（按原因） |
| `nv_ready` | 就绪状态（1=ready） |

指标保持低基数，不包含用户文本、Key 或 session id。

### `GET /v1/metrics/summary`

```json
{
  "ready": true,
  "quota": {"active_total": 1, "active_keys": 1}
}
```

适合网页指标面板。

## 短票

### `POST /v1/realtime/ticket`

请求头：`Authorization: Bearer <key>`（需要 `conversation` scope）。

```json
{"ticket": "eyJ...", "expires_in": 60}
```

短票用于浏览器连接 `/ws/conversation`，避免把长期 Key 放进 URL 或前端日志。

```bash
curl -X POST http://127.0.0.1:8090/v1/realtime/ticket \
  -H "Authorization: Bearer $NV_KEY"
```

## 音色

### `GET /v1/voices`

无需认证：

```json
{
  "voices": [
    {"name": "default", "description": "本地默认中文女声"},
    {"name": "clone_870b1491460b", "description": "克隆 · 我的音色"}
  ]
}
```

具体字段由 TTS backend 返回，`name` 是合成时使用的 `voice` 参数。

### `POST /v1/voices/clone`

`multipart/form-data`，需要 Bearer Key + `tts` scope。

| 字段 | 必填 | 说明 |
|---|---|---|
| `file` | ✅ | 参考音频，wav/flac/mp3/ogg，≤ 20 MB |
| `name` | | 显示名称，最长 40 字符 |
| `text` | | 参考文本；留空时用本地流式 ASR 自动转写 |
| `consent` | ✅ | 必须为 `true`，确认拥有声音授权 |

服务端处理：

- 解码、转单声道、重采样到 24 kHz；
- 裁剪首尾 50 ms 近静音并归一化；
- 校验时长在 `voice_store.min_seconds`–`max_seconds`（1.5–30 s）；
- 参考文本前加 `You are a helpful assistant.<|endofprompt|>`；
- 保存到 `.data/voices/` 并注册到 TTS engine。

响应：

```json
{
  "voice": {"name": "clone_xxx", "display_name": "我的音色", "description": "克隆 · 我的音色"},
  "transcript": "希望你以后能够做的比我还好呦",
  "duration": 3.5
}
```

```bash
curl -X POST http://127.0.0.1:8090/v1/voices/clone \
  -H "Authorization: Bearer $NV_KEY" \
  -F file=@reference.wav \
  -F name=myvoice \
  -F consent=true
```

错误：

| 状态码 | 原因 |
|---|---|
| 400 | 未确认授权、音频无法解码/静音、时长不符、无法识别文本 |
| 403 | Key 缺少 `tts` scope |
| 413 | 文件超过上限 |
| 501 | 当前 TTS backend 不支持克隆 |
| 503 | TTS engine 注册失败 |

### `DELETE /v1/voices/{voice_id}`

需要 `tts` scope。删除克隆音色（内置音色不可删除）：

```json
{"ok": true, "voice": "clone_xxx"}
```

## 网页

### `GET /ui/`

托管 `web/` 静态文件（ES modules + AudioWorklet），无需构建。

`GET /` 会重定向到 `/ui/`。

## CORS 与反向代理

- 跨域浏览器访问需在 `server.cors_origins` 中配置来源。
- 生产环境建议放在 Nginx/Caddy 后启用 TLS/WSS，并限制 `/metrics` 的访问来源。
- 反向代理需允许 WebSocket Upgrade，并适当调大帧/消息大小限制。
