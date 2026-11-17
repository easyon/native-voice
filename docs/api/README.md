# 接口总览

| 类型 | 路径 | 认证 | 文档 |
|---|---|---|---|
| 百炼兼容 ASR/TTS | `WS /api-ws/v1/inference` | `Authorization: Bearer <key>` | [bailian-realtime.md](bailian-realtime.md) |
| ASR 别名 | `WS /ws/asr` | 同上 | [bailian-realtime.md](bailian-realtime.md) |
| TTS 别名 | `WS /ws/tts` | 同上 | [bailian-realtime.md](bailian-realtime.md) |
| 全双工对话 | `WS /ws/conversation` | Bearer Key 或首帧短票 | [conversation.md](conversation.md) |
| 健康检查 | `GET /healthz`、`GET /readyz` | 无 | [rest.md](rest.md) |
| 指标 | `GET /metrics`、`GET /v1/metrics/summary` | 无（生产建议限制） | [rest.md](rest.md) |
| 短票 | `POST /v1/realtime/ticket` | Bearer Key | [rest.md](rest.md) |
| 音色 | `GET /v1/voices`、`POST /v1/voices/clone`、`DELETE /v1/voices/{id}` | Bearer Key（部分需 `tts` scope） | [rest.md](rest.md) |
| 网页 | `GET /ui/` | 页面内输入 Key | [../getting-started.md](../getting-started.md) |

## 认证

### Bearer Key

```http
Authorization: Bearer nv_xxxxxxxxxxxxxxxx
```

每个 Key 有 scopes：`asr`、`tts`、`conversation`。创建：

```bash
python -m native_voice.cli.main key create --name my-app --scopes asr,tts,conversation
```

无效、禁用或过期 Key 在 WebSocket 握手阶段以 close code `1008` 拒绝。

### 对话短票

浏览器不应把长期 Key 放进 URL。推荐流程：

1. `POST /v1/realtime/ticket`（Bearer Key）→ `{"ticket": "...", "expires_in": 60}`
2. 连接 `/ws/conversation`，首帧发送 `{"type":"auth","ticket":"..."}`。

短票由 HMAC 签名，默认 60 s 有效；多副本部署时必须共享 `NV_TICKET_SECRET`。

## 错误格式

百炼兼容端点通过 `task-failed` 返回：

```json
{
  "header": {
    "task_id": "...",
    "event": "task-failed",
    "error_code": "InvalidParameter",
    "error_message": "..."
  },
  "payload": {}
}
```

对话端点通过 `error` 事件返回：

```json
{"type": "error", "code": "Throttling", "message": "..."}
```

常见错误码：

| code | 含义 |
|---|---|
| `InvalidParameter` | 参数缺失、格式错误、未知音色 |
| `UnsupportedFeature` | SSML、变速/变调、不支持的音频格式等 |
| `Forbidden` | Key 缺少对应 scope |
| `Throttling` | 并发、RPM 或每日额度超限 |
| `TaskAlreadyRunning` | 同一连接已有任务 |
| `InternalError` | 模型推理失败 |

## 示例客户端

- `examples/bailian_ws_client.py`：百炼兼容 ASR/TTS。
- `examples/conversation_client.py`：全双工对话。
- `scripts/load_test.py`：并发压测。

运行示例前先设置 `NV_KEY`：

```bash
NV_KEY=nv_xxx python examples/conversation_client.py \
  --key "$NV_KEY" --http http://127.0.0.1:8090 --ws ws://127.0.0.1:8090
```
