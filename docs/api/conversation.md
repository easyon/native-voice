# 全双工对话 WebSocket

端点：`WS /ws/conversation`

## 1. 连接与认证

两种方式：

**原生客户端（Bearer）**

```http
GET /ws/conversation
Authorization: Bearer nv_xxx
```

Key 需要 `conversation` scope。

**浏览器（短票）**

1. `POST /v1/realtime/ticket`（Bearer Key）获取 60 s 短票。
2. 建立连接后发送首帧：

```json
{"type": "auth", "ticket": "..."}
```

认证失败会以 close code `1008` 关闭。浏览器页面已封装该流程。

## 2. 音频格式

- 上行：二进制帧，**16 kHz / PCM16 LE / 单声道**，建议 20 ms（640 字节）一帧。
- 下行：服务端先发 `tts.audio.start` 声明采样率（默认 24 kHz），随后是
  PCM16 LE 单声道二进制帧。
- 单帧超过 `server.max_message_bytes` 会被拒绝。

## 3. 客户端事件

### `session.start`

```json
{"type": "session.start", "voice": "default", "endpoint_ms": 300, "system_prompt": "..."}
```

所有字段可选。`endpoint_ms` 可写 200/300/400/600 等值，运行中可通过
`session.config` 再改。

### `session.config`

```json
{"type": "session.config", "voice": "clone_xxx", "endpoint_ms": 400, "system_prompt": "..."}
```

### `input.text`

```json
{"type": "input.text", "text": "你好"}
```

把文字当作一次用户输入；如果助手正在说话，会先打断。

### `control.interrupt`

```json
{"type": "control.interrupt"}
```

手动打断当前回复。网页默认不提供按钮，插话由 VAD 自动触发，但客户端仍可使用。

### `playback.started` / `playback.done`

```json
{"type": "playback.started", "turn_id": "..."}
{"type": "playback.done", "turn_id": "..."}
```

客户端真正开始播放首帧时发送 `playback.started`，队列播完发送
`playback.done`。服务端据此：

- 计算真实听感时延（`metrics.playback`）；
- 启动回声抑制与打断判断；
- 在播放完成后清理本轮状态。

浏览器客户端会自动发送；原生客户端建议实现这两个 ACK。

### `session.end`

```json
{"type": "session.end"}
```

正常结束会话。

## 4. 服务端事件

| 事件 | 关键字段 | 说明 |
|---|---|---|
| `session.ready` | `session_id`, `voice`, `sample_rate`, `system_prompt` | 会话建立 |
| `asr.partial` | `text`, `sentence_id`, `begin_ms` | 实时部分结果 |
| `asr.final` | `text`, `sentence_id`, `begin_ms`, `end_ms` | 用户句定稿 |
| `assistant.text.delta` | `turn_id`, `text` | LLM 文本增量 |
| `assistant.text.done` | `turn_id`, `text` | 本轮文本完成 |
| `tts.audio.start` | `turn_id`, `sentence_index`, `sample_rate`, `format` | 一句音频开始 |
| `tts.audio.done` | `turn_id`, `sentence_index` | 一句音频结束 |
| `assistant.audio.all_done` | `turn_id` | 本轮所有音频下发完毕 |
| `assistant.backchannel.start` | `backchannel_id`, `sample_rate`, `format`, `volume` | 附和音频开始 |
| `assistant.backchannel.done` | `backchannel_id` | 附和音频结束 |
| `assistant.interrupted` | `turn_id`, `reason` | 本轮被用户打断 |
| `assistant.cancelled` | `turn_id`, `reason` | 本轮被取消 |
| `metrics.turn` | 见下 | 每轮服务端时延 |
| `metrics.playback` | 见下 | 客户端播放 ACK 后时延 |
| `error` | `code`, `message` | 错误 |
| `session.closed` | `reason` | 会话关闭 |

### 音频下发

每一句：

```text
tts.audio.start
  ├─ 二进制 PCM16 帧
  ├─ 二进制 PCM16 帧
  └─ ...
tts.audio.done
```

全部句子结束后：

```text
assistant.audio.all_done
```

同一轮的多句 `sentence_index` 从 0 递增。

### 附和音轨

```text
assistant.backchannel.start {backchannel_id, sample_rate, format, volume}
  └─ 一个二进制 PCM16 帧
assistant.backchannel.done {backchannel_id}
```

附和是**独立低音量音轨**：不要把它拼进本轮 assistant 音频，也不要计入
`metrics.turn`。它不进入 LLM 历史，不产生用户轮次。

### 时延指标

`metrics.turn`：

```json
{
  "type": "metrics.turn",
  "turn_id": "...",
  "endpoint_ms": 300,
  "asr_ms": 140,
  "llm_first_ms": 160,
  "tts_first_ms": 550,
  "total_ms": 2100
}
```

| 字段 | 含义 |
|---|---|
| `endpoint_ms` | 用户说完到 VAD 判定端点的等待 |
| `asr_ms` | 端点判定到 ASR final |
| `llm_first_ms` | 文本提交到 LLM 首 token |
| `tts_first_ms` | LLM 首 token 到 TTS 首块 |
| `total_ms` | 用户说完（最后一个有声帧）到 TTS 首块的端到端时间 |

`metrics.playback`（收到 `playback.started` 后发送）：

```json
{"type": "metrics.playback", "turn_id": "...", "playback_ms": 240, "audible_total_ms": 2340}
```

| 字段 | 含义 |
|---|---|
| `playback_ms` | TTS 首块到客户端真正开始播放 |
| `audible_total_ms` | 用户说完到真正出声 |

## 5. 插话与合并语义

- **自动打断**：助手思考/说话时，用户持续说话 `barge_sustain_ms`（400 ms）
  后，服务端等待 ASR partial 出现有意义内容再取消当前回复。
  “嗯/啊/对/好的”等附和不会打断。
- **兜底打断**：用户持续说话超过 `barge_content_wait_ms`（900 ms）仍没有
  可用 partial 时也会打断（GPU 与 TTS 争用时 ASR 会变慢）。
- **碎片合并**：用户说完后、AI 出声前（`merge_window_s`，默认 1.6 s）又接着
  说，两段识别会合并为同一轮。
- **上下文连续**：被打断时，已经说出的部分回复会写入历史（被合并的碎片不写），
  下一轮 LLM 能看到“上一轮说到哪”。
- **回声抑制**：播放开始后有 350 ms 探测窗；客户端发送 `playback.started`
  有助于服务端估计回声电平。

收到 `assistant.interrupted` 后，客户端应立即停止播放当前轮音频。

## 6. 网页客户端

`web/` 是参考实现，展示：

- 短票认证与自动重连；
- AudioWorklet 采集并转换为 16 kHz PCM；
- 实时字幕、AI 文本流式显示；
- PCM 播放调度与 `playback.started/done` ACK；
- 指标面板、音色选择与克隆。

可直接参考 `web/js/` 实现自己的客户端。
