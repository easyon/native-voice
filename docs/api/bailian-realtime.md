# 百炼兼容 ASR / TTS WebSocket

端点：

- `WS /api-ws/v1/inference`：根据 `payload.task` 自动分发。
- `WS /ws/asr`：强制 ASR。
- `WS /ws/tts`：强制 TTS。

认证：握手头 `Authorization: Bearer <api_key>`。无效 Key 返回 close code `1008`。

所有控制帧都是 JSON：

```json
{
  "header": {"action": "run-task", "task_id": "...", "streaming": "duplex"},
  "payload": {}
}
```

`header.action` 取值：`run-task`、`continue-task`、`finish-task`。
二进制帧只用于 ASR 输入音频和 TTS 输出音频。

## ASR

### 客户端流程

```text
run-task(task=asr) ──► PCM16 帧 ... ──► finish-task
```

`run-task` 示例：

```json
{
  "header": {"action": "run-task", "task_id": "task-1", "streaming": "duplex"},
  "payload": {
    "task_group": "audio",
    "task": "asr",
    "function": "Recognition",
    "model": "local",
    "parameters": {
      "format": "pcm",
      "sample_rate": 16000,
      "max_sentence_silence": 1300,
      "speech_noise_threshold": 0.0
    }
  }
}
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `format` | `pcm` | `pcm` 或 `wav`；其他格式不支持 |
| `sample_rate` | `16000` | 输入采样率；8k–48k 自动重采样到 16k |
| `max_sentence_silence` | 配置值（1300） | 句尾静音分句阈值，毫秒 |
| `speech_noise_threshold` | `0.0` | 分句器的噪声阈值 |

随后发送 **PCM16 LE 单声道** 二进制帧。`format=wav` 时第一帧可带 44 字节
RIFF 头，服务端会自动跳过。

### 服务端事件

`task-started`：

```json
{"header": {"task_id": "task-1", "event": "task-started", "attributes": {}}, "payload": {}}
```

`result-generated`（partial 与 final 共用）：

```json
{
  "header": {"task_id": "task-1", "event": "result-generated", "attributes": {}},
  "payload": {
    "output": {
      "sentence": {
        "begin_time": 120,
        "end_time": 1840,
        "text": "你好，我想问一下天气",
        "heartbeat": false,
        "sentence_begin": false,
        "sentence_end": true,
        "sentence_id": 1,
        "words": []
      }
    },
    "usage": {"duration": 2}
  }
}
```

- partial：`sentence_begin=false`、`sentence_end=false`，`end_time` 可能为 `null`。
- 句首：`sentence_begin=true`，文本为空。
- final：`sentence_end=true`，`text` 为定稿。
- `words` 当前为空；时间戳为近似值。
- 文本在 partial 之间可能修订，客户端应以 `sentence_id` 覆盖而不是追加。

`task-finished`：

```json
{
  "header": {"task_id": "task-1", "event": "task-finished", "attributes": {"request_uuid": "..."}},
  "payload": {"usage": {"duration": 12}}
}
```

发送 `finish-task` 后，服务端会 flush 当前句并返回最终的 `result-generated`
（如果有内容），然后返回 `task-finished`。

### 限制

- 同一连接同一时刻只能运行一个任务。
- 单帧超过 `server.max_message_bytes`（默认 64 KB）会返回 `InvalidParameter`。
- 分句使用能量阈值，不是语义切分；可通过 `max_sentence_silence` 调整。

## TTS

### 客户端流程

```text
run-task(task=tts) ──► continue-task(文本) ... ──► finish-task
```

`run-task` 示例：

```json
{
  "header": {"action": "run-task", "task_id": "tts-1", "streaming": "duplex"},
  "payload": {
    "task_group": "audio",
    "task": "tts",
    "function": "SpeechSynthesizer",
    "model": "local",
    "input": {},
    "parameters": {
      "text_type": "PlainText",
      "voice": "default",
      "format": "pcm",
      "sample_rate": 24000,
      "volume": 50
    }
  }
}
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `text_type` | `PlainText` | 仅支持 `PlainText` |
| `voice` | `default` | 内置音色或 `clone_*`；可用 `tts.voice_aliases` 重定向 |
| `format` | `pcm` | `pcm`、`wav`、`mp3`、`opus` |
| `sample_rate` | `24000` | 8000/16000/22050/24000/44100/48000 |
| `volume` | `50` | 0–100 映射到 0–2 增益 |
| `rate` | `1.0` | 必须为 1.0（不支持变速） |
| `pitch` | `1.0` | 必须为 1.0（不支持变调） |
| `enable_ssml` | `false` | 不支持 SSML |

`continue-task` 追加文本：

```json
{
  "header": {"action": "continue-task", "task_id": "tts-1"},
  "payload": {"input": {"text": "你好，欢迎使用本地语音服务。"}}
}
```

`finish-task` 结束并等待剩余句子：

```json
{"header": {"action": "finish-task", "task_id": "tts-1"}, "payload": {"input": {}}}
```

取消当前任务（已合成的部分保留，未合成的丢弃）：

```json
{"header": {"action": "finish-task", "task_id": "tts-1"}, "payload": {"input": {"directive": "cancel"}}}
```

### 服务端事件

每个分句：

1. `sentence-begin`
2. 若干 `sentence-synthesis` JSON + 紧跟的二进制音频帧
3. `sentence-end`

```json
{
  "header": {"task_id": "tts-1", "event": "result-generated", "attributes": {}},
  "payload": {
    "output": {
      "type": "sentence-begin",
      "sentence": {"index": 0, "words": []},
      "original_text": "你好，欢迎使用本地语音服务。"
    }
  }
}
```

最后返回 `task-finished`，`usage.characters` 为总字符数。

- `pcm` 是无容器头的 PCM16 LE 单声道，实时逐块下发，需要客户端知道采样率。
- `wav`/`mp3`/`opus` 按句编码：整句合成完成后一次性下发（`wav` 带 RIFF 头，
  `mp3`/`opus` 依赖 ffmpeg），延迟更高。
- 默认 `pcm@24k` 延迟最低。

### 限制

- 单个 `continue-task` 最多 20,000 字符，整个任务最多 200,000 字符。
- 相邻 `continue-task` 间隔超过 23 s 会报错。
- 同一连接只能运行一个任务；任务完成后可复用连接发起新任务。
- 未知音色返回 `InvalidParameter`。

## 示例

```bash
NV_KEY=nv_xxx python examples/bailian_ws_client.py \
  --url ws://127.0.0.1:8090/api-ws/v1/inference \
  --key "$NV_KEY" \
  tts --text "你好，这是一段测试语音。"

NV_KEY=nv_xxx python examples/bailian_ws_client.py \
  --url ws://127.0.0.1:8090/api-ws/v1/inference \
  --key "$NV_KEY" \
  asr --wav audio/asr_test.wav
```

并发压测：

```bash
python scripts/load_test.py --key "$NV_KEY" tts -n 4
python scripts/load_test.py --key "$NV_KEY" mixed -n 4 --wav audio/asr_test.wav
```
