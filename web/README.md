# 网页客户端

无构建步骤的 ES modules + AudioWorklet 页面，由网关在 `/ui/` 静态托管。

## 文件

| 文件 | 作用 |
|---|---|
| `index.html` | 页面结构 |
| `styles.css` | 样式 |
| `pcm-worklet.js` | AudioWorklet：麦克风采样转 16 kHz PCM |
| `js/api.js` | REST / 短票请求 |
| `js/app.js` | 页面状态与事件绑定 |
| `js/audio-capture.js` | 麦克风采集 |
| `js/audio-playback.js` | PCM 播放调度与 ACK |
| `js/transcript.js` | 实时字幕 |
| `js/voice-orb.js` / `js/waveform.js` | 状态动画与波形 |
| `js/voice-recorder.js` | 音色克隆录音 |
| `js/metrics.js` | 时延指标面板 |
| `js/wav.js` | WAV 编码辅助 |

## 开发

```bash
scripts/serve_dev.ps1 -Backend fake   # 或 serve_dev.sh
```

打开 <http://127.0.0.1:8090/ui/>，输入开发 Key。`fake` 后端无需 GPU。

- API Key 只存 `sessionStorage`，通过短票连接对话 WebSocket。
- 麦克风使用 `getUserMedia({echoCancellation, noiseSuppression, autoGainControl})`。
- 播放使用 WebAudio 调度 24 kHz PCM，并发送 `playback.started/done`。
- 插话由服务端自动判断，页面不提供手动打断按钮。

协议细节见 [`docs/api/conversation.md`](../docs/api/conversation.md)。
