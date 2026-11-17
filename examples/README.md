# 客户端示例

| 文件 | 协议 |
|---|---|
| `bailian_ws_client.py` | 百炼兼容 ASR/TTS WebSocket |
| `conversation_client.py` | 全双工对话 WebSocket |

先启动服务并创建 Key：

```bash
scripts/serve_dev.ps1 -Backend fake
python -m native_voice.cli.main key create --name demo --scopes asr,tts,conversation
```

## 百炼兼容 TTS

```bash
python examples/bailian_ws_client.py --key "$NV_KEY" tts \
  --text "你好，这是一段测试语音。" --out tts_out.wav
```

## 百炼兼容 ASR

```bash
python examples/bailian_ws_client.py --key "$NV_KEY" asr \
  --wav audio/asr_test.wav
```

## 全双工对话

```bash
python examples/conversation_client.py --key "$NV_KEY" \
  --wav audio/asr_test.wav --out conversation_out.wav
```

`--http` / `--ws` 可指向远程地址；完整事件说明见
[`docs/api/`](../docs/api/README.md)。
