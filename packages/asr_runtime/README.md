# funasr-runtime

FunASR 私有化部署运行时，对外提供与阿里云百炼 **Fun-ASR Realtime** 兼容的 WebSocket 接口，以及 OpenAI 兼容的离线转写接口。

## 能力

- 实时识别：`/api-ws/v1/inference`，`run-task` → 二进制 PCM/WAV 帧 → `finish-task`，服务端返回 `task-started` / `result-generated` / `task-finished` / `task-failed`。
- 离线转写：`POST /v1/audio/transcriptions`，支持 `json`、`text`、`verbose_json`、`srt`、`vtt`。
- 浏览器测试台：`GET /ui`，支持麦克风实时识别（16 kHz PCM）、WAV/音频文件 WS 回放、离线转写和事件日志；`server.ui_enabled=false` 可关闭。麦克风需要 HTTPS 或 `localhost`，局域网 HTTP 访问请用文件回放；浏览器 WebSocket 无法携带 `Authorization`，开启鉴权时实时模式会 4401，离线转写仍会带 Bearer key。
- 模型后端：
  - `funasr`：进程内 PyTorch `AutoModel`，支持 Fun-ASR-Nano、Paraformer、SenseVoice，fsmn-vad 流式断句；实时 partial 采用“VAD 分段 + 有界窗口重解码”。
  - `llamacpp`：调用 `llama-funasr-*` GGUF 二进制，面向 CPU/边缘与离线长音频；实时仅句末结果，partial 默认关闭。
  - `mock`：确定性引擎，不需要 torch/模型，用于联调与 CI。

## 安装

```bash
pip install -e packages/common
pip install -e "packages/asr_runtime[dev]"          # mock / 网关
pip install -e "packages/asr_runtime[funasr]"       # PyTorch 实时引擎
```

## 快速开始（mock）

```bash
funasr-runtime -c configs/funasr.yaml
# 浏览器测试台（麦克风 / WAV 回放 / 离线转写）
open http://localhost:8001/ui
curl http://localhost:8001/readyz
curl -F file=@sample.wav -F model=fun-asr-realtime \
  http://localhost:8001/v1/audio/transcriptions
```

WebSocket 客户端：

```bash
python scripts/asr_ws_smoke.py ws://127.0.0.1:8001 fun-asr-realtime sample.wav
```

## 真实模型

模型默认从本地目录加载，不联网：

```bash
python scripts/download_funasr_models.py --models fun-asr-nano,fsmn-vad --dest ./models
FUNASR_RUNTIME_CONFIG=configs/funasr.gpu.yaml funasr-runtime
```

纯 CPU 机器用 SenseVoiceSmall + fsmn-vad：

```bash
python scripts/download_funasr_models.py --models sensevoice,fsmn-vad --dest ./models
FUNASR_RUNTIME_CONFIG=configs/funasr.cpu.yaml funasr-runtime
```

Paraformer 原始输出不带标点，需要额外挂载 `ct-punc`：

```bash
python scripts/download_funasr_models.py --models paraformer,fsmn-vad,punc --dest ./models
# configs/funasr.cpu.yaml 中设置 punc_artifact: ./models/punc
```

`configs/funasr.gpu.yaml` 使用 `/models` 下的 Fun-ASR-Nano + fsmn-vad；`configs/funasr.cpu.yaml` 使用 SenseVoiceSmall + fsmn-vad 的纯 CPU 组合；`configs/funasr.llamacpp.yaml` 使用 GGUF 二进制与文件路径。改 `models.<key>.artifact` / `vad_artifact` 即可切换。

## 协议要点

- 音频：16 kHz、单声道、16-bit PCM。实时 v1 只接受 `pcm` / `wav`；mp3/opus 等压缩流式格式返回 `UnsupportedParameter`，离线接口可通过 ffmpeg 解码。
- `sentence_id` 从 1 递增；`sentence_begin=true` 出现在一句的首个结果；`sentence_end=true` 时 `usage.duration` 为累计音频秒数。
- 成功任务后连接可复用；`task-failed` 后连接关闭（与百炼文档一致）。
- `heartbeat=true` 时无结果期间返回 `heartbeat=true`、`sentence_id=0` 的心跳事件。
- 认证：`Authorization: Bearer <api_key>`；生产通过 `FUNASR_API_KEYS` 或 `auth.api_keys` 开启。

## 环境变量

| 变量 | 说明 |
|---|---|
| `FUNASR_RUNTIME_CONFIG` | 配置文件路径，默认 `configs/funasr.yaml` |
| `FUNASR_ENGINE_TYPE` | 覆盖 `engine.type` |
| `FUNASR_API_KEYS` | 逗号分隔的 API Key；设置后自动开启鉴权 |
| `FUNASR_MODELS_DIR` | 覆盖 `engine.funasr.model_root` |
| `FUNASR_DEVICE` | 覆盖 `engine.funasr.device` |
| `FUNASR_PARTIAL_INTERVAL_MS` | 覆盖 partial 间隔 |
| `FUNASR_PARTIAL_WINDOW_S` | 覆盖 partial 解码窗口 |
| `FUNASR_LOG_LEVEL` / `FUNASR_LOG_FORMAT` | 日志级别/格式 |

## 边界

- Fun-ASR-Nano 没有逐 chunk cache；partial 是重识别。句子超过 `partial_window_s` 后不再发 partial，句末仍用完整段返回 final。
- 模型不提供字级时间戳时返回 `words=[]`，不伪造。
- 首版不含说话人分离、预编译热词 ID、多路混音、Helm chart。
