# 单机全双工 CLI

`desktop/` 是不经过网关的轻量全双工实现：直接打开麦克风和扬声器，
由 `nv` 包完成 VAD、ASR、LLM、TTS 和会话状态管理。

它适合：

- 快速验证模型链路和声卡；
- 离线单机使用；
- 复现打断、端点、TTS 延迟问题。

生产部署请使用服务模式（见 [../docs/getting-started.md](../docs/getting-started.md)）。

## 运行

```powershell
conda activate cosyvoice
cd <repo>
python desktop\voice_call.py
```

或双击 `run_voice_call.bat`（如 Python 路径不同，请修改脚本中的 `PY`）。

前置条件：

- conda 环境 `cosyvoice`（Python 3.10，torch 2.3.1+cu121，`x-transformers==2.11.24`）
- `models/Fun-CosyVoice3-0.5B`、`models/Qwen3.5-2B-Q8_0.gguf`、`bin/llama/llama-server.exe`
- SenseVoiceSmall / 流式 Paraformer 缓存
- `sounddevice` 已安装

首次启动约 20 秒加载模型。看到 `[状态] 聆听中` 后直接说话。

**强烈建议戴耳机。** 不戴耳机时扬声器声音会被麦克风拾取，可能误触发插话；
可以用 `--barge-db` 调高打断阈值。

## 键盘

| 操作 | 作用 |
|---|---|
| 输入文字 + 回车 | 发送一条文字消息（AI 说话时会先打断） |
| 直接回车 | 打断当前回答 |
| `Ctrl+C` / `/quit` | 退出 |

## 参数

```powershell
python desktop\voice_call.py --list-devices      # 查看声卡设备编号
python desktop\voice_call.py --mic 1 --speaker 3 # 指定设备
python desktop\voice_call.py --no-play           # 只跑 TTS 不出声（调试）
python desktop\voice_call.py --text              # 纯文字模式，不开麦克风/扬声器
python desktop\voice_call.py --wav audio\asr_test.wav --save-audio
                                                 # 用 wav 模拟麦克风
python desktop\voice_call.py --asr streaming     # 默认：只用流式 Paraformer
python desktop\voice_call.py --asr hybrid        # 流式部分结果 + SenseVoice 定稿
python desktop\voice_call.py --asr sensevoice    # 只用 SenseVoice 整句
python desktop\voice_call.py --asr-chunk-ms 600  # 部分结果刷新粒度（默认 600ms）
python desktop\voice_call.py --no-partials       # 隐藏实时字幕
python desktop\voice_call.py --barge-db -24      # 打断阈值（dBFS，默认 -30）
python desktop\voice_call.py --endpoint-ms 650   # 句尾静音判定（默认 500ms，650 更稳）
python desktop\voice_call.py --llm models\...gguf --port 8081
python desktop\voice_call.py --no-graph          # 关闭 TTS CUDA graph（排障）
python desktop\voice_call.py --verbose           # 打印识别耗时等诊断信息
```

`--save-audio` 会把用户语音和 AI 回复写到 `audio/calls/`。

## 全双工行为

- 麦克风全程采集，VAD 用自适应噪声底 + 迟滞阈值断句。
- 说话过程中每 600 ms 输出一次 `[识别] ...` 部分结果。
  `hybrid` 模式句尾再用 SenseVoice 校正标点/数字；`streaming` 直接使用流式结果。
- VAD 判定说完时 `finalize()` 会立即 flush 尾音，收尾约 30–80 ms。
- 助手说话时检测到持续 160 ms 的语音 → 清空播放队列并取消本轮 LLM/TTS；
  打断前 450 ms 音频作为 pre-roll 保留。
- 用户说完后、AI 出声前（1.6 s 内）又接着说，两段识别合并为同一轮。
- 播放开始有 350 ms 回声探测窗；打断阈值取
  `max(噪声+12dB, --barge-db, 回声估计+6dB)`。
- TTS 取消时静默排空生成器，避免 CosyVoice 内部线程泄漏。

## 性能（RTX 4070 Laptop 8 GB 实测）

| 项目 | 数值 |
|---|---|
| TTS 首块延迟 | 0.55–0.85 s |
| TTS 持续合成 RTF | 0.38–0.52 |
| ASR 实时部分结果 | 每 600 ms，流式 RTF ~0.1 |
| ASR 句尾收尾 | 0.03–0.15 s |
| LLM 首 token | 预热后 ~0.16 s |
| TTS 显存 | ~1.8 GB |
| 回答首字到出声 | ~2.1 s |

## 排障

| 现象 | 处理 |
|---|---|
| 说话没反应 | `--verbose`；`--mic N` 换设备；检查系统麦克风音量 |
| 不戴耳机时自己打断自己 | 戴耳机，或 `--barge-db -20` |
| 一直不结束/结束太慢 | `--endpoint-ms 500` 或 `450` |
| llama-server 端口被占用 | `--port 8082` |
| 显存不够 | 关闭其他 GPU 程序；当前链路峰值约 3.7 GB |
| 首次回答很慢 | 正常：system prompt 首次 prefill |
| 把 AI 回声当用户 | 戴耳机；日志会显示“忽略回声” |
| 没有实时字幕 | 检查 `--no-partials` 或 `--asr sensevoice` |

## 代码结构

```text
desktop/
├── voice_call.py     # 入口 / 参数
└── nv/
    ├── config.py     # 配置
    ├── audio.py      # 麦克风、扬声器、wav 模拟输入
    ├── vad.py        # 能量 VAD、端点检测、打断、回声阈值
    ├── asr.py        # SenseVoiceSmall（整句定稿）
    ├── stream_asr.py # 流式 Paraformer
    ├── llm.py        # llama-server 启动 + SSE 流式客户端
    ├── tts.py        # CosyVoice3 加载与实时化
    └── conversation.py # 会话状态机、分句、打断、历史
```

`desktop/nv` 是自包含实现，与服务包 `src/native_voice/` 不共享代码。
