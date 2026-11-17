# 部署指南

## 1. 部署拓扑

推荐的生产结构：

```text
Nginx / Caddy（TLS、WSS、限流）
        │
        ▼
Gateway × N（CPU，无状态）
        │ 内部网络
        ├── ASR engine（GPU）
        ├── TTS engine（GPU）
        └── llama-server（CPU/GPU）
```

- Gateway 不加载 torch，可以多副本水平扩展。
- ASR / TTS engine 各自单进程、单模型；GPU 资源按 engine 分配。
- llama-server 可以独立部署，也可以由 Gateway 启动。

## 2. 生产配置

`configs/service.yaml` 关键改动：

```yaml
server:
  host: 0.0.0.0
  port: 8090
  cors_origins: ["https://your-domain.example"]

auth:
  db_path: /var/lib/native-voice/native_voice.db
  ticket_secret: "从密钥管理系统注入"
  dev_key: false

engine:
  mode: external          # engine 由 systemd / Docker 管理
  asr_ws: ws://asr-engine:8101/asr
  tts_ws: ws://tts-engine:8102/tts
  token: "内部 engine 共享 token"

llm:
  mode: external
  # 或保持 spawn 并指向生产路径

logging:
  json_logs: true
```

多副本时必须：

1. 所有 Gateway 使用相同的 `NV_TICKET_SECRET`；
2. 共享同一个 Key 数据库（当前 SQLite 仅适合单节点）；
3. 共享 `voice_store.dir`（克隆音色需要被 TTS engine 读取）。

## 3. 启动顺序

```bash
# 1. engine
python -m native_voice.engine.cli asr --config configs/service.yaml --port 8101
python -m native_voice.engine.cli tts --config configs/service.yaml --port 8102

# 2. gateway
python -m native_voice.cli.main serve --backend worker --config configs/service.yaml
```

生产建议用 systemd 或容器编排管理，并设置自动重启：

```ini
# /etc/systemd/system/native-voice-asr.service
[Unit]
Description=Native Voice ASR engine
After=network.target

[Service]
WorkingDirectory=/opt/native-voice
Environment=PYTHONPATH=/opt/native-voice/src
ExecStart=/opt/native-voice/.venv/bin/python -m native_voice.engine.cli asr --config configs/service.yaml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

TTS engine 同理，改 `kind=tts` 与端口即可。

## 4. TLS 与反向代理

必须放在 TLS 代理后。Nginx 示例：

```nginx
server {
    listen 443 ssl http2;
    server_name voice.example.com;

    ssl_certificate     /etc/ssl/voice.crt;
    ssl_certificate_key /etc/ssl/voice.key;

    location / {
        proxy_pass http://127.0.0.1:8090;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 3600s;
    }

    # 指标只允许内网访问
    location = /metrics {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://127.0.0.1:8090;
    }
}
```

注意：

- WebSocket 需要 `Upgrade` / `Connection` 头。
- 调大 `proxy_read_timeout`，避免长连接被断开。
- 限制请求体和帧大小，配合 `server.max_message_bytes`。
- `/metrics` 不要直接暴露公网。

## 5. 容量与扩容

| 场景 | 建议 |
|---|---|
| 8 GB 单卡 | 承诺 2 路对话；4 路可用但 TTS 排队 |
| 更大显存 | 增加 TTS engine 或提高单引擎并发 |
| 多副本 Gateway | 共享 Key DB + ticket secret + voice store |
| 高并发 ASR | 多 ASR engine，按 Key 或轮询分配 |

瓶颈通常在 TTS：单 engine 的句子合成是串行的。ASR 首结果很快，但 GPU 被
TTS 占满时 partial 会变慢，服务端有 `barge_content_wait_ms` 兜底。

压测：

```bash
python scripts/load_test.py --key "$NV_KEY" tts -n 4
python scripts/load_test.py --key "$NV_KEY" mixed -n 4 --wav audio/asr_test.wav
```

## 6. vLLM-Omni 真流式 TTS

`qwen3tts_vllm` / `cosyvoice3_vllm` 需要外部 vLLM-Omni 容器：

```bash
cd deploy/vllm-omni
docker compose build
docker compose up -d
curl http://127.0.0.1:8091/health
```

CosyVoice3 使用 profile：

```bash
docker compose --profile cosyvoice3 up -d tts-cosyvoice3
curl http://127.0.0.1:8092/health
```

然后 Gateway 使用 `-Tts qwen3tts_vllm` 或 `-Tts cosyvoice3_vllm`。
容器不由 engine supervisor 管理，需独立保活。详细说明见
[`deploy/vllm-omni/README.md`](../deploy/vllm-omni/README.md)。

8 GB 显存不要同时运行两个 vLLM-Omni 容器，也不要在本地 CosyVoice3 之外
再启动 Qwen3-TTS。

## 7. 安全清单

- [ ] TLS/WSS 由反向代理终止，后端只监听内网。
- [ ] `auth.dev_key=false`，删除开发 Key。
- [ ] `NV_TICKET_SECRET` 使用强随机值并妥善保管。
- [ ] `engine.token` 限制内部 engine 的访问。
- [ ] `/metrics`、`/readyz` 限制来源。
- [ ] Key 数据库与 `.data/voices` 定期备份并限制权限。
- [ ] 声音克隆要求 `consent=true`，并保留授权记录。
- [ ] 日志默认不记录用户正文；生产开启 JSON 日志并集中收集。
- [ ] 限制 `server.max_message_bytes`、上传大小和并发。
- [ ] 确认模型许可与商用条款。

## 8. 可观测性

- 日志：`logging.json_logs: true`，包含 session id、任务类型、错误码。
- 指标：Prometheus 抓取 `/metrics`；建议关注
  `nv_tts_first_chunk_seconds`、`nv_tts_rtf`、`nv_quota_rejections_total`、
  `nv_errors_total`、`nv_ready`。
- 健康检查：容器用 `/healthz` 判存活，`/readyz` 判就绪。
- 每轮时延：客户端可上报 `metrics.turn` / `metrics.playback`，网页已实现。

## 9. 已知限制 / 路线图

- 单 TTS engine 串行；4 路并发时延较高。
- 非人声过滤是概率性的：很长的噪声/音乐仍可能被当成语音。
- 声音克隆为 zero-shot，不是微调；不支持多语种 ASR、热词、SSML、变调/变速。
- 对话会话状态驻留 Gateway 内存，断线即回收，无跨重连恢复。
- SQLite 配额与 Key 仅适合单节点；生产多副本建议换 Postgres/Redis。
- 网页未跑 Playwright 自动化；JS 已做语法检查，API 已用 Python 客户端 E2E 验证。
- 待办：Docker Compose 一键部署、Nginx TLS 模板、审计与配额持久化、
  Grafana 面板、压测/故障演练、模型商用许可确认。

## 10. 升级与回滚

```bash
git pull
uv sync --extra gpu
# 重启 engine 与 gateway
```

- 模型权重与二进制位于 `models/`、`bin/`、`third_party/`，升级代码不会覆盖。
- 配置变更前先备份 `configs/`、`.data/`。
- 大版本升级后先运行 `pytest -q` 和 `scripts/load_test.py`，再切流量。
