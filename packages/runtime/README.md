# cosyvoice-runtime

阿里云百炼兼容的 CosyVoice 在线服务：HTTP 非流式、HTTP SSE、WebSocket 双工。

安装（先装 `cosyvoice-common`）：

```bash
pip install -e packages/common
pip install -e "packages/runtime[dev]"
```

运行：

```bash
cosyvoice-runtime
```

默认读取仓库根目录的 `configs/runtime.yaml`。本包只依赖
`cosyvoice-common` + FastAPI/uvicorn/pydantic/PyYAML；torch 仅在
`[cosyvoice]` extra 中安装，训练包不会被引入运行时镜像。

生产部署使用 Helm chart `deploy/helm/cosyvoice-runtime`，运维说明见
`docs/ops/cosyvoice-runtime-runbook.md`。生产基线包括：

- Bearer 鉴权、租户并发/RPM/字符速率限制、独立 metrics token；
- 请求体、WS 消息、WS 连接数、全局/租户并发上限；
- 优雅摘流与在途请求 drain；
- 首音频/单句/请求/任务时长超时；
- JSON 日志、request_id/tenant/task_id 上下文、Prometheus 直方图；
- 签名音频 URL、TTL 清理；
- 非 root 只读容器、K8s 探针、HPA/PDB/NetworkPolicy。

流式音频：`pcm` 原始 s16le；`wav` 首块带一次流式头、后续为 PCM；`mp3` 为独立帧；
`opus` 流式会被拒绝。高并发建议使用 PCM/MP3，并让网关与模型服务分离。

