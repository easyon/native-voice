# Native Voice 文档

本目录是项目的完整文档。建议按以下顺序阅读。

## 入门

| 文档 | 适合谁 | 内容 |
|---|---|---|
| [getting-started.md](getting-started.md) | 第一次使用 | 环境要求、安装依赖、下载模型、启动服务 / 单机 CLI |
| [configuration.md](configuration.md) | 部署与调参 | `service.yaml` 全字段、环境变量、provider 选择 |
| [troubleshooting.md](troubleshooting.md) | 遇到问题 | 常见错误、音频设备、显存、延迟排查 |

## 理解系统

| 文档 | 内容 |
|---|---|
| [architecture.md](architecture.md) | 组件职责、数据流、全双工与插话状态机、代码分层 |
| [models.md](models.md) | ASR / TTS 提供方对比、显存占用、实测性能、音色克隆 |

## 接口

| 文档 | 内容 |
|---|---|
| [api/README.md](api/README.md) | 接口总览、认证、错误码 |
| [api/bailian-realtime.md](api/bailian-realtime.md) | 百炼兼容 ASR / TTS WebSocket 协议 |
| [api/conversation.md](api/conversation.md) | 全双工对话 WebSocket 事件 |
| [api/rest.md](api/rest.md) | REST、健康检查、指标、音色管理 |

## 运维与开发

| 文档 | 内容 |
|---|---|
| [deployment.md](deployment.md) | 生产部署、外部 engine、Docker、TLS、容量规划 |
| [development.md](development.md) | 本地开发、测试、代码规范、手工检查与基准脚本 |

## 代码内文档

- 主包入口：`src/native_voice/`
- 单机 CLI：`desktop/README.md`
- 网页客户端：`web/README.md`
- vLLM-Omni 部署：`deploy/vllm-omni/README.md`
- 脚本说明：`scripts/README.md`
- 基准脚本：`benchmarks/README.md`
- 手工检查：`tests/manual/README.md`
