# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- 文档体系重构：入门、架构、配置、模型、API、部署、开发、排障分册。
- 英文 README（`README.en.md`）与第三方组件说明（`THIRD_PARTY_NOTICES.md`）。
- 标准开源项目文件：CONTRIBUTING、CODE_OF_CONDUCT、SECURITY、Issue/PR 模板、CI、
  pre-commit、Dependabot。
- `desktop/` 单机 CLI、`benchmarks/` 调优脚本、`tests/manual/` GPU 手工检查的独立目录。
- 包元数据：项目链接、license、classifiers、`py.typed`。
- 对话工具调用（function calling）：内置 `get_current_time`、`web_search`、
  `web_fetch`，支持多轮工具执行、执行前垫话、结果回填、轮数上限和
  `tool_choice` 配置。`web_fetch` 只允许公网 http/https 地址。
- 新事件 `assistant.tool.call` / `assistant.tool.result`，新指标
  `nv_tool_calls_total`、`nv_tool_duration_seconds`。
- llama-server 启动参数增加 `--jinja`，使用模型内嵌 chat template。

### Fixed

- `ThinkFilter` 的分隔标签字面量异常，导致 `</think>` 可能被当作正文送进 TTS；
  同时处理缺失开标签的推理残留。
- 小模型在工具结果后重复生成时提前结束本轮，避免 TTS 反复朗读同一句。
- 系统提示词明确声明“不知道当前时间”，修复询问“现在几点了”时模型直接编造时间
  而不调用工具的问题。
- 移除内置 `calculate` 工具（模型对简单运算可直接答对，收益有限）。

### Changed

- 仓库目录重新划分：`app/` 拆分为 `desktop/`、`benchmarks/`、`tests/manual/`。
- README 重写为项目入口，详细内容下沉到 `docs/`。
- 默认 LLM 从 Qwen3-1.7B 切换为 Qwen3.5-2B（Q8_0 GGUF，约 2.0 GB），
  工具调用在默认系统提示词下可稳定触发；`fetch_deps.py` 改为从
  `unsloth/Qwen3.5-2B-GGUF` 下载。
- 桌面 CLI 启动 llama-server 时同样加上 `--jinja`。

## [0.1.0] - 2026-11-16

### Added

- 本地全双工语音对话链路：VAD、流式 ASR、流式 LLM、流式 TTS、插话打断。
- 百炼兼容 ASR/TTS WebSocket 协议（`/api-ws/v1/inference`）。
- 全双工对话 WebSocket（`/ws/conversation`），支持实时字幕、音频下发、
  附和音轨、播放 ACK 和每轮时延指标。
- FastAPI 网关：API Key、scope、并发 / RPM / 每日额度、60 s 短票。
- 独立 ASR / TTS GPU engine 与网关调度，支持 `spawn` / `external` 模式。
- ASR provider：Paraformer（流式）、Qwen3-ASR。
- TTS provider：CosyVoice3、Qwen3-TTS、Qwen3-TTS vLLM、CosyVoice3 vLLM。
- 无构建网页客户端：麦克风采集、实时字幕、AI 音频、指标面板、音色克隆。
- zero-shot 音色克隆与 `.data/voices` 存储。
- vLLM-Omni 部署配置（Qwen3-TTS / CosyVoice3，8 GB 变体）。
- Prometheus 指标、结构化日志、健康检查。
- 协议客户端示例、压测脚本、模型下载与独立 venv 安装脚本。
