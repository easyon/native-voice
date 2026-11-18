# 开发指南

## 1. 目录结构

```text
native-voice/
├── src/native_voice/         主包
│   ├── api/                  FastAPI 应用、路由、百炼兼容协议
│   │   ├── routers/          HTTP / WebSocket 路由
│   │   └── protocol/         协议编解码与任务状态机
│   ├── backends/             fake / local / worker 后端
│   ├── conversation/         全双工状态机、VAD、分句、打断
│   ├── engine/               独立 ASR / TTS GPU 引擎
│   ├── models/               ASR / TTS / LLM 适配器
│   ├── service/              进程监督、音色存储
│   ├── security/             Key、配额、短票
│   ├── audio/                PCM、重采样、编码
│   ├── observability/        日志与指标
│   ├── config/               配置模型与加载
│   ├── cli/                  命令行入口
│   └── utils/
├── desktop/                  单机全双工 CLI（可选，独立依赖）
├── web/                      无构建网页客户端
├── tests/
│   ├── unit/                 纯逻辑单元测试
│   ├── protocol/             FastAPI TestClient + fake 后端
│   └── manual/               GPU / 声卡手工检查（不参与 CI）
├── benchmarks/               历史调优脚本
├── configs/                  服务配置
├── deploy/vllm-omni/         vLLM-Omni 容器
├── docs/                     文档
├── examples/                 协议客户端示例
├── scripts/                  安装、启动、压测脚本
└── assets/                   运行时资源（Silero VAD）
```

不入库的运行时目录：`models/`、`bin/`、`third_party/`、`audio/`、`.data/`、
`envs/`。它们由 `scripts/fetch_deps.py` 或运行时生成。

## 2. 开发环境

```bash
uv sync --extra dev          # 基础依赖 + pytest/ruff/mypy
uv sync --extra gpu          # 需要真实模型时
```

只开发网页 / 协议时，不需要 GPU：

```powershell
scripts\serve_dev.ps1 -Backend fake
```

页面：<http://127.0.0.1:8090/ui/>。`fake` 后端会立即返回假识别和合成音频，
并驱动完整的对话状态机。

## 3. 测试

```bash
uv run pytest -q             # 单元 + 协议（fake 后端，无需 GPU）
uv run pytest tests/unit -q
uv run pytest tests/protocol -q
```

手工 GPU / 端到端检查（不参与 CI）：

```bash
python tests/manual/check_pipeline.py     # wav → VAD → ASR → LLM → TTS
python tests/manual/check_stream_asr.py   # 流式 Paraformer partial/final
python tests/manual/check_barge.py        # 文字模式打断
python tests/manual/check_tts.py          # CosyVoice3 直接合成
```

这些脚本依赖本地模型、`third_party/` 和 `audio/` 下的样例音频，详见
[`tests/manual/README.md`](../tests/manual/README.md)。

基准脚本位于 `benchmarks/`，记录调优时的对比实验，通常只在改模型代码时使用。

## 4. 代码规范

- 行宽 110，目标 Python 3.10。
- `ruff` 检查：

```bash
uv run ruff check src tests
uv run ruff check --fix src tests
```

- 类型提示尽量完整；公共接口使用 `from __future__ import annotations`。
- 日志使用标准 `logging`，不要在库代码里 `print`。
- 指标标签保持低基数，禁止把用户文本、Key、session id 放进标签。
- 协议字段兼容百炼；新增字段必须可选，避免破坏现有客户端。

## 5. 修改代码时从哪里开始

| 需求 | 位置 |
|---|---|
| 新增 ASR 模型 | `src/native_voice/models/asr/` + `backends/base.py` + `models/asr/factory.py` |
| 新增 TTS 模型 | `src/native_voice/models/tts/` + `backends/base.py` + `models/tts/factory.py` |
| 修改百炼协议 | `src/native_voice/api/protocol/` |
| 修改对话/打断逻辑 | `src/native_voice/conversation/` |
| 新增/修改工具调用 | `src/native_voice/conversation/tools.py` |
| 新增 REST 接口 | `src/native_voice/api/routers/` 并在 `api/app.py` 注册 |
| 修改认证/配额 | `src/native_voice/security/` |
| 修改网页 | `web/` |
| 修改配置字段 | `src/native_voice/config/schema.py` + `configs/service.yaml` + 本文档 |
| 新增 GPU 引擎 | `src/native_voice/engine/` |

### 新增 provider 的步骤

1. 在 `models/<kind>/` 实现适配器，暴露与现有 provider 一致的方法。
2. 在对应 `factory.py` 注册 name。
3. 在 `configs/service.yaml` 的 `providers` 下增加解释器、模型和 options。
4. 如需要独立 venv，在 `scripts/setup_qwen_engines.*` 或文档中说明。
5. 增加单元测试（可以用 fake/mock 避免加载大模型）。
6. 更新 `docs/models.md` 的 provider 表和限制。

### 新增内置工具

工具定义的唯一入口是 `src/native_voice/conversation/tools.py` 的
`BUILTIN_TOOLS`：

```python
def get_weather(city: str = "上海") -> str:
    return f"{city}今天多云，18 到 24 度。"

Tool(
    name="get_weather",
    description="查询城市天气。用户询问天气时调用。",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "城市名"}},
        "required": [],
    },
    handler=get_weather,
)
```

步骤：

1. 实现 handler：同步或 `async` 均可，接收 JSON 参数同名关键字参数；
   返回 `str`（或可 JSON 序列化的对象）。
2. 在 `BUILTIN_TOOLS` 中注册，并在 `configs/service.yaml` 的 `llm.tools` 里启用。
3. 保持轻量：handler 运行在网关事件循环里，单次默认 5 s 超时
   （`Tool.timeout_s` 可调）；慢操作请自行封装 HTTP 客户端和更短超时。
4. 加单元测试：正常返回、参数错误、异常、超时（参考 `tests/unit/test_tools.py`）。

handler 抛出的异常、非法 JSON、未知工具都不会中断会话：注册表会把错误文本
作为工具结果回填给模型，由模型决定怎么向用户解释。

## 6. 调试

- `--verbose`（单机 CLI）或 `NV_LOG_LEVEL=DEBUG` 打印更多日志。
- `GET /readyz` 检查 engine / LLM 是否就绪。
- `GET /v1/metrics/summary` 查看配额占用；`/metrics` 查看时延直方图。
- 协议联调用 `examples/` 下的客户端；抓包时注意 WebSocket 二进制帧。
- 单机 CLI 可用 `--wav audio/asr_test.wav --save-audio` 复现整条链路。
- 显存不足时先用 `LLM_DEVICE=cpu`，再考虑 `--no-graph`（只影响 TTS 速度）。

## 7. 提交与 PR

- 提交信息使用 Conventional Commits：`feat:`、`fix:`、`docs:`、`test:`、
  `refactor:`、`chore:`。
- 一个 PR 只做一件事；行为变更需要同步更新文档和测试。
- 提交前本地运行：

```bash
uv run ruff check src tests
uv run pytest -q
```

- PR 描述请说明：动机、改动、如何验证、是否有破坏性变更。
- 新增模型/协议时附上实测数据或截图。

详见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

## 8. 文档维护

- 文档入口：[docs/README.md](README.md)。
- 配置字段变更：同步 `docs/configuration.md`。
- 接口变更：同步 `docs/api/`。
- 模型/性能变更：同步 `docs/models.md`。
- 目录结构变更：同步 `README.md` 和本文档。
