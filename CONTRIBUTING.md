# 参与贡献

感谢你愿意参与 Native Voice！本文说明如何报告问题、提交代码和参与评审。

参与本项目即表示你同意遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

## 可以做什么

- 🐛 报告 Bug 或复现步骤
- 💡 提出新功能、模型适配或协议兼容建议
- 📝 改进文档、示例和翻译
- 🧪 补充测试、压测数据和兼容性验证
- 🔧 修复代码、优化延迟和显存占用
- 🎨 改进网页客户端和交互

不确定从哪里开始？找带
[`good first issue`](https://github.com/easyon/native-voice/labels/good%20first%20issue)
标签的任务，或在 Issue 里说明你想做的方向。

## 报告问题

提交 Issue 前请先搜索是否已有相同问题。Bug 报告请包含：

1. 操作系统、GPU / 显存、驱动版本；
2. Python、torch、transformers 版本；
3. 运行方式（服务 / 单机 CLI / fake 后端）与启动命令；
4. 相关配置片段（隐去 API Key、短票、路径中的用户名）；
5. 完整错误日志；
6. 预期行为与实际行为。

功能建议请说明使用场景、期望接口和替代方案。

## 开发环境

```bash
git clone https://github.com/easyon/native-voice.git
cd native-voice
uv sync --extra dev
```

需要真实模型时再安装 GPU 依赖并下载资源：

```bash
uv sync --extra gpu
uv run --extra gpu python scripts/fetch_deps.py
```

只改网页/协议时可以直接使用 fake 后端：

```powershell
scripts\serve_dev.ps1 -Backend fake
```

## 提交代码

1. Fork 仓库并从 `main` 创建分支：

   ```bash
   git checkout -b feat/my-feature
   ```

2. 保持改动聚焦。一个 PR 只解决一个问题。
3. 遵循现有代码风格：
   - 行宽 110，Python 3.10；
   - 公共接口加类型提示与 docstring；
   - 日志用 `logging`，不要在库代码里 `print`；
   - 指标标签不要包含用户文本、Key 或 session id。
4. 添加或更新测试。纯逻辑放 `tests/unit/`，协议相关放 `tests/protocol/`。
5. 更新文档：配置、接口、模型或目录结构变化时，同步修改 `docs/`。
6. 提交前运行：

   ```bash
   uv run ruff check src tests
   uv run pytest -q
   ```

7. 提交信息使用 [Conventional Commits](https://www.conventionalcommits.org/)：

   ```text
   feat: 支持新的流式 TTS provider
   fix: 修复打断时首字丢失
   docs: 补充生产部署说明
   test: 增加短票过期用例
   refactor: 拆分 ASR 协议处理
   chore: 更新依赖
   ```

8. 推送并发起 Pull Request，说明动机、改动、验证方式和破坏性变更。

## PR 检查清单

- [ ] 改动聚焦，没有无关格式化或重排
- [ ] 新增/修改的行为有测试覆盖
- [ ] `ruff check` 与 `pytest -q` 通过
- [ ] 文档已更新（配置、API、目录结构、性能数据）
- [ ] 没有提交模型权重、二进制、日志、密钥或 `.data/`
- [ ] 涉及第三方代码/模型时，确认许可兼容并注明来源

## 新增 ASR / TTS provider

1. 在 `src/native_voice/models/<asr|tts>/` 实现适配器，遵循现有接口。
2. 在对应 `factory.py` 注册名称。
3. 在 `configs/service.yaml` 的 `providers` 下增加解释器、模型和选项。
4. 如依赖不同的 `transformers` 版本，说明独立 venv 的创建方式。
5. 补充测试（尽量使用 mock/fake，避免 CI 加载大模型）。
6. 在 `docs/models.md` 增加 provider 说明、限制和实测数据。

## 安全与隐私

- 不要提交 API Key、短票、数据库、用户音频或日志正文。
- 发现安全漏洞请按 [SECURITY.md](SECURITY.md) 私下报告，不要公开 Issue。
- 声音克隆功能必须保留 `consent` 校验；不要提交未经授权的声音样本。

## 文档风格

- 使用简体中文，术语与代码保持一致。
- 命令区分 Windows PowerShell 与 POSIX shell。
- 链接使用相对路径，确保在 GitHub 和本地都能打开。
- 性能数据注明硬件、软件版本和测量方式。

## 许可证

提交贡献即表示你同意以 [Apache License 2.0](LICENSE) 授权你的贡献。
