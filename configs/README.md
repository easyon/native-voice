# 配置

| 文件 | 说明 |
|---|---|
| `service.yaml` | 服务配置：监听、认证、配额、模型、engine、LLM、对话 |
| `voices.yaml` | 内置音色（参考音频与参考文本） |

字段说明见 [`docs/configuration.md`](../docs/configuration.md)。

注意：

- 不要在这里写入真实 API Key、短票密钥或生产数据库路径；
  敏感值通过环境变量或部署配置注入。
- 相对路径以仓库根目录为基准。
- 修改 provider 或 engine 拓扑后，重启网关与对应 engine。
