# g2-policy-protocol

连接 wholebody 域控与 LeRobot、OpenPI 或第三方模型服务的共享 Python 包：通信层提供消息校验、JSON/MessagePack 与 WebSocket 服务，`g2_policy_protocol.data` 提供数据契约、policy 语义、状态/动作编码和时序采样。

模型训练与推理由模型项目维护；采集、调度和运动约束由 wholebody 维护。共享包不依赖 Torch、JAX、LeRobot 或机器人运行时。

## 文档导航

| 要做什么 | 文档 |
|---|---|
| 安装、跑通示例、接入模型或 wholebody、验收 | [安装与接入](docs/guide.md) |
| 查消息、控制量、context与 RTC | [通信协议](docs/protocol.md) |
| 配置训练输入、动作、坐标、差分和采样 | [policy 配置](docs/policy-config.md) |
| 理解架构与数据边界、修改代码、测试和交付 | [开发维护](docs/development.md) |

从[安装与接入](docs/guide.md)开始。当前源码版本 **0.6.0**，只使用 `g2.policy.v3`；两端使用同一份源码或 wheel。

完整训练 YAML 和训练命令由模型仓库维护，见 [OpenPI](../openpi/scripts/g2/docs/training.md)和 [LeRobot](../lerobot-finetune/scripts/g2/docs/training.md)。
