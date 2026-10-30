# g2-policy-protocol

连接 wholebody 域控与 LeRobot、OpenPI 或第三方模型服务的共享 Python 包：通信层提供消息校验、JSON/MessagePack 与 WebSocket 服务，`g2_policy_protocol.data` 提供数据契约、policy 语义、状态/动作编码和时序采样。

模型训练与推理由模型项目维护；采集、调度和运动约束由 wholebody 维护。共享包不依赖 Torch、JAX、LeRobot 或机器人运行时。

## 部署拓扑

```text
域控（G2 真机）                                   GPU 服务器
  wholebody                                         openpi 或 lerobot-finetune
    采集 · 调度 · 安全门 · HAL 执行                   训练 · 推理 · 模型适配（backend）
    G2WebSocketPolicyClient  ── obs ──────────▶     PolicyServer
                             ◀── action/error ──
  g2-policy-protocol（通信层）                      g2-policy-protocol（通信层 + data）

                  WebSocket，协议 g2.policy.v3，MessagePack 或 JSON
```

两台机器都安装**同一个** `g2-policy-protocol` wheel（或同一份源码），协议不做跨版本兼容：

| 机器 | 使用方 | 用到的部分 | 安装方式 |
|---|---|---|---|
| 域控 | wholebody | 通信层：握手校验、观测编码、动作解析 | 装进 wholebody 的 venv；不装 Torch/JAX |
| GPU 服务器 | openpi / lerobot-finetune | 训练用 `data`（数据契约、编码、采样）；部署用通信层起 server | 各仓库 bootstrap，设 `G2_PROTOCOL_WHEELS` 时安装 wheel，否则安装相邻源码 |

升级顺序：两端换成同一个新 wheel → 重启 GPU server → 启动域控 client。详见[安装与版本](docs/guide.md#安装与版本)。

## 文档导航

| 要做什么 | 文档 |
|---|---|
| 安装、跑通示例、接入模型或 wholebody、验收、排错 | [安装与接入](docs/guide.md) |
| 查术语、消息格式、控制量、context 与 RTC | [通信协议](docs/protocol.md) |
| 配置训练输入、动作、坐标、差分和采样 | [policy 配置](docs/policy-config.md) |
| 理解架构与数据边界、修改代码、测试和交付 | [开发维护](docs/development.md) |
| 版本变更与迁移 | [CHANGELOG](CHANGELOG.md) |

从[安装与接入](docs/guide.md)开始；不熟悉 pose20、rot6d、RTC 等缩写时先看[术语](docs/protocol.md#术语)。当前源码版本见 `pyproject.toml`（0.6.0），只使用 `g2.policy.v3`。

完整训练 YAML 和训练命令由模型仓库维护，见 [OpenPI](../openpi/scripts/g2/docs/training.md)和 [LeRobot](../lerobot-finetune/scripts/g2/docs/training.md)。这类跨仓库链接需要四个仓库位于同一父目录。
