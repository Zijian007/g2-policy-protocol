# g2-policy-protocol

0.6.0 在同一发行包中提供 `g2.policy.v2` 与模型无关的 [v3](docs/protocol-v3.md)。v2 保留固定 pose20 API；v3 用握手声明控制量和观测需求。分层边界见 [architecture.md](docs/architecture.md)。共用配置语义、几何与数据工具位于 `g2_policy_tools`；模型训练和推理由各 policy 仓库维护。

框架无关的 G2 policy 通信契约，连接 wholebody 域控与 lerobot-finetune、OpenPI 或第三方模型服务。client 按握手 `protocol` 字段选择校验：v2 固定 pose20，v3 按 `action_spec` 校验。第一方 LeRobot、OpenPI G2 服务使用 checkpoint 配置快照声明 v3 能力，并返回完整预测块。第三方或旧服务仍可发送 v2 metadata，由同一份 0.6.0 client 接入。

```text
wholebody（域控 / client）                 lerobot-finetune / OpenPI（GPU / server）
采集实测状态、准备图像     ── obs ──>      适配模型输入、执行推理
执行经过校验的动作         <─ action ──    转换为握手声明的绝对控制量
                    两端共用 g2-policy-protocol
```

本包提供消息构造、解析、校验及 MessagePack / JSON 编解码；`g2_policy_tools` 提供共享配置、几何转换和时序采样。安装可选依赖后还可使用通用 WebSocket 服务与图像解码工具。模型推理和机器人运动控制由使用它的项目负责。

## 按使用角色阅读

| 读者 | 文档 | 内容 |
|---|---|---|
| 域控开发与部署人员 | [For 域控 / Client](docs/client.md) | wholebody 环境安装、连接与能力校验、观测发送、动作接收 |
| GPU 服务开发与部署人员 | [For GPU Server](docs/server.md) | 安装、最小服务、能力配置、模型适配、独立验收与交付 |
| v2 接入 | [协议参考](docs/protocol.md) | v2 消息、pose20、图像、RTC、编码与可运行回环示例 |
| v3 接入 | [v3 协议](docs/protocol-v3.md) | action_spec、observation_spec、timing、reset、控制量 |
| 多模型分层 | [分层接入](docs/architecture.md) | PolicyBackend、几何工具、执行器绑定 |
| 训练调试 | [policy 字段速查](docs/policy-config.md) | state/action 候选、条件约束、坐标与切换示例 |
| 训练配置 | [公共训练配置](docs/training-config.md) | OpenPI / LeRobot 公共字段、后端选项和 checkpoint 格式 |
| 协议维护人员 | [开发与发布](docs/development.md) | Schema、代码生成、测试、打包与跨端升级 |

## 安装与版本要求

**域控和 GPU server 都必须安装同一版本 `g2-policy-protocol==0.6.0`。** 各自环境的安装方式见对应接入文档。跨机部署推荐分发同一个 wheel。

- Python ≥3.10；基础依赖只有 NumPy、MessagePack。`[geometry]` 增加 SciPy，`[tools]` 增加 WebSocket 与图像工具所需的 websockets、Pillow；不依赖机器人库、Torch、JAX、LeRobot 或 OpenCV。
- 0.6.0 同时识别 `g2.policy.v2` 和 `g2.policy.v3`。不兼容 `g2.policy.v1` 和更早的两套 G2 消息。
- **0.4.1 client 不能连接 v3 server。** v2 校验会把 `action_spec`、`observation_spec`、`reset_supported`、`timing` 判为 unknown 并断开。
- 升级时同步更新两端代码和协议包，重启 GPU server 后再启动域控 client。不能只替换一端。
- OpenPI 非 G2 机器人的通用原生 server 不依赖本包。

结构定义：v2 为 [`g2.schema.json`](src/g2_policy_protocol/g2.schema.json)，v3 为 [`g2.v3.schema.json`](src/g2_policy_protocol/g2.v3.schema.json)。共享字段和固定语义见对应协议文档，两端文档只描述各自的接入职责。

链路职责、段边界、H/R/K、升级和跨环境联调见 [完整链路](docs/pipeline.md)。
