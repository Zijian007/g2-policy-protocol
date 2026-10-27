# g2-policy-protocol

0.5.0 在同一发行包中提供 `g2.policy.v2` 与模型无关的 [v3](docs/protocol-v3.md)。v2 保留固定 pose20 API；v3 用握手声明控制量和观测需求。分层边界见 [architecture.md](docs/architecture.md)。共用几何/数据工具位于 `g2_policy_tools`，训练配置与网络实现仍由各 policy 仓库维护。

框架无关的 G2 policy 通信契约，连接 wholebody 域控与 G2_pi、OpenPI 或第三方模型服务。client 按握手 `protocol` 字段选择校验：v2 固定 pose20，v3 按 `action_spec` 校验。OpenPI 的 G2 入口在 `feat/policy-config-v2` 的 `openpi/src/openpi/adapters/g2/server.py`：有训练配置快照时发 `g2.policy.v3`，否则仍是 v2 pose20。仍发送 v2 metadata 的服务可继续用同一份 0.5.0 client。

```text
wholebody（域控 / client）                 G2_pi / OpenPI（GPU / server）
采集实测状态、准备图像     ── obs ──>      适配模型输入、执行推理
执行经过校验的动作         <─ action ──    转换为握手声明的绝对控制量
                    两端共用 g2-policy-protocol
```

本包提供消息构造、解析、校验以及 MessagePack / JSON 编解码。WebSocket 连接、模型推理、图像处理和机器人运动控制由使用它的项目负责。

## 按使用角色阅读

| 读者 | 文档 | 内容 |
|---|---|---|
| 域控开发与部署人员 | [For 域控 / Client](docs/client.md) | wholebody 环境安装、连接与能力校验、观测发送、动作接收 |
| GPU 服务开发与部署人员 | [For GPU Server](docs/server.md) | 安装、最小服务、能力配置、模型适配、独立验收与交付 |
| v2 接入 | [协议参考](docs/protocol.md) | v2 消息、pose20、图像、RTC、编码与可运行回环示例 |
| v3 接入 | [v3 协议](docs/protocol-v3.md) | action_spec、observation_spec、timing、reset、控制量 |
| 多模型分层 | [分层接入](docs/architecture.md) | PolicyBackend、几何工具、执行器绑定 |
| 协议维护人员 | [开发与发布](docs/development.md) | Schema、代码生成、测试、打包与跨端升级 |

## 安装与版本要求

**域控和 GPU server 都必须安装同一版本 `g2-policy-protocol==0.5.0`。** 各自环境的安装方式见对应接入文档。跨机部署推荐分发同一个 wheel。

- Python ≥3.10，运行依赖只有 NumPy、MessagePack；不依赖机器人库、Torch、JAX、LeRobot、OpenCV 或 WebSocket 库。
- 0.5.0 同时识别 `g2.policy.v2` 和 `g2.policy.v3`。不兼容 `g2.policy.v1` 和更早的两套 G2 消息。
- **0.4.1 client 不能连接 v3 server。** v2 校验会把 `action_spec`、`observation_spec`、`reset_supported`、`timing` 判为 unknown 并断开。
- 升级时同步更新两端代码和协议包，重启 GPU server 后再启动域控 client。不能只替换一端。
- OpenPI 非 G2 机器人的通用原生 server 不依赖本包。

结构定义：v2 为 [`g2.schema.json`](src/g2_policy_protocol/g2.schema.json)，v3 为 [`g2.v3.schema.json`](src/g2_policy_protocol/g2.v3.schema.json)。共享字段和固定语义见对应协议文档，两端文档只描述各自的接入职责。
