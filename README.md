# g2-policy-protocol

框架无关的 G2 policy 通信契约，连接 wholebody 域控与 G2_pi、OpenPI 或第三方模型服务。ACT、LeRobot π₀.₅ 和 OpenPI 共用 `g2.policy.v2`。

```text
wholebody（域控 / client）                 G2_pi / OpenPI（GPU / server）
采集实测状态、准备图像     ── obs ──>      适配模型输入、执行推理
执行经过校验的动作         <─ action ──    转换为绝对 pose20 动作
                    两端共用 g2-policy-protocol
```

本包提供消息构造、解析、校验以及 MessagePack / JSON 编解码。WebSocket 连接、模型推理、图像处理和机器人运动控制由使用它的项目负责。

## 按使用角色阅读

| 读者 | 文档 | 内容 |
|---|---|---|
| 域控开发与部署人员 | [For 域控 / Client](docs/client.md) | wholebody 环境安装、连接与能力校验、观测发送、动作接收 |
| GPU 服务开发与部署人员 | [For GPU Server](docs/server.md) | 安装、最小服务、能力配置、模型适配、独立验收与交付 |
| 两端接入开发人员 | [协议参考](docs/protocol.md) | 消息、pose20、图像、RTC、编码与可运行回环示例 |
| 协议维护人员 | [开发与发布](docs/development.md) | Schema、代码生成、测试、打包与跨端升级 |

## 安装与版本要求

**域控和 GPU server 都必须安装同一版本 `g2-policy-protocol==0.4.1`。** 各自环境的安装方式见对应接入文档。跨机部署推荐分发同一个 wheel。

- Python ≥3.10，运行依赖只有 NumPy、MessagePack；不依赖机器人库、Torch、JAX、LeRobot、OpenCV 或 WebSocket 库。
- 当前通信协议为 `g2.policy.v2`，不兼容 `g2.policy.v1` 和更早的两套 G2 消息。
- 升级时同步更新两端代码和协议包，重启 GPU server 后再启动域控 client。
- OpenPI 非 G2 机器人的通用原生 server 不依赖本包。

**唯一结构定义是 [`g2.schema.json`](src/g2_policy_protocol/g2.schema.json)**。共享字段和固定语义集中在[协议参考](docs/protocol.md)，两端文档仅描述各自的接入职责。
