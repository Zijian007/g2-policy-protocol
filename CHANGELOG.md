# CHANGELOG

## 2026-11-13：协议库结构收敛

- 统一为一个顶层包 `g2_policy_protocol`；通用采样、编码、几何、统计和混合归入 `processing/`。
- 将状态编码和动作编解码合并为 `processing.encoding`，将 policy 类型、默认值和校验集中到 `contracts.policy`。
- 将诊断样本和服务检查合并为 `diagnostics.py`，更新 `g2-policy-check` 入口。
- 将 PI0.5 输入布局、历史图像拼接和模型维度补齐迁回 OpenPI / LeRobot 各自适配器。
- 保持 `g2.policy.v3`、消息字段、动作数值语义和推理/reset 不自动重发的行为不变。
- 清理生成的旧源码副本、旧 wheel 和已完成的图像历史实施计划；现行图像历史规则统一维护在 `docs/protocol.md`。

## 2026-11-13：运行时和消息入口整理

- 完整消息类型、固定向量布局、单位、历史顺序和 session 语义集中在 `types.py`。
- `protocol.py` 按 metadata、obs、action/error、reset 顺序提供构造和解析入口；`runtime/server.py` 清楚呈现解析、解码、推理、校验和回包流程。
- `PolicyClient` 负责握手、请求 ID、超时、取消、响应校验和连接清理；连接尝试可重试，推理和 reset 不自动重发。
- 图像解码在模型工作线程完成；backend 接收 RGB uint8 的 `Observation`，返回完整双臂绝对动作。

## 2026-11-02：图像历史传输

- 图像历史支持 raw 数组和 JPEG bytes 列表，历史从旧到新排列，最后一帧为当前帧。
- 有完整图像历史时可以省略重复的 `images`；服务端运行时统一解码并校验尺寸。
- JSON 和 MessagePack 都支持历史图像的对应表示；不匹配的表示会在请求校验阶段拒绝。

## 2026-10-30：g2.policy.v3

- 线上只接受 `g2.policy.v3`。
- 握手包含 `action_spec`、`observation_spec`、`timing` 和 reset 能力；状态、关节状态、历史和 context 按声明校验。
- 动作响应是完整双臂的绝对 pose20 或 joint16；模型内部的 delta、坐标系和 padding 由适配器还原。
- 磁盘数据继续使用固定的 state/action 列布局；训练配置中的 action 监督通过 policy source 表达。

更早的版本历史只保留线上兼容性相关内容；当前源码结构和导入路径以 README 及开发文档为准。
