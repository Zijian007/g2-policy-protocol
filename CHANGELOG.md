# CHANGELOG

协议不做跨版本兼容：升级任何一项都需要两端同时换成同一个新 wheel，先重启 GPU server，再启动域控 client。

## 0.6.0

线上协议：只接受 `g2.policy.v3`。

不兼容变更与迁移：

- 移除 `g2.policy.v2` 及版本选择参数；握手中不再有 `execution_steps`，执行步数完全由 client（wholebody 的 `infer.execution_steps`）决定。
- 握手新增 `action_spec`、`observation_spec`、`timing`、`reset_supported`；`state` / `joint_state` 只在 `observation_spec` 声明需要时强制携带。
- `build_response` / `parse_response` 必须传入 metadata，按 `action_spec.dim` 校验动作宽度；动作布局用 `build_action_spec(quantity)` 构造。
- 新增 `reset` / `reset_ack` 消息。
- Python API：数据工具统一在 `g2_policy_protocol.data`，图像工具移到 `g2_policy_protocol.images`；旧导入路径不保留。
- 导出配置 `action_supervision: state` 在入口被拒绝；测量监督改由 policy 的 `source: state` 选择。

不变：磁盘列格式沿用现有契约。
