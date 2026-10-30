# CHANGELOG

线上通信协议当前为 `g2.policy.v3`。不兼容协议升级需要同时更新客户端和服务端；新增训练工具不改变线上消息格式。

## 0.7.0

- 新增 `data.sources`：本地 `path`、可选名称、全填或全不填的正权重，以及可复现的 episode 级重复计划和虚拟区间映射。
- 新增 `data.image_size` 校验、RGB letterbox、数据身份与混合指纹工具；共享模块不依赖训练框架。
- 训练 YAML 保持 schema 7，旧 `data.root + data.repo_id` 不变。包含 `sources` 或 `image_size` 的新快照需要配套升级训练端和部署端；旧部署代码会拒绝这些新字段。
- 线上 `g2.policy.v3` 与 0.6.0 相同，磁盘数据契约不变。

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
