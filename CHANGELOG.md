# CHANGELOG

按日期记录。线上通信协议当前为 `g2.policy.v3`；包版本号固定不变，两端以 `git pull` 同步。

## 2026-11-02

- 有完整 `context.image_history` 的观测可省略 `images`，`build_request(images=None)` 不生成该字段，解析后也不恢复重复图像；无历史时仍要求 `images`。wholebody 与诊断客户端跳过历史请求的当前帧准备和发送；两个第一方 backend 直接使用历史。保持 `g2.policy.v3`；新服务端接受旧请求，新客户端历史请求需要先更新服务端协议包及适配代码。
- 图像历史编码跟随握手 `image_encoding`：JPEG 用每帧 bytes 列表，raw 保持 uint8 (L,H,W,3)；新增 `images.decode_image_history`，解码后校验尺寸。`ImageDecodeError` 是 `ValueError` 子类，`PolicyServer` 将其归为 `invalid_request`，模型的其他异常仍为 `inference_error`。保持 `g2.policy.v3`，JPEG 历史的新旧表示会被请求校验拒绝，协议包与两端适配代码需同步更新。
- 删除 `g2.schema.json`、`tools/generate_protocol.py` 和 `_generated.py`：固定布局与枚举改由手写的 `_constants.py` 定义，消息静态校验写在 `protocol.py`；`data.contract` 和 `build_action_spec` 从同一份常量派生。开发依赖不再需要 `jsonschema`。
- `_validation.py` 并入 `validation.py`；`from g2_policy_protocol.validation import validate_rigid_transform` 不变。
- 文档合并为 README（职责、开发、跨仓库验证）、`docs/install.md`（域控侧 / GPU 服务器侧安装）和 `docs/protocol.md`；删除 `docs/guide.md`、`docs/development.md`。
- 安装方式改为相邻 checkout + editable：OpenPI 的 uv source 设 `editable = true`，LeRobot bootstrap 与项目同一次解析安装协议，避免被拷贝覆盖；不再人工核对两端版本，wheel 仅用于无仓库权限的交付。
- 字段 unknown/missing、协议版本或动作布局不符时，报错末尾提示两端可能是不同版本的协议包。结构不变但线上语义改变的改动必须升级 `PROTOCOL`。
- `connection_uri` 从 `check_server` 移到 `codec`（与服务端的 `encoding_from_path` 对应）；`images.decode_images` 遇到损坏的 JPEG 抛 `ValueError`，server 回 `invalid_request`。
- 源码结构调整（线上协议不变，下游导入需同步）：
  - 消息静态结构检查移到 `_messages.py`；`protocol.py` 只保留公开 API。
  - `samples`、`check_server` 移到 `diagnostics/`；命令行入口为 `g2_policy_protocol.diagnostics.check_server:main`。
  - 旋转与位姿转换从 `data.encoding` 拆到 `data.geometry`，并按方向改名：`pose_matrix` → `pose_to_matrix`、`matrix_pose` → `matrix_to_pose`、`rotation_matrix` → `rotation_to_matrix`、`rotation_vector` → `matrix_to_rotation`。
  - `data.policy` 新增 `uses_kind`、`command_reference_kinds`、`uses_arm_base`，`policy_capabilities`、`validate_dataset_contract`、`TemporalDataset` 共用；`action_branch` 改名 `active_action`。
  - `images.resize_rgb` 新增 `mode`（`letterbox` / `resize`）。
- 不再使用包版本号区分两端：`pyproject.toml` 版本固定，下游依赖不再锁定版本。
- 线上 `g2.policy.v3` 和公开的 `protocol` 接口不变。

## 2026-10-31

- 新增 `data.sources`：本地 `path`、可选名称、全填或全不填的正权重，以及可复现的 episode 级重复计划和虚拟区间映射。
- 新增 `data.image_size` 校验、RGB letterbox、数据身份与混合指纹工具；共享模块不依赖训练框架。
- 训练 YAML 保持 schema 7，旧 `data.root + data.repo_id` 不变。包含 `sources` 或 `image_size` 的新快照需要配套升级训练端和部署端；旧部署代码会拒绝这些新字段。
- 线上 `g2.policy.v3` 与 2026-10-30 相同，磁盘数据契约不变。

## 2026-10-30

线上协议：只接受 `g2.policy.v3`。

不兼容变更与迁移：

- 移除 `g2.policy.v2` 及版本选择参数；握手中不再有 `execution_steps`，执行步数完全由 client（wholebody 的 `infer.execution_steps`）决定。
- 握手新增 `action_spec`、`observation_spec`、`timing`、`reset_supported`；`state` / `joint_state` 只在 `observation_spec` 声明需要时强制携带。
- `build_response` / `parse_response` 必须传入 metadata，按 `action_spec.dim` 校验动作宽度；动作布局用 `build_action_spec(quantity)` 构造。
- 新增 `reset` / `reset_ack` 消息。
- Python API：数据工具统一在 `g2_policy_protocol.data`，图像工具移到 `g2_policy_protocol.images`；旧导入路径不保留。
- 导出配置 `action_supervision: state` 在入口被拒绝；测量监督改由 policy 的 `source: state` 选择。

不变：磁盘列格式沿用现有契约。
