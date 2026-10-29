# g2.policy.v3

v3 为不同模型声明相同的 G2 观测和控制契约。v2 pose20 API 仍保留，client 按握手 `protocol` 选择校验。私有 `g2.openpi.v1` 已移除。升级 v3 服务时必须同步升级客户端到 `g2-policy-protocol==0.6.0`：0.4.1 会把 `action_spec`、`observation_spec`、`reset_supported`、`timing` 判为 unknown 并断开。当前 OpenPI G2 server 发送 `g2.policy.v3`。

基础握手保留 action_horizon、policy_hz、arm、required_cameras、图像传输设置、rtc_supported、requires_joint_state。新增：

| 字段 | 含义 |
| --- | --- |
| `action_spec` | control_quantity、dim、fields、units、frame、ee_frame、representation |
| `observation_spec` | pose/joint/prompt 是否必需、history_steps、history_hz、必需 context_keys |
| `timing.action_dt` | 相邻预测动作间隔，必须等于 1/policy_hz |
| `timing.first_target_offset_s` | 首个目标相对本次观测时刻的偏移 |
| `reset_supported` | 是否实现当前会话的 episode reset |

`required_cameras` 可以为空或只选腕部；`prompt` 仅在 observation_spec.prompt=true 时要求非空。`state` 仍是 canonical pose20，`joint_state` 仍是 joint16，但只在声明需要时强制携带。`requires_joint_state` 必须与 observation_spec.joint 一致。

控制量固定为规范化的完整双臂布局，左右顺序，每臂最后一维都是绝对夹爪 close_ratio（0 开、1 闭）。使用 `protocol.v3.action_spec(quantity)` 构造，校验拒绝错配的维度、字段、单位和坐标系：

| control_quantity | D | 每臂分量与单位 |
| --- | --- | --- |
| pose_target | 20 | xyz 米、rot6d 两列按 F 顺序、gripper |
| joint_position | 16 | 7 关节 rad、gripper |
| joint_velocity | 16 | 7 关节 rad/s、gripper |
| joint_torque | 16 | 7 关节 N*m、gripper |

位姿使用 base_link 下的 flange；TCP、局部参考、delta 或旋转表示由模型适配层转换。关节按 joint_fields 顺序。`arm` 限定实际执行侧，另一侧位置保持实测值，另一侧速度/力矩为零。

请求为 `type=obs`、request_id、可选 prompt/state/joint_state、images、context、observation_time_ns 与可选 RTC 字段。历史 L>1 时必须给 observation_time_ns（客户端本机单调时钟）；历史从最旧到当前时刻，间隔 1/history_hz。服务端不将这个值与另一台机器的时钟相减。

context 支持 command_pose、command_joint、previous_command_pose、start_pose、start_joint、object_pose、object_joint、base_to_parent、pose_history、joint_history、image_history。位姿、关节参考分别为20/16维，坐标系变换为有效4×4刚体矩阵。历史形状为 (L,20)/(L,16)，相机历史为按相机名组织的原始 RGB uint8 (L,H,W,3)，尺寸遵循 image_sizes；当前 images 可使用 JPEG。context 的必需字段由 observation_spec 声明。当前训练配置若相对指令，只声明 command_pose / command_joint；其余键仍可在握手中选用。previous_command_pose 对应 observation_time_ns - 1/history_hz 的命令，而不是上一拍机器人控制发布值。

响应 `type=action`、相同 request_id、有限数组 actions(T,D)，1≤T≤H，可携带 timing/rtc_info。调用 `build_response(..., metadata=meta)`、`parse_response(..., metadata=meta)` 校验 D。RTC prefix 也必须使用相同控制量和 D，delay 不得超过 prefix 或 H。支持 RTC 不意味着客户端负责某种特定模型的 guidance 算法。

显式 episode 边界：客户端发 `{"type":"reset","request_id":N}`，server 清理该会话的历史、参考和模型状态后回 `reset_ack`。只有握手声明 reset_supported 时可调用。同一连接可以承载多个 episode；换连接也创建独立 session。reset 与机器人归位命令无关。

H 是模型预测长度，K 是客户端最多承诺执行的前缀长度，r 是该前缀从预测起点经过多少步后请求下一段。K 不属于新训练模板；旧握手 execution_steps 作为兼容建议，客户端显式 K 优先。wholebody 保存 H，并只将前 K 个点放入执行/RTC 队列；实际已过期的前缀仅裁掉一次。例如 H=50、K=20、r=10，在经过10步时补段，最多提交20步。

schema 位于 `g2.v3.schema.json`；`tools/generate_protocol.py` 同时生成并检查 v2/v3 静态验证。数组形状、有限数值和协商后的动态需求由 protocol/v3.py 检查。模型后端实现见 [分层与接入](architecture.md)。
