# 多 policy 共用 G2 基础设施

各模型保留自己的训练配置、权重格式、网络、损失和推理算法。服务端适配器把这些差异封装在 `PolicyBackend` 后面；机器人端按动作控制量、观测需求和已绑定的执行器处理请求，不读取模型名称。

| 层 | 归属与接口 |
| --- | --- |
| 模型与训练 | 各 policy 仓库；OpenPI 的 Pi05/Gemma/LoRA/token/JAX 配置继续留在 OpenPI |
| G2 数据与几何 | `g2_policy_tools.contract`、`geometry.ActionCodec`、`dataset.G2Dataset`；无需导入 OpenPI、wholebody、PyTorch 或 JAX |
| 通信 | `g2_policy_protocol`：版本化 schema、生成校验、JSON/MessagePack、`server.PolicyServer` 和 `PolicyBackend` 接口 |
| 模型适配 | 各 policy 实现 canonical request → 模型观测 → canonical actions；OpenPI 用 `OpenPIBackend`，训练转换也使用同一个 `ActionCodec` |
| 机器人执行 | wholebody 的观测采集、时间队列、插值、安全门、HAL；按 `control_quantity` 选择绑定的 executor |

两个 Python 命名空间暂由同一个 `g2-policy-protocol==0.6.0` 轻量发行包提供，便于三个仓库同步升级。域控与 GPU 必须同装这一版本：v3 server 发给 0.4.1 client 会在握手时被拒。基础依赖是 numpy/msgpack；几何旋转使用可选 `geometry` extra 的 SciPy，WebSocket server 使用 `tools` extra。导入协议不加载模型或机器人运行时。原 `openpi.adapters.g2` 与 `wholebody.shared.g2_lerobot_contract` 的导入路径保留兼容转发。

模型内部可以预测 delta、SE(3)、不同旋转表示、单臂或补齐维度；适配器必须还原到声明的完整双臂绝对控制量。协议不会传输 Pi05 token 设置、归一化参数、LoRA 或 padded action_dim=32。

新模型接入只需：

1. 在自己的仓库定义训练配置与模型加载；可复用几何/数据工具。
2. 实现 backend 的 `infer(request)`，返回 `{"actions": array}`；有 episode 状态时实现独立 `new_session()` 与 `reset()`。共享模型权重，隔离会话状态。
3. 用 `protocol.build_metadata(version=3, ...)` 声明真实能力，并交给 `PolicyServer`。需要 RTC 时自行实现对应算法，只有实现后才声明支持。
4. wholebody 校验该控制量已有 executor，且能提供所有观测。接入一种新模型不需要添加客户端模型分支。

当前真实应用入口绑定 `pose_target`（EE 安全门）和 `joint_position`（Gello 真机关节限位 + 每拍 slew 转成 rad/s，经 `ArmJointPublisher`）。关节速度、力矩的协议和数据转换已支持；这些执行器尚未绑定，应用在打开硬件前拒绝。

历史策略通过独立采样线程在 `history_hz` 上采集，episode 开头重复首帧，之后的采样缺口会报错。命令、物体、动态坐标系需要显式 provider，不能从当前实测末端伪造。默认应用支持历史和 episode 起点；外部 provider 在 `PolicyControlLoop(context_provider=...)` 接入。

当前机器人调度要求 `first_target_offset_s=0`。非零偏移可在训练和 server/协议中表达，应用在硬件启动前拒绝，避免提前执行未来目标。没有为新 policy 实现或替换其神经网络，也没有验证真机轨迹。
