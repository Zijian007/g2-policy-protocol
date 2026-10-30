# 通信协议

[文档入口](../README.md#文档导航) · [安装与接入](guide.md) · [policy 配置](policy-config.md)

通信只接受 `g2.policy.v3`，握手阶段拒绝其他协议。包版本与升级要求见[安装与版本](guide.md#安装与版本)。

## 连接与消息

WebSocket 默认使用二进制 MessagePack；URL 带 `?encoding=json` 时整条连接使用 JSON text frames。server 先发送 `metadata`，client 完成能力校验后发送 `obs`，server 返回相同 `request_id` 的 `action` 或 `error`。编码由连接参数确定，不在 metadata 中重复声明；整条连接不混发帧类型。

## 握手与能力

基础握手保留 action_horizon、policy_hz、arm、required_cameras、图像传输设置、rtc_supported、requires_joint_state。新增：

| 字段 | 含义 |
| --- | --- |
| `action_spec` | control_quantity、dim、fields、units、frame、ee_frame、representation |
| `observation_spec` | pose/joint/prompt 是否必需、history_steps、history_hz、必需 context_keys |
| `timing.action_dt` | 相邻预测动作间隔，必须等于 1/policy_hz |
| `timing.first_target_offset_s` | 首个目标相对本次观测时刻的偏移 |
| `reset_supported` | 是否实现当前会话的 episode reset |

`required_cameras` 可以为空或只选腕部；`prompt` 仅在 observation_spec.prompt=true 时要求非空。`state` 仍是 canonical pose20，`joint_state` 仍是 joint16，但只在声明需要时强制携带。`requires_joint_state` 必须与 observation_spec.joint 一致。

## 动作控制量

控制量固定为规范化的完整双臂布局，左右顺序，每臂最后一维都是绝对夹爪 close_ratio（0 开、1 闭）。使用 `protocol.build_action_spec(quantity)` 构造，校验拒绝错配的维度、字段、单位和坐标系：

| control_quantity | D | 每臂分量与单位 |
| --- | --- | --- |
| pose_target | 20 | xyz 米、rot6d 两列按 F 顺序、gripper |
| joint_position | 16 | 7 关节 rad、gripper |
| joint_velocity | 16 | 7 关节 rad/s、gripper |
| joint_torque | 16 | 7 关节 N*m、gripper |

pose20 顺序为左臂 xyz(3)、rot6d(6)、close_01(1)，再右臂；rot6d 为 `R[:, :2].reshape(-1, order="F")`。joint16 为左右各 7 个关节及夹爪。

位姿使用 base_link 下的 flange；TCP、局部参考、delta 或旋转表示由模型适配层转换。关节按 joint_fields 顺序。`arm` 限定实际执行侧，另一侧位置保持实测值，另一侧速度/力矩为零。

## 观测、历史与参考

请求为 `type=obs`、request_id、可选 prompt/state/joint_state、images、context、observation_time_ns 与可选 RTC 字段。历史 L>1 时必须给 observation_time_ns（客户端本机单调时钟）；历史从最旧到当前时刻，间隔 1/history_hz。服务端不将这个值与另一台机器的时钟相减。

context 支持 command_pose、command_joint、previous_command_pose、start_pose、start_joint、object_pose、object_joint、base_to_parent、pose_history、joint_history、image_history。位姿、关节参考分别为20/16维，坐标系变换为有效4×4刚体矩阵。历史形状为 (L,20)/(L,16)，相机历史为按相机名组织的原始 RGB uint8 (L,H,W,3)，尺寸遵循 image_sizes；当前 images 可使用 JPEG。context 的必需字段由 observation_spec 声明。当前训练配置若相对指令，只声明 command_pose / command_joint；其余键仍可在握手中选用。previous_command_pose 对应 observation_time_ns - 1/history_hz 的命令，而不是上一拍机器人控制发布值。

共享 G2 policy 选择 `parent_frame: arm_base_link` 时声明 `base_to_parent`，其方向为当前观测时刻的 `T_armbase←base`。客户端从实测腰部关节计算；整段历史、动作 chunk 和 RTC 前缀共用本次请求的变换。模型输出还原到 base_link 后返回，wire 上的 state、command_pose 和 pose_target 始终使用 base_link。

## 响应与 RTC

响应 `type=action`、相同 request_id、有限数组 actions(T,D)，1≤T≤H，可携带 timing/rtc_info。调用 `build_response(..., metadata=meta)`、`parse_response(..., metadata=meta)` 校验 D。`extensions.response_mode: full_chunk` 进一步要求 T=H；`variable_chunk` 允许范围内的可变长度，未知模式在握手时拒绝。

RTC prefix 也必须使用相同控制量和 D，delay 不得超过 prefix 或 H；prefix 允许短于 H，不受完整响应长度约束。仅在 `rtc_supported=true` 时可以启用 RTC。支持 RTC 不意味着客户端负责某种特定模型的 guidance 算法。

## Episode reset

显式 episode 边界：客户端发 `{"type":"reset","request_id":N}`，server 清理该会话的历史、参考和模型状态后回 `reset_ack`。只有握手声明 reset_supported 时可调用。同一连接可以承载多个 episode；换连接也创建独立 session。reset 与机器人归位命令无关。

## 预测与执行时间

`timing.action_dt=1/policy_hz`，例如 30 Hz 时相邻目标间隔约 33.3 ms。`action_horizon=50` 表示返回 50 个采样点，首末目标时间差为 49/30 秒；不要求 GPU 每次推理只耗时 33.3 ms。

预测长度 H、返回长度 R 和客户端执行前缀 K 的职责，以及重规划行为，见[预测块与执行预算](development.md#预测块与执行预算)。

## 图像

- 相机名称为 `head_color`、`hand_left_color`、`hand_right_color`。仅 `metadata.required_cameras` 必须存在；不需要的相机可省略或为 null。
- 原始图像为 RGB uint8 H×W×3；JPEG 为 bytes。
- `image_sizes` 按相机声明 `[H,W]`，`image_resize` 为 `resize` 或 `letterbox`。
- 编码、缩放和解压由适配层负责。JPEG 解码后 server 可调用 `validate_images(..., decoded=True)` 再检查实际尺寸。协议包不解压图像。

## 编解码

两种编码使用相同的协议 API，通过 `codec.dumps(message, encoding=...)` 和 `codec.loads(frame, encoding=...)` 切换；`encoding` 默认为 `"msgpack"`。底层 `packb/unpackb/json_dumps/json_loads` 也保留。

普通 JSON 数值列表可以直接表示 state/action；NumPy 数组与 bytes 的跨语言表示如下：

```json
{
  "state": {"$g2":"ndarray","dtype":"<f4","shape":[2],"data":[1.0,2.0]},
  "image": {"$g2":"bytes","encoding":"base64","data":"/9j/2Q=="}
}
```

这是类型表示示例，两个数不是合法 pose20，图片仅占位。数组 data 为与 shape 对应的嵌套列表；NumPy 标量用 `{"$g2":"scalar","dtype":"<f4","data":1.0}`。`$g2` 为保留键；JSON 拒绝非有限数字、错误 shape/Base64 和超范围整数。MessagePack 保持 OpenPI 兼容的 NumPy 标记，不使用 pickle。

## 协议 API

```python
from g2_policy_protocol import codec, protocol as p
```

| 阶段 | API | 调用方责任 |
|---|---|---|
| 握手 | `build_metadata` / `validate_metadata` | 直接构造 v3；客户端保留协商后的 metadata |
| 观测 | `build_request` / `parse_request` | 传入同一份 metadata，以及声明需要的图像、状态、context 和观测时间 |
| 动作 | `build_response` / `parse_response` | 传入 metadata；响应使用原请求 ID，客户端核对预期 ID |
| 错误 | `build_error` | 有效 ID 原样返回，否则用 `0`；客户端不执行错误响应 |
| 传输 | `codec.dumps` / `codec.loads` | 编码与 URL 一致；调用方负责实际收发 |

`parse_response` 收到 `error` 时抛出 `RuntimeError`，非法响应抛出 `ValueError`。网络边界处理解码、校验、模型错误与断连；动作可达性和运动约束由机器人执行层负责。

## 本地编解码示例

下面的示例可在安装本包后直接运行，覆盖两种编码的握手、观测和动作往返。图像和动作均为模拟数据；示例不建立 WebSocket 连接，也不执行机器人动作。

```python
import numpy as np
from g2_policy_protocol import codec, protocol as p

# 模拟双臂单位旋转位姿，仅用于本地协议验证。
state = np.zeros(20, dtype=np.float32)
state[[3, 7, 13, 17]] = 1.0

for encoding in ("msgpack", "json"):
    # server 声明能力，client 接收并校验。
    metadata = p.build_metadata(
        policy_hz=30, action_horizon=2,
        required_cameras=["head_color"],
        image_sizes={"head_color": [224, 224]},
    )
    client_metadata = codec.loads(codec.dumps(metadata, encoding=encoding), encoding=encoding)
    p.validate_metadata(client_metadata, policy_hz=30)

    # client 发送观测，server 解析。
    request = p.build_request(
        request_id=42, prompt="拿起饮料", state=state,
        images={"head_color": np.zeros((224, 224, 3), np.uint8)},
        metadata=client_metadata,
    )
    observation = p.parse_request(
        codec.loads(codec.dumps(request, encoding=encoding), encoding=encoding),
        metadata=metadata,
    )

    # 模拟模型返回两帧保持当前位姿的动作。
    response = p.build_response(
        request_id=observation["request_id"],
        actions=np.repeat(observation["state"][None, :], 2, axis=0),
        metadata=metadata,
    )
    actions = p.parse_response(
        codec.loads(codec.dumps(response, encoding=encoding), encoding=encoding),
        request_id=42,
        metadata=client_metadata,
    )
    assert actions.shape == (2, 20)
    np.testing.assert_array_equal(actions[0], state)
    print(f"{encoding}: round trip OK")
```

Schema 描述静态结构，Python 校验动态 shape、有限值、相机/RTC 能力、频率和 request_id。动作限速、可达性和关节规划属于 G2 控制层。

## 定义来源

结构源为 [`g2.schema.json`](../src/g2_policy_protocol/g2.schema.json)，其中 `x-g2` 记录固定向量布局。运行时由 `protocol.py` 检查数组形状、有限值与协商后的动态需求；生成和验证步骤见[开发维护](development.md#修改与验证)。
