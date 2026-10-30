# 通信协议

[文档入口](../README.md#文档导航) · [安装与接入](guide.md) · [policy 配置](policy-config.md)

本页定义域控 client 与 GPU server 之间的线上契约 `g2.policy.v3`。握手阶段拒绝其他协议。包版本与升级要求见[安装与版本](guide.md#安装与版本)。

## 术语

| 术语 | 含义 |
|---|---|
| client / server | client 是域控上的 wholebody；server 是 GPU 上的模型服务（OpenPI、LeRobot 或第三方） |
| metadata / 握手 | 连接建立后 server 发出的第一条消息，声明模型能力；client 校验后才发送观测 |
| pose20 | 双臂末端位姿向量：每臂 xyz(3) + rot6d(6) + 夹爪(1)，左臂在前，共 20 维 |
| joint16 | 双臂关节向量：每臂 7 个关节 + 夹爪，左臂在前，共 16 维 |
| rot6d | 旋转矩阵前两列按列拼接：`R[:, :2].reshape(-1, order="F")`；单位旋转为 `[1,0,0,0,1,0]` |
| close_01 / close_ratio | 夹爪绝对闭合比例，0 开、1 闭；任何控制量下都不作差分 |
| base_link / flange | 线上位姿的父坐标系和末端点：机器人基座坐标系下的法兰位姿 |
| control_quantity | server 返回的动作含义：`pose_target`、`joint_position`、`joint_velocity`、`joint_torque` |
| H | `action_horizon`，模型一次预测的动作步数 |
| R | 一次响应实际返回的步数，1 ≤ R ≤ H |
| K | client 从一次响应中提交执行的步数，由 wholebody 的 `execution_steps` 决定 |
| chunk | 一次推理返回的一段动作序列，形状 `(R, D)`，D=`action_spec.dim` |
| context | 观测之外、模型可能需要的参考量，例如指令位姿、历史、坐标变换 |
| RTC | Real-Time Chunking：推理期间机器人继续执行旧 chunk，新 chunk 的开头需与已执行部分衔接。见[RTC](#rtc) |
| episode / reset | 一次任务尝试；`reset` 消息清理 server 端该会话的状态，与机器人归位无关 |

## 连接与消息

WebSocket 默认使用二进制 MessagePack；URL 带 `?encoding=json` 时整条连接使用 JSON text frames。编码由连接参数确定，不在 metadata 中重复声明；整条连接不混发帧类型。

```text
client                                server
  │ ◀──────────── metadata ──────────── │   连接后 server 主动发送
  │ ─── obs(request_id=1) ────────────▶ │
  │ ◀── action(1) 或 error(1) ───────── │
  │ ─── obs(request_id=2) ────────────▶ │
  │ ◀── action(2) ───────────────────── │
  │ ─── reset(request_id=3) ──────────▶ │   仅 reset_supported=true
  │ ◀── reset_ack(3) ────────────────── │
```

| type | 方向 | 必含字段 | 构造 / 解析 |
|---|---|---|---|
| `metadata` | server → client | 见[握手与能力](#握手与能力) | `build_metadata` / `validate_metadata` |
| `obs` | client → server | request_id、images，以及握手要求的 prompt/state/joint_state/context | `build_request` / `parse_request` |
| `action` | server → client | request_id、actions `(R, D)` | `build_response` / `parse_response` |
| `error` | server → client | request_id、code、message | `build_error`；`parse_response` 收到时抛 `RuntimeError` |
| `reset` | client → server | request_id | `build_reset` |
| `reset_ack` | server → client | request_id | `build_reset_ack` |

error 的 code 为 `invalid_request`（请求无法解析或不满足握手）或 `inference_error`（模型或响应构造失败）。请求 ID 无效时 server 使用 `0`。

## 握手与能力

metadata 字段：

| 字段 | 含义 |
|---|---|
| `protocol` | 固定 `g2.policy.v3` |
| `policy_hz` | 动作时间轴频率；不是 GPU 推理频率 |
| `action_horizon` | H |
| `arm` | `left` / `right` / `both`，实际执行侧 |
| `required_cameras` | 必须携带的相机；可以为空或只选腕部 |
| `image_encoding` / `image_resize` / `image_sizes` | 图像传输格式（raw / jpeg）、缩放方式（resize / letterbox）、按相机的 `[H,W]` |
| `rtc_supported` | server 是否处理 RTC 字段 |
| `requires_joint_state` | 必须与 `observation_spec.joint` 一致 |
| `action_spec` | control_quantity、dim、fields、units、frame、ee_frame、representation |
| `observation_spec` | pose/joint/prompt 是否必需、history_steps、history_hz、必需 context_keys |
| `timing.action_dt` | 相邻预测动作间隔，必须等于 1/policy_hz |
| `timing.first_target_offset_s` | 首个目标相对本次观测时刻的偏移 |
| `reset_supported` | 是否实现当前会话的 episode reset |
| `extensions` | 可选；普通键为诊断信息，`response_mode` 受校验 |

`prompt` 仅在 `observation_spec.prompt=true` 时要求非空。`state`（pose20）和 `joint_state`（joint16）只在声明需要时强制携带。

## 消息样例

以下为 `build_*` 实际生成的 JSON 编码（数组用 `$g2` 表示，见[编解码](#编解码)），对应右臂、只用头部相机、H=2 的服务。

metadata：

```json
{
  "type": "metadata",
  "protocol": "g2.policy.v3",
  "policy_hz": 30.0,
  "action_horizon": 2,
  "arm": "right",
  "image_encoding": "raw",
  "image_resize": "resize",
  "image_sizes": {"head_color": [224, 224]},
  "required_cameras": ["head_color"],
  "rtc_supported": false,
  "requires_joint_state": false,
  "action_spec": {
    "control_quantity": "pose_target",
    "dim": 20,
    "fields": ["left_x", "left_y", "left_z",
               "left_rot6d_0", "left_rot6d_1", "left_rot6d_2", "left_rot6d_3", "left_rot6d_4", "left_rot6d_5",
               "left_gripper",
               "right_x", "right_y", "right_z",
               "right_rot6d_0", "right_rot6d_1", "right_rot6d_2", "right_rot6d_3", "right_rot6d_4", "right_rot6d_5",
               "right_gripper"],
    "units": ["m", "m", "m", "dimensionless", "dimensionless", "dimensionless",
              "dimensionless", "dimensionless", "dimensionless", "close_ratio",
              "m", "m", "m", "dimensionless", "dimensionless", "dimensionless",
              "dimensionless", "dimensionless", "dimensionless", "close_ratio"],
    "frame": "base_link",
    "ee_frame": "flange",
    "representation": "absolute"
  },
  "observation_spec": {
    "pose": true, "joint": false, "prompt": true,
    "history_steps": 1, "history_hz": 30.0, "context_keys": []
  },
  "timing": {"action_dt": 0.03333333333333333, "first_target_offset_s": 0.0},
  "reset_supported": false,
  "extensions": {"response_mode": "full_chunk"}
}
```

obs（图像缩成 1×1 仅为示意，实际为 224×224×3）：

```json
{
  "type": "obs", "request_id": 1, "prompt": "拿起饮料",
  "state": {"$g2": "ndarray", "dtype": "<f4", "shape": [20],
            "data": [0,0,0, 1,0,0,0,1,0, 0, 0,0,0, 1,0,0,0,1,0, 0]},
  "images": {"head_color": {"$g2": "ndarray", "dtype": "|u1", "shape": [1,1,3], "data": [[[0,0,0]]]}},
  "rtc_enabled": false, "inference_delay": 0, "prev_chunk_left_over": null,
  "observation_time_ns": 123, "context": {}
}
```

action（R=2，保持当前位姿）：

```json
{
  "type": "action", "request_id": 1,
  "actions": {"$g2": "ndarray", "dtype": "<f4", "shape": [2, 20],
              "data": [[0,0,0, 1,0,0,0,1,0, 0, 0,0,0, 1,0,0,0,1,0, 0],
                       [0,0,0, 1,0,0,0,1,0, 0, 0,0,0, 1,0,0,0,1,0, 0]]}
}
```

error / reset / reset_ack：

```json
{"type": "error", "request_id": 1, "code": "inference_error", "message": "model failed"}
{"type": "reset", "request_id": 7}
{"type": "reset_ack", "request_id": 7}
```

## 动作控制量

控制量固定为规范化的完整双臂布局，左右顺序，每臂最后一维都是绝对夹爪 close_ratio（0 开、1 闭）。使用 `protocol.build_action_spec(quantity)` 构造，校验拒绝错配的维度、字段、单位和坐标系：

| control_quantity | D | 每臂分量与单位 |
| --- | --- | --- |
| pose_target | 20 | xyz 米、rot6d、gripper |
| joint_position | 16 | 7 关节 rad、gripper |
| joint_velocity | 16 | 7 关节 rad/s、gripper |
| joint_torque | 16 | 7 关节 N*m、gripper |

字段顺序：

- pose20：`left_x, left_y, left_z, left_rot6d_0…5, left_gripper`，然后右臂同样 10 个。
- joint16：`left_joint_1…7, left_gripper`，然后 `right_joint_1…7, right_gripper`。

位姿使用 base_link 下的 flange；TCP、局部参考、delta 或旋转表示由模型适配层转换。`arm` 限定实际执行侧，另一侧位置保持实测值，另一侧速度/力矩为零。

## 观测、历史与参考

请求为 `type=obs`、request_id、可选 prompt/state/joint_state、images、context、observation_time_ns 与可选 RTC 字段。历史 L>1 时必须给 observation_time_ns（客户端本机单调时钟）；历史从最旧到当前时刻，间隔 1/history_hz。服务端不将这个值与另一台机器的时钟相减。

context 可用的键：

| 键 | 形状 | 含义 |
|---|---|---|
| `command_pose` / `command_joint` | (20,) / (16,) | 当前指令参考 |
| `previous_command_pose` | (20,) | observation_time_ns − 1/history_hz 时刻的指令，不是上一拍控制发布值 |
| `start_pose` / `start_joint` | (20,) / (16,) | episode 起点 |
| `object_pose` / `object_joint` | (20,) / (16,) | 物体等额外参考 |
| `base_to_parent` | 4×4 刚体矩阵 | 见下文 |
| `pose_history` / `joint_history` | (L,20) / (L,16) | 状态历史，最旧在前 |
| `image_history` | 按相机名：(L,H,W,3) RGB uint8 | 尺寸遵循 image_sizes；当前帧 images 可用 JPEG，历史只用 raw |

必需的键由 `observation_spec.context_keys` 声明。当前训练配置若相对指令，只声明 command_pose / command_joint。

policy 选择 `parent_frame: arm_base_link` 时声明 `base_to_parent`，方向为当前观测时刻的 `T_armbase←base`。客户端从实测腰部关节计算；整段历史、动作 chunk 和 RTC 前缀共用本次请求的变换。模型输出还原到 base_link 后返回，线上的 state、command_pose 和 pose_target 始终使用 base_link。

## 响应

响应 `type=action`、相同 request_id、有限数组 actions(R,D)，1≤R≤H，可携带 timing/rtc_info。调用 `build_response(..., metadata=meta)`、`parse_response(..., metadata=meta)` 校验 D。`extensions.response_mode: full_chunk` 进一步要求 R=H；`variable_chunk` 允许范围内的可变长度，未知模式在握手时拒绝。

## RTC

RTC（Real-Time Chunking）解决的问题：server 推理需要时间，这段时间里机器人仍在执行上一个 chunk。新 chunk 的开头必须与"推理期间已经执行、无法更改"的那几步衔接，否则会在 chunk 切换处跳变。

只有 client 知道执行进度，因此分工如下：

| 层 | 负责 |
|---|---|
| wholebody（client） | 记录旧 chunk 剩余部分、估计推理期间会执行的步数，填入 obs |
| 本协议 | 声明能力、传输字段、校验格式 |
| 模型（server） | 用前缀约束新 chunk 的生成（例如 OpenPI 的 guided 采样）；算法不属于协议 |

obs 中的 RTC 字段：

| 字段 | 含义 |
|---|---|
| `rtc_enabled` | 本次请求是否使用 RTC；仅 `rtc_supported=true` 时可为 true |
| `prev_chunk_left_over` | 旧 chunk 尚未执行的部分，`(P, D)`，与动作同一控制量；P 可小于 H |
| `inference_delay` | 推理期间机器人将执行的步数；须 < H 且 ≤ P |

`rtc_enabled=false` 时另两个字段必须为默认值（0 / null）。

## Episode reset

显式 episode 边界：客户端发 `{"type":"reset","request_id":N}`，server 清理该会话的历史、参考和模型状态后回 `reset_ack`。只有握手声明 reset_supported 时可调用。同一连接可以承载多个 episode；换连接也创建独立 session。reset 与机器人归位命令无关。

## 预测与执行时间

`timing.action_dt=1/policy_hz`，例如 30 Hz 时相邻目标间隔约 33.3 ms。`action_horizon=50` 表示返回 50 个采样点，首末目标时间差为 49/30 秒；不要求 GPU 每次推理只耗时 33.3 ms。

H、R、K 的职责和重规划行为见[预测块与执行预算](development.md#预测块与执行预算)。

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
| reset | `build_reset` / `build_reset_ack` | 仅 reset_supported 时使用 |
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
