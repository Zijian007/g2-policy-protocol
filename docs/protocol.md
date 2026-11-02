# 通信协议 g2.policy.v3

[README](../README.md) · [安装](install.md) · [policy 配置](policy-config.md)

域控上的 wholebody 是 **client**，GPU 上的模型服务（OpenPI、LeRobot 或第三方）是 **server**。两者通过 WebSocket 交换三种核心消息：

```text
client（域控）                           server（GPU）
   │ ◀──────── ① metadata ────────────── │  连接后 server 先声明"我需要什么、返回什么"
   │ ───────── ② obs ──────────────────▶ │  client 按声明发送观测
   │ ◀──────── ③ action（或 error）────── │  server 返回一段动作
   │            ②③ 循环 ...              │
```

本页先讲跑通这个循环需要知道的内容，再讲按需开启的[可选能力](#可选能力)。

## 一次完整交互

server 用 `build_metadata` 声明能力，两端所有 `build_*` / `parse_*` 都传入这同一份 metadata：

```python
from g2_policy_protocol import protocol as p

# ① server：右臂模型，用头部和右腕相机，一次预测 50 步，30 Hz
metadata = p.build_metadata(
    policy_hz=30, action_horizon=50, arm="right",
    required_cameras=["head_color", "hand_right_color"],
    image_sizes={"head_color": [224, 224], "hand_right_color": [224, 224]},
)

# ② client：发送当前观测
obs = p.build_request(
    request_id=1, prompt="拿起饮料", metadata=metadata,
    state=pose20,                                    # 当前双臂位姿，见下文"数据约定"
    images={"head_color": head_rgb, "hand_right_color": right_rgb},   # 224×224×3 uint8
)

# ③ server：返回 (步数, 20) 的绝对位姿目标；client 解析并校验
response = p.build_response(request_id=1, actions=actions, metadata=metadata)
actions = p.parse_response(response, request_id=1, metadata=metadata)
```

实际线上消息就是这几个 dict，经 MessagePack（默认）或 JSON 编码后发送：

```python
{"type": "obs", "request_id": 1, "prompt": "拿起饮料", "state": <20 个 float32>, "images": {...}, ...}
{"type": "action", "request_id": 1, "actions": <50×20 float32>}
{"type": "error", "request_id": 1, "code": "inference_error", "message": "..."}
```

- `request_id` 由 client 递增，server 原样返回；client 只接受 ID 对得上的响应。
- server 出错时回 `error`：请求本身不合法为 `invalid_request`，模型推理失败为 `inference_error`。client 收到 `error` 时 `parse_response` 抛 `RuntimeError`，不执行任何动作。
- 任何字段不合法都会抛 `ValueError`，非法消息不会到达模型或机器人。

## 数据约定

**位姿向量 pose20**：左右臂各 10 维，左臂在前。位置单位为米，坐标系为机器人基座 `base_link` 下的法兰 `flange`。

```text
索引   0  1  2 │ 3  4  5  6  7  8 │  9    │ 10 11 12 │ 13 … 18 │ 19
       x  y  z │ rot6d_0 … rot6d_5 │ 夹爪  │  x  y  z │ rot6d   │ 夹爪
       └──────────── 左臂 ──────────────┘ └────────── 右臂 ─────────┘
```

- **rot6d**：旋转矩阵 R 的前两列按列拼接，即 `R[:, :2].reshape(-1, order="F")`。单位旋转为 `[1, 0, 0, 0, 1, 0]`。
- **夹爪**：绝对闭合比例，0 为全开，1 为全闭。
- **关节向量 joint16**：每臂 7 个关节角（rad）加夹爪，左臂在前。

**动作**：server 返回的永远是上述格式的**绝对目标**，形状为 `(步数, 20)`，步数为 1 到 `action_horizon`。

- 单臂模型（`arm="right"`）也要返回完整 20 维，未执行一侧填当前实测值。
- 模型内部用的增量、其他旋转表示、补齐维度等，都由 server 适配层还原后再返回。
- 动作的时间间隔为 `1 / policy_hz`。`policy_hz` 是动作时间轴的频率，不是 GPU 每秒推理次数。
- 限速、可达性和关节规划由 wholebody 控制层负责，协议不检查。

**图像**：RGB uint8，形状 H×W×3，相机名为 `head_color`、`hand_left_color`、`hand_right_color`。client 按 metadata 声明的尺寸缩放，按声明的方式编码，server 收到后再解码。

当前帧和 `context.image_history` 都跟随 `image_encoding`：`raw` 历史是 `uint8 (L,H,W,3)` 数组，`jpeg` 历史是长度为 L 的非空 JPEG `bytes` 列表，从旧到新，最后一帧为当前帧。协议核心只校验 JPEG 的表示；backend 解码并检查尺寸，非法图像返回 `invalid_request`。

有完整 `context.image_history` 时可省略 `images`，backend 直接使用历史（包含当前帧），请求解析也不补建 `images`。无图像历史时仍须发送 `images`（不需要相机时发 `{}`）。两者都发送的旧请求仍会校验并接受；第一方客户端只发送历史，避免重复缩放、编码、打包和传输当前帧。

## 握手 metadata

`build_metadata` 的参数里，**通常只需要设置下面这些**，其余字段会自动推导出一致的值：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `policy_hz` | 必填 | 动作频率，须与训练数据一致 |
| `action_horizon` | 50 | 一次最多返回的动作步数 |
| `arm` | `both` | 实际执行哪一侧：`both` / `left` / `right` |
| `required_cameras` | 按 `arm` 推导 | 必须发送的相机；按模型真实输入声明，可以为空 |
| `image_sizes` | 不缩放 | 每个相机的 `[高, 宽]` |
| `image_resize` | `resize` | `resize` 拉伸，或 `letterbox` 保持比例并补黑边；须与训练预处理一致 |
| `image_encoding` | `raw` | `raw` 发原始像素，`jpeg` 发压缩字节（体积小，server 需解码） |

开启可选能力时还会用到 `requires_joint_state`、`rtc_supported`、`reset_supported`、`observation_spec`、`action_spec`、`extensions`，见下一节。

metadata 里还有一些字段由上述参数自动推导，并在握手时校验一致性，一般不需要手写：`protocol`（固定 `g2.policy.v3`）、`timing.action_dt`（等于 `1/policy_hz`）、`action_spec` 的字段名、单位和坐标系。

## 可选能力

每项能力都由 server 在 metadata 中声明，client 按声明配合。不声明就不会用到。

### 关节状态与关节控制

- **模型需要关节角作为输入**：设 `requires_joint_state=True`，client 会在 obs 中加入 `joint_state`（joint16）。
- **模型输出关节角**：设 `action_spec=p.build_action_spec("joint_position")`，动作变为 `(步数, 16)` 的 joint16。默认的 `pose_target` 输出 pose20。

wholebody 只执行已绑定执行器的控制量，否则会在打开硬件前拒绝。

### 历史与参考量（context）

模型需要过去几帧或额外参考时，在 `observation_spec` 中声明 `history_steps`、`history_hz` 和 `context_keys`，client 会在 obs 的 `context` 字段中只发送声明过的键。

| context 键 | 形状 | 含义 |
|---|---|---|
| `pose_history` / `joint_history` | (帧数, 20) / (帧数, 16) | 状态历史，从旧到新，间隔 `1/history_hz` |
| `image_history` | 每个必需相机的历史 | 编码随 `image_encoding`：raw 为 uint8 (L,H,W,3)，jpeg 为 L 个 bytes 的列表 |
| `command_pose` / `command_joint` | (20,) / (16,) | 当前下发的指令 |
| `previous_command_pose` | (20,) | 上一个历史采样时刻的指令 |
| `start_pose` / `start_joint` | (20,) / (16,) | 本次 episode 起点 |
| `object_pose` / `object_joint` | (20,) / (16,) | 物体等额外参考 |
| `base_to_parent` | 4×4 矩阵 | 坐标系为 `arm_base_link` 时使用：`T_armbase←base`，把 base_link 下的位姿换算到臂基座坐标系，即当前时刻臂基座位姿的逆 |

`history_steps > 1` 时，obs 必须带 `observation_time_ns`（client 本机单调时钟）。server 不会把它和自己的时钟比较。

这些声明通常不手写：训练配置经 `data.policy.policy_capabilities` 自动生成，见 [policy 配置](policy-config.md)。

### RTC（推理期间的动作衔接）

推理需要时间，这段时间里机器人还在执行上一段动作。RTC 让新动作的开头与已执行的部分衔接，避免切换时跳变。

server 设 `rtc_supported=True` 后，client 可以在 obs 中附加三个字段：

| obs 字段 | 含义 |
|---|---|
| `rtc_enabled` | 本次请求是否使用 RTC |
| `prev_chunk_left_over` | 上一段动作中尚未执行的部分，形状 `(步数, 20)` |
| `inference_delay` | 预计推理期间会执行掉的步数，须小于 `action_horizon`，且不超过上一项的步数 |

如何利用这些信息约束新动作，属于模型算法（例如 OpenPI 的引导采样），不在协议范围内。

### Episode reset

server 设 `reset_supported=True` 后，client 可以在新任务开始时发 `{"type": "reset", "request_id": N}`，server 清理该连接的历史和模型状态后回 `reset_ack`。它与机器人归位无关。同一连接可以跑多个 episode，换连接就是新会话。

### 响应长度

`extensions={"response_mode": "full_chunk"}` 表示每次都返回满 `action_horizon` 步，返回不足会被拒绝；`variable_chunk` 允许返回 1 到 `action_horizon` 之间任意步数。`extensions` 里的其他键只作诊断信息。

## 编解码

用 `codec.dumps(message, encoding=...)` 和 `codec.loads(frame, encoding=...)` 编解码。默认 MessagePack，URL 带 `?encoding=json` 时整条连接改用 JSON。

MessagePack 中的 NumPy 数组采用与 OpenPI 兼容的格式，不使用 pickle。JSON 中数组和字节的表示如下：

```json
{"state": {"$g2": "ndarray", "dtype": "<f4", "shape": [2], "data": [1.0, 2.0]},
 "image": {"$g2": "bytes", "encoding": "base64", "data": "/9j/2Q=="}}
```

`$g2` 是保留键。非有限数、shape 不符、Base64 错误都会被拒绝。

## 接入模型（server 侧）

`PolicyServer` 负责连接、编解码、校验、请求 ID 和错误响应，你只需要实现 backend 的 `infer`：

```python
from g2_policy_protocol.images import decode_image_history, decode_images
from g2_policy_protocol.server import PolicyServer

class Backend:
    def infer(self, request):                    # request 已通过校验
        history = request["context"].get("image_history")
        images = (decode_images(request["images"], metadata) if history is None
                  else decode_image_history(history, metadata))  # 历史包含当前帧
        actions = ...                            # 模型推理，并还原为 (步数, 20) 绝对位姿
        return {"actions": actions}
    # 可选：new_session() 为每条连接建独立状态；reset() 配合 reset_supported=True

PolicyServer(Backend(), metadata=metadata, host="0.0.0.0", port=8000).serve_forever()
```

- 所有连接的 `infer` / `reset` 在同一个模型线程上依次执行，网络收发不会被阻塞。
- 训练和部署共用 `data.encoding.ActionCodec` 来还原动作，用法见 [examples/model_adapter.py](../examples/model_adapter.py)。最小可运行服务见 [examples/minimal_server.py](../examples/minimal_server.py)。
- OpenPI 和 LeRobot 已有现成的 backend，启动方式见各自的部署文档。

## client 侧（wholebody）

wholebody 的 `G2WebSocketPolicyClient` 依次做这几件事：

1. 连接并接收 metadata，用 `validate_metadata` 校验。频率不符、需要 RTC 但 server 不支持、控制量没有对应执行器，都会在打开硬件前拒绝。
2. 按 metadata 采集观测、缩放和编码图像，发送 obs。
3. 用 `parse_response` 校验响应，通过后才交给控制层。校验失败或收到 error 时不执行动作，关闭连接并上报。

编码方式由推理 YAML 的 `wire_encoding` 选择。细节见 [wholebody policy 通信契约](../../wholebody/docs/dev/policy-protocol.md)。

## 验收工具

```bash
python examples/minimal_server.py --port 8000          # 启动模拟服务；加 --jpeg 测 JPEG 路径
g2-policy-check --url ws://<gpu-host>:8000 --report report.json
g2-policy-check --url ws://<gpu-host>:8000 --sample observation.npz --rtc --report rtc.json
```

`g2-policy-check` 用 MessagePack 和 JSON 各发两次合成观测，检查握手、响应 ID 和动作形状，并额外检查夹爪在 [0, 1] 内、rot6d 没有退化（`--protocol-only` 跳过这两项）。退出码 0 表示通过。

通过只说明协议接入正确，不代表模型质量、坐标系正确或满足实时性。要用真实数据测试，可以用 `diagnostics.samples.save_sample` 把一帧观测导出为 `.npz`。

## 常见错误

| 报错（节选） | 原因与处理 |
|---|---|
| 末尾带 `may run different g2-policy-protocol versions` | 两端协议不兼容。两端更新到最新，先重启 server，见[兼容性](install.md#兼容性如何保证) |
| `server policy_hz disagrees with configured policy_hz` | client 配置的频率与模型不同。改 client 配置，或设为 null 接受 server 的声明 |
| `server does not support RTC` | client 开了 RTC，server 没有声明支持 |
| `missing required cameras` / `size disagrees with metadata image_sizes` | client 发的相机名或图像尺寸与 metadata 不符 |
| `server requires a nonempty prompt` / `history requires observation_time_ns` | 缺少 metadata 要求的观测字段 |
| `response request_id does not match observation` | server 没有原样返回 ID，或 client 读到了旧响应 |
| `full_chunk response must contain action_horizon actions` / `actions exceeds action_horizon` | 返回的步数与声明不符 |
| `actions must be finite nonempty (T, D)` | 模型输出没有还原成完整双臂维度（单臂没补齐、补齐维度没去掉），或含 NaN / Inf |
| `requires_joint_state disagrees ...` / `action_dt disagrees ...` / `action_spec must use the canonical layout` | 手写的 metadata 前后不一致。改用 `build_metadata` / `build_action_spec` 生成 |

协议校验都通过、但机器人动作异常时，问题一般不在协议层。先核对坐标系（`base_link` / `flange`）、rot6d 的列顺序、夹爪 0 开 1 闭、单位，再核对训练和部署的图像预处理是否一致。
