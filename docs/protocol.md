> 本文只描述保留兼容的 `g2.policy.v2`。v3 见 [protocol-v3.md](protocol-v3.md)。0.6.0 同一个发行包同时提供两版；client 按握手 `protocol` 分流。

# 协议参考（v2）

[返回 README](../README.md) · [域控 / Client](client.md) · [GPU Server](server.md) · [v3](protocol-v3.md)

`g2.policy.v2` 的结构定义是 [`g2.schema.json`](../src/g2_policy_protocol/g2.schema.json)，其中 `x-g2` 记录向量字段顺序、单位和固定语义。当前 OpenPI G2 server 已改发 `g2.policy.v3`，不要把本文当成线上唯一握手。

## 连接与消息

WebSocket 默认使用二进制 MessagePack；连接地址带 `?encoding=json` 时，整条连接使用 JSON text frames。编码只由连接参数确定，不在 metadata 中重复声明，不混发或自动猜测帧类型。

server 首先发送 metadata，client 完成能力校验后才发送观测。

| 消息 | 方向 | 内容 |
|---|---|---|
| `metadata` | server → client | protocol、Hz、horizon、执行臂、相机要求、图像编码/尺寸/resize、RTC、joint state 要求 |
| `obs` | client → server | request_id、prompt、实测 pose20、images；可选 joint_state、RTC prefix/delay、观测时间 |
| `action` | server → client | 原样回传 request_id、绝对 `(T,20)` 动作、可选计时和 RTC 信息 |
| `error` | server → client | 原样回传有效 request_id、code、message；client 不执行该响应 |

metadata 只声明服务端可变的能力，client 不检查模型名称。pose20、base_link、米/rot6d/绝对动作由 `g2.policy.v2` 固定。模型名称等诊断信息可以放在 `extensions`，不影响执行。

## 状态与动作

- `state` / `actions` 的每一帧：左臂 xyz(3)、rot6d(6)、close_01(1)，再右臂，共 20 维。
- `state` 必须是实测 pose20；`actions` 为有限非空 `(T,20)` 数组。两者均为 `base_link` 下绝对位姿，位置单位为米。
- rot6d：`R[:, :2].reshape(-1, order="F")`；夹爪 `0=open`、`1=closed`。
- `joint_state`：可选 16 维，左右各 7 个弧度关节角及夹爪；用于关节 checkpoint，不替代实测 pose20。
- 单臂模型仍接收完整 pose20，返回完整绝对动作；server 负责裁切/扩展，G2 执行端只向 `metadata.arm` 指定的臂发令。

## 图像

- 相机名称为 `head_color`、`hand_left_color`、`hand_right_color`。仅 `metadata.required_cameras` 必须存在；不需要的相机可省略或为 null。
- 原始图像为 RGB uint8 H×W×3；JPEG 为 bytes。
- `image_sizes` 按相机声明 `[H,W]`，`image_resize` 为 `resize` 或 `letterbox`。
- 编码、缩放和解压由适配层负责。JPEG 解码后 server 可调用 `validate_images(..., decoded=True)` 再检查实际尺寸。协议包不解压图像。

## RTC

- 仅当 `metadata.rtc_supported=true` 时，client 才能设置 `rtc_enabled=true`。
- `prev_chunk_left_over` 是有限非空 `(T,20)` 绝对动作 prefix。
- `inference_delay` 是非负整数，单位为策略步数。
- 未启用 RTC 时，不能携带 prefix 或非零 delay。

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

## 本地回环示例

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

## 版本边界

包版本 `0.6.0` 同时提供 `g2.policy.v2` 和 `g2.policy.v3`。v2 删除握手中的 `schema_version / state_layout / action_layout / frame / rotation / action_semantics / encoding / policy_type`，不兼容 `g2.policy.v1` 和更早的两套 G2 消息。v3 在握手中新增 `action_spec`、`observation_spec`、`timing`、`reset_supported`；0.4.1 会把这些字段判为 unknown。

域控与 GPU 两端必须同步安装 0.6.0、更新代码并重启。v3 服务不能搭配 0.4.1 client。协议升级步骤见[开发与发布](development.md)。
