# For 域控 / Client

[返回 README](../README.md) · [GPU Server](server.md) · [v2 协议](protocol.md) · [v3 协议](protocol-v3.md)

本文面向运行 wholebody 的域控。client 负责采集实测状态、按 server 声明准备图像、发送观测，并在响应通过校验后交给执行层。

无需机器人硬件的网络联调可使用[最小 WebSocket Server 和配套客户端](server.md#可直接启动的最小-websocket-server)。

## 安装

在 wholebody 使用的 Python 环境中安装协议包，例如：

```bash
source /home/agi/.venv/bin/activate
python -m pip install /path/to/g2-policy-protocol
```

跨机部署也可以安装分发的 wheel：

```bash
python -m pip install /path/to/g2_policy_protocol-0.5.0-py3-none-any.whl
```

源码安装以上述路径为准；需要可编辑安装时使用 `python -m pip install -e /path/to/g2-policy-protocol`。域控只需安装协议包及 wholebody 自身的运行依赖，无需为通信安装 Torch、JAX 或 LeRobot。

**GPU server 必须同时使用 0.5.0 和兼容的服务代码。** 安装或升级后先重启 GPU server，再启动 client。0.4.1 只能校验 `g2.policy.v2`；接到 v3 metadata 时会报 `unknown=['action_spec', 'observation_spec', 'reset_supported', 'timing']`。

## 在 wholebody 中使用

wholebody 的三个启动入口共用 `policy.runtime.transport.G2WebSocketPolicyClient`。现有客户端已经封装连接、图像准备、消息编解码和请求 ID 管理；`parse_response` 会传入握手 metadata，以便按 v2 / v3 校验动作宽度。

推理 YAML 的 `wire_encoding` 或 CLI `--wire-encoding json` 选择编码，默认是 MessagePack。连接地址可以使用 `ws://<gpu-host>:8000`；实际主机和端口以 GPU server 配置为准。JSON 连接使用 `?encoding=json`，wholebody 客户端会根据编码配置设置连接参数。

一次连接和推理按以下顺序进行：

1. 连接 server，接收首条 `metadata`。
2. 校验协议与服务能力；比较本地配置的 `policy_hz`，需要 RTC 时确认 server 支持。v2 / v3 由 `metadata.protocol` 决定，不要按模型名称分支。
3. 按握手采集观测：v2 始终发送实测 pose20；v3 只发送 `observation_spec` 要求的 `state` / `joint_state` / `prompt` / `context`。图像按声明的编码、尺寸及 resize 方式准备。
4. 发送带唯一 `request_id` 的 `obs`，接收 `action` 或 `error`。v3 且 `reset_supported` 时，episode 边界可另发 `reset`。
5. 用同一份 metadata 校验响应 ID、动作维度和有限值，通过后交给控制层执行。

`metadata.arm` 决定执行臂。v2 线上观测和动作始终是完整 pose20，单臂由 server 裁切/扩展。v3 动作宽度是 `action_spec.dim`，控制量见 [v3 协议](protocol-v3.md)。固定坐标系、单位和字段顺序：v2 见 [协议参考](protocol.md#状态与动作)，v3 见 `action_spec`。

当前 OpenPI G2 server 握手为 `g2.policy.v3`、`control_quantity=pose_target`（仍是 20 维绝对位姿）。域控仍按 `arm` 只向所选侧发令。

## 自定义 client 时使用的 API

下面的函数展示协议边界，可用于自己的 WebSocket 客户端。调用方负责收发 frame、传入递增的请求 ID，并提前按 metadata 准备图像。`encoding` 必须与连接参数一致。v3 解析动作时 **必须** 传入握手 metadata，否则会按 v2 的 `(T,20)` 检查。

```python
from g2_policy_protocol import codec, protocol as p


def accept_metadata(frame, *, encoding="msgpack", policy_hz=30, require_rtc=False):
    metadata = codec.loads(frame, encoding=encoding)
    p.validate_metadata(metadata, policy_hz=policy_hz, require_rtc=require_rtc)
    return metadata


def encode_observation(*, request_id, prompt, state, images, metadata,
                       joint_state=None, encoding="msgpack"):
    request = p.build_request(
        request_id=request_id, prompt=prompt, state=state,
        images=images, metadata=metadata, joint_state=joint_state,
    )
    return codec.dumps(request, encoding=encoding)


def accept_action(frame, *, request_id, metadata, encoding="msgpack"):
    payload = codec.loads(frame, encoding=encoding)
    return p.parse_response(payload, request_id=request_id, metadata=metadata)
```

RTC 接入还需向 `build_request` 传入 `rtc_enabled`、`inference_delay` 和可选的 `prev_chunk_left_over`，含义见 [v2 RTC](protocol.md#rtc) 与 [v3](protocol-v3.md)。v3 的 prefix 宽度必须与 `action_spec.dim` 一致。

## 校验与错误处理

- `validate_metadata` 在连接时检查协议和能力；`protocol == g2.policy.v3` 时走 v3 字段与 `action_spec` 布局校验。client 不依赖模型名称判断能否执行。
- `build_request` 按握手检查观测：v2 要求实测 pose20、必需相机和可选关节状态；v3 还检查 `observation_spec` 与 `context`。
- `parse_response(..., metadata=metadata)` 检查对应的 `request_id` 和有限非空 `(T, D)` 动作（v2 的 D=20，v3 的 D=`action_spec.dim`）；收到 `error` 时抛出 `RuntimeError`，非法响应抛出 `ValueError`。
- 校验失败或收到服务端错误时，不将该响应作为动作执行。wholebody 的 `G2WebSocketPolicyClient` 在响应解析抛出上述异常时会关闭连接并向上抛出异常。

协议校验不检查机器人动作的可达性或运动约束。动作限速、可达性和关节规划属于 G2 控制层。v3 的 `control_quantity` 还必须有已绑定的执行器，否则域控在打开硬件前拒绝。
