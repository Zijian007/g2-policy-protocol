# For GPU Server

[返回 README](../README.md) · [域控 / Client](client.md) · [协议参考](protocol.md)

本文面向开发和部署 policy server 的人员。只需本仓库和模型运行环境，即使无法访问或修改域控，也可以独立完成服务实现和协议验收。

server 负责声明模型能力、接收统一观测、适配模型输入，并将推理结果转换为绝对 pose20 动作返回。

阅读顺序：[安装](#安装) → [启动最小服务](#可直接启动的最小-websocket-server) → [配置能力](#metadata-参数从哪里来) → [适配模型](#模型适配示例) → [独立验收](#独立验收) → [交付](#模型状态与交付)。

## 安装

**协议包固定为 `g2-policy-protocol==0.5.0`，与域控保持一致。** 在模型服务的 Python 环境中，进入本仓库根目录安装：

```bash
python -m pip install -e '.[tools]'
```

`tools` 额外安装 WebSocket 和图像处理依赖，用于示例和验收工具；核心协议运行依赖仍只有 NumPy 和 MessagePack。模型框架及其 GPU 依赖由服务项目管理。

只有 wheel 时，可安装协议包和验收工具：

```bash
python -m pip install '/path/to/g2_policy_protocol-0.4.1-py3-none-any.whl[tools]'
```

wheel 不包含 examples 和 Markdown 文档；需要完整示例时使用源码交付包。安装或升级后先重启 GPU server，再启动域控 client。

## 可直接启动的最小 WebSocket Server

完整实现见 [`examples/minimal_server.py`](../examples/minimal_server.py)，包含一个 `MockPolicy` 实例。它接收模拟域控发送的 observation，返回重复当前输入位姿的动作，无需 GPU、模型权重或机器人硬件。

```text
client 发送 obs request
  → server 解码并校验，得到 observation
  → policy.infer(observation) 返回 {"actions": (T,20)}
  → server 构造带原 request_id 的 action response
  → 编码后发回 client
```

policy 在监听前实例化，可以在构造时加载真实模型。协议的 request_id、消息类型和错误响应由 server 管理，policy 处理模型输入输出。最小实现的关键调用是：

```python
policy = MockPolicy(action_horizon=metadata["action_horizon"])
# 收到一条 request 后：
observation = p.parse_request(payload, metadata=metadata)
result = policy.infer(observation)
response = p.build_response(
    request_id=observation["request_id"], actions=result["actions"],
)
```

`observation` 包含 `prompt`、实测 `state`、`images` 和可选关节状态/RTC 字段。示例的 mock 使用 state 生成动作；真实 policy 还需要使用图像和任务文本。

在一个终端启动服务：

```bash
python examples/minimal_server.py --host 127.0.0.1 --port 8000 \
  --policy-hz 30 --action-horizon 2
```

看到 `READY mock server ws://127.0.0.1:8000` 后，在另一个使用相同环境的终端运行：

```bash
g2-policy-check --url ws://127.0.0.1:8000 --report report.json
```

默认分别建立 MessagePack 和 JSON 连接，各发送两次观测。报告总状态应为 `passed`，退出码为 0；失败时为 `failed`、退出码为 1，并包含错误信息。工具仅校验返回动作，不执行机器人动作。没有安装命令入口时，可用 `python -m g2_policy_protocol.check_server`。

替换 `minimal_server.py` 中的 `MockPolicy` 实例，接入真实模型后，仍可使用同一条验收命令。工具接受模型生成的动作，不要求与输入状态相等。专用于模拟 server 的 [`examples/minimal_client.py`](../examples/minimal_client.py) 则要求返回保持原位的动作，不用于真实模型验收。

跨机器联调时，server 使用 `--host 0.0.0.0`，验收命令使用 `--url ws://<gpu-host>:8000`；实际主机和端口以部署配置为准。

这个最小 server 包含完整的连接流程：

- 从连接 URL 读取编码；默认 MessagePack，`?encoding=json` 使用 JSON，非法编码关闭连接。
- 连接后首先发送 metadata，再循环接收观测和返回动作。
- 校验请求、调用模拟模型、构造响应，并回传对应的请求 ID。
- 请求无效时返回 `invalid_request`，模型或动作构造失败时返回 `inference_error`；无法取得有效请求 ID 时回退为 `0`。
- 处理客户端断连，使用 Ctrl+C 停止服务。

示例声明双臂、头部相机、224×224 raw RGB、无 RTC、无需 joint_state。调整能力时修改 `main()` 中的 `build_metadata(...)`，并同步更新实际处理逻辑。接入真实模型时实现 `policy.infer(observation)`，完成模型输入转换、推理以及绝对 pose20 输出转换；需要 JPEG 时再补充解码和解码后的尺寸检查。

## 服务端接入流程

1. 根据模型和部署配置构造 `metadata`，在监听前验证能力配置。
2. 接受连接，根据连接参数确定编码，首先发送 `metadata`。
3. 解码收到的 frame，调用 `parse_request` 检查观测和能力匹配。
4. 按需解码 JPEG，再检查图像实际尺寸，将观测适配成模型输入。
5. 执行推理，将模型输出转换为 `base_link` 下的绝对 `(T,20)` 动作。
6. 构造 `action`，原样回传请求 ID；处理请求失败时返回 `error`。

metadata 应声明实际的策略频率、动作 horizon、执行臂、必需相机、图像编码/尺寸/resize 方式、RTC 支持和关节状态需求。ACT 的 `execution_steps` 可以小于 `action_horizon`，以保留 temporal ensemble 的单步执行行为。模型名称等诊断信息可放入 `extensions`，不参与客户端执行判断。

## metadata 参数从哪里来

| 参数 | 默认值 | 推荐来源与约束 |
|---|---|---|
| `policy_hz` | 无，必须传入 | 数据/训练的动作频率；不是 GPU 每秒推理次数，与域控约定匹配 |
| `action_horizon` | 50 | 模型一次预测的步数，与实际模型配置一致 |
| `execution_steps` | 不设置 | 此时 client 按 horizon 规划；需要较短执行前缀时显式声明，范围 1..horizon |
| `arm` | both | 模型的执行臂；单臂也返回完整 pose20 |
| `required_cameras` | 双臂三相机；单臂头部加同侧腕部 | 按模型真实输入声明，至少一个；不要照抄示例只声明头部 |
| `image_encoding` | raw | 部署传输方式；声明 jpeg 时必须实现解码 |
| `image_sizes` | 空字典 | 按相机声明输入 `[H,W]`；缺少某相机尺寸时 client 保留原尺寸 |
| `image_resize` | resize | 与训练预处理约定匹配，可选 letterbox；避免 server 再次重复缩放 |
| `requires_joint_state` | False | 模型需要关节状态时设 True，并实现关节状态到模型输入的映射 |
| `rtc_supported` | False | 仅在实际模型和服务实现支持 prefix/delay 时设 True |
| `extensions` | 不设置 | 诊断信息；不能依赖它改变域控执行行为 |

`p.build_metadata()` 只构造和校验声明。部署参数、实际预处理和推理逻辑必须使用一致的配置。

例如 30 Hz、horizon=50：相邻动作相隔约 33.3 ms，50 步覆盖约 1.67 秒的动作时段，首末动作的采样时间差为 49/30 秒。不是每次 GPU 推理必须耗时 33.3 ms。`execution_steps=1` 表示按较短执行前缀规划，不代表把模型 horizon 改成 1；服务端是否截断输出、进行 temporal ensemble，也必须在模型适配中明确。域控的实际发起推理时机还受补充策略和延迟影响。

## 自定义 server 时使用的 API

以下函数只展示协议与模型的边界。WebSocket 监听、连接参数解析、收发 frame 和模型加载由服务项目实现。示例使用 raw RGB 图像；`policy` 是调用方提供的模型实例，`infer()` 返回包含绝对 pose20 动作的字典，与最小 server 的接口一致。

```python
from g2_policy_protocol import codec, protocol as p

metadata = p.build_metadata(
    policy_hz=30,
    action_horizon=50,
    arm="both",
    required_cameras=["head_color"],
    image_encoding="raw",
    image_sizes={"head_color": [224, 224]},
    image_resize="resize",
    rtc_supported=False,
    requires_joint_state=False,
)


def encode_metadata(*, encoding="msgpack"):
    return codec.dumps(metadata, encoding=encoding)


def infer_frame(frame, policy, *, encoding="msgpack"):
    payload = codec.loads(frame, encoding=encoding)
    observation = p.parse_request(payload, metadata=metadata)
    try:
        result = policy.infer(observation)
        response = p.build_response(
            request_id=observation["request_id"], actions=result["actions"],
        )
    except Exception as exc:
        response = p.build_error(
            request_id=observation["request_id"],
            code="inference_error", message=f"{type(exc).__name__}: {exc}",
        )
    return codec.dumps(response, encoding=encoding)
```

示例捕获已解析请求后的模型与响应构造错误。外层连接处理还需处理 frame 解码、请求校验失败和断连；请求 ID 可用时回传原 ID。非法请求没有有效 ID 时，本仓库示例使用 `0` 作为错误响应的回退 ID。

## 模型适配示例

可运行的转换函数见 [`examples/model_adapter.py`](../examples/model_adapter.py)：

```bash
python examples/model_adapter.py
```

- `quaternion_wxyz_to_rot6d()`：明确输入为 wxyz、主动旋转；输出旋转矩阵前两列按列展开。xyzw 或反向变换需先转换。夹爪与坐标变换由模型适配另行处理。
- `expand_single_arm()`：将 `(T,10)` 放入对应手臂，另一臂保持输入实测位姿，返回 `(T,20)`；metadata.arm 仍需声明相应执行臂。
- `additive_pose_delta_to_absolute()`：仅适用于“每帧 xyz 和 rot6d 分量都相对同一实测 pose 做差、夹爪保持绝对值”的模型。不是通用刚体变换；轴角增量、局部坐标增量、逐步累积增量均不能直接套用。旋转矩阵增量需按训练约定明确左乘/右乘，不应直接相加。

在输出之前完成反归一化、去除模型填充维度、单位及坐标转换。关节动作模型需要在 server 内解决到末端 pose20 的转换；仅声明 requires_joint_state 不会让 client 接受关节动作输出。模型不使用另一臂，也不能把它的旋转填零。

## JPEG 可运行示例

停止占用 8000 端口的其他示例后：

```bash
python examples/jpeg_server.py
```

另一个终端运行：

```bash
g2-policy-check --report jpeg-report.json
```

该示例要求头部和右腕两张 JPEG 图像、letterbox 到 224×224，声明右臂执行。解码使用可选的 [`image_tools.decode_images()`](../src/g2_policy_protocol/image_tools.py)，得到 RGB uint8 数组后再检查尺寸。核心协议库不会主动解码图像。若使用 OpenCV 的 `imdecode()`，结果为 BGR，必须通过 `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` 转为 RGB 后再交给模型。

## 独立验收

工具根据 metadata 选择相机、缩放尺寸、raw/JPEG 编码和可选 joint_state，检查 metadata、响应类型、请求 ID、非空 `(T,20)` 动作及有限值。额外检查夹爪在 `[0,1]` 内、rot6d 两列非退化；这些是工具附加诊断，不是核心解析器的新规则。`--protocol-only` 可跳过附加诊断。返回 T 与 `action_horizon` 不同时记录警告，需结合 `execution_steps` 和实际调度确认。

默认使用明确标注为 synthetic 的合成观测。全套通过只代表该输入下的协议接入成功，不能证明模型质量、坐标系正确、动作可达、真实图像预处理一致或推理能满足机器人运行时限。报告中的 `round_trip_ms` 包含通信与处理时间，不等于 GPU 推理时间。

```bash
# 指定机器人侧约定频率；不指定时接受服务端声明的频率
g2-policy-check --url ws://gpu-host:8000 --policy-hz 30 --report report.json

# 使用提供的真实观测；慢模型可增加超时
g2-policy-check --sample observation.npz --timeout 120 --requests 3 --report report.json

# 服务端支持 RTC 时单独检查：首次无 prefix，后续带上次动作，delay=0
g2-policy-check --sample observation.npz --rtc --report rtc-report.json
```

可用 `--encoding msgpack` 或 `--encoding json` 单独排查一种编码。默认 all 才会覆盖两种。默认不启用 RTC；RTC 检查也不模拟真实异步延迟和轨迹裁切。工具重复发送同一条样本，不模拟动作执行后的机器人状态变化，也不主动发送错误请求。

## 观测样本格式

使用无 pickle 的 `.npz` 文件。保存原始 RGB 图像，由工具根据 metadata 处理，而非预先压成 JPEG。文件字段为：

| 字段 | 类型和含义 |
|---|---|
| `state` | 有限 `(20,)` 数组，实测绝对 pose20，base_link、米、rot6d、close_01 |
| `joint_state` | 可选有限 `(16,)` 数组；服务端要求关节状态时必须存在 |
| `prompt` | Unicode 标量字符串，非空任务文本 |
| `provenance` | Unicode 标量字符串，记录数据为合成或真实来源、脱敏情况及必要的标定背景 |
| 三个相机名称 | `head_color` / `hand_left_color` / `hand_right_color`，各自为 RGB uint8 H×W×3；至少一张，且覆盖服务端所需相机 |

生成可复现的合成文件：

```bash
g2-policy-sample synthetic.npz
g2-policy-check --sample synthetic.npz --report synthetic-report.json
```

合成样本含三张 240×320 彩色图案、单位旋转、示例位置和零关节状态。它们不是机器人录制数据，示例位置与关节状态也不保证符合真实机器人运动学。若模型需要真实场景，请由有权限的数据持有人导出并分发脱敏样本。本仓库未附真实录制。

真实样本的导出方式如下，变量由数据持有人提供：

```python
from g2_policy_protocol.samples import save_sample

save_sample("observation.npz", {
    "state": measured_pose20,
    "joint_state": measured_joint16,  # 不需要时省略
    "prompt": "拿起饮料",
    "provenance": "real; anonymized; describe capture and calibration here",
    "images": {
        "head_color": head_rgb,
        "hand_left_color": left_rgb,
        "hand_right_color": right_rgb,
    },
})
```

图像与状态应来自同一观测时刻；保存的是物理量，不是模型归一化后的值。样本格式是离线工具约定，不是新增的线上消息。真实数据的准确性和 provenance 文本由提供方负责，工具无法验证来源声明。

工具使用 Pillow 双线性缩放、居中黑色补边、JPEG quality=90。域控的 OpenCV 实现采用相同几何约定，但插值和 JPEG 像素值不保证逐像素一致。要求复现训练预处理时，应额外对比真实 client 数据或双方约定的参考图像。

## 模型状态与交付

无状态模型可以共享权重；temporal ensemble、历史图像、RTC 缓存等应按连接隔离，连接建立时初始化、断开时清理。示例在监听前创建一个共享的无状态 MockPolicy 实例；若替换为带可变缓存的 policy，需要为每条连接单独创建会话实例（权重可共享），避免并发请求修改同一缓存。v2 没有单独的 reset/任务结束消息；任务切换或重连是否表示新 episode 必须与域控集成人员约定，不能只靠 request_id 自行推断。通用工具的两种编码使用两个连接，因此可能触发两次初始化；显式的模型 reset 语义不在验收范围内。

交付给集成人员时提供 server 启动命令、模型/协议包版本、连接地址、metadata 配置、普通验收报告以及启用 RTC 时的专项报告。附上使用的样本 provenance、已确认的模型坐标与夹爪约定，并说明尚未进行真实机器人运行验证。

整包分发方法见[开发与发布](development.md#对外交付)。
