# 安装与接入

[文档入口](../README.md#文档导航) · [通信协议](protocol.md) · [policy 配置](policy-config.md)

从安装、模拟服务开始，再接入模型或 wholebody，最后使用同一套工具验收。以下命令在本仓库根目录执行。

[安装](#安装与版本) · [模拟服务](#启动模拟服务) · [模型接入](#接入模型) · [wholebody](#接入-wholebody) · [验收](#独立验收)

## 安装与版本

域控和 GPU 环境使用同一份 `g2-policy-protocol==0.6.0` 源码或 wheel。Python ≥3.10，基础依赖为 NumPy、MessagePack；`tools` 增加 WebSocket 与图像工具，`geometry` 增加 SciPy。

```bash
# 运行本页的服务、图像和验收示例
python -m pip install -e '.[tools]'

# 使用状态/动作几何转换时额外安装
python -m pip install -e '.[geometry]'
```

只有 wheel 时：

```bash
python -m pip install '/path/to/g2_policy_protocol-0.6.0-py3-none-any.whl[tools]'
```

域控只使用通信时可直接安装 wheel，无需 extras；wholebody 自身的运行依赖由该项目管理。模型框架和 GPU 依赖由模型项目管理。wheel 不含 examples 和 Markdown，完整源码交付见[开发维护](development.md#对外交付)。

当前源码统一使用 `g2.policy.v3`，旧端需要同步更新。源码调整后需重建 wheel，同版本号不保证内容相同。升级两端代码与协议包后，先重启 GPU server，再启动 client。

## 启动模拟服务

在一个终端运行：

```bash
python examples/minimal_server.py --host 127.0.0.1 --port 8000 \
  --policy-hz 30 --action-horizon 2
```

看到 `server listening on 127.0.0.1:8000` 后，在另一个使用相同环境的终端运行：

```bash
g2-policy-check --url ws://127.0.0.1:8000 --report report.json
```

报告总状态应为 `passed`，退出码为 0；失败时为 `failed`、退出码为 1，并包含错误信息。默认分别建立 MessagePack 和 JSON 连接，各发送两次观测。没有命令入口时，可用 `python -m g2_policy_protocol.check_server`。

[`minimal_server.py`](../examples/minimal_server.py) 使用共享 `PolicyServer` 和无状态 `MockPolicy`，声明 v3、双臂 pose20、头部相机、224×224 raw RGB、完整预测块、无 RTC。返回动作重复输入位姿；Ctrl+C 停止服务。

接入真实模型后仍使用同一条验收命令，工具不要求返回动作等于输入。只有 [`minimal_client.py`](../examples/minimal_client.py) 专门检查模拟模型的保持原位行为。跨机联调时 server 使用 `--host 0.0.0.0`，检查命令改为 `--url ws://<gpu-host>:8000`；实际端口以部署配置为准。

## 检查 JPEG 传输

停止占用 8000 端口的其他示例后：

```bash
python examples/jpeg_server.py
```

另一个终端运行：

```bash
g2-policy-check --report jpeg-report.json
```

该示例要求头部和右腕两张 JPEG 图像、letterbox 到 224×224，声明右臂执行。解码使用可选的 [`images.decode_images()`](../src/g2_policy_protocol/images.py)，得到 RGB uint8 数组后再检查尺寸。核心协议库不会主动解码图像。若使用 OpenCV 的 `imdecode()`，结果为 BGR，必须通过 `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` 转为 RGB 后再交给模型。

## 接入模型

`PolicyServer` 管理连接、编解码、请求校验、请求 ID、错误响应和线程调度。backend 负责解码图像、准备模型输入、执行推理，并将输出还原到声明的完整双臂绝对控制量。

```text
obs → 协议解码/校验 → backend.infer(request)
    → 模型输入转换 → 推理 → 动作还原
    → 协议响应校验/编码 → action
```

每次请求使用同一份 metadata 解析和构造响应。请求无效时返回 `invalid_request`，模型或响应构造失败时返回 `inference_error`；没有有效请求 ID 时使用 `0`。完整实现见 [`server.py`](../src/g2_policy_protocol/server.py)。

### 声明模型能力

| 参数 | 默认值 | 推荐来源与约束 |
|---|---|---|
| `policy_hz` | 无，必须传入 | 数据/训练的动作频率；不是 GPU 每秒推理次数，与域控约定匹配 |
| `action_horizon` | 50 | 模型一次预测的步数，与实际模型配置一致 |
| `arm` | both | 模型的执行臂；返回完整 `action_spec.dim`，未选侧保持实测或置零 |
| `required_cameras` | 双臂三相机；单臂头部加同侧腕部 | 按模型真实输入声明。允许空列表或只选腕部。不要照抄示例只声明头部 |
| `image_encoding` | raw | 部署传输方式；声明 jpeg 时必须实现解码 |
| `image_sizes` | 空字典 | 按相机声明输入 `[H,W]`；缺少某相机尺寸时 client 保留原尺寸 |
| `image_resize` | resize | 与训练预处理约定匹配，可选 letterbox；避免 server 再次重复缩放 |
| `requires_joint_state` | False | 模型需要关节状态时设 True，并实现关节状态到模型输入的映射 |
| `rtc_supported` | False | 仅在实际模型和服务实现支持 prefix/delay 时设 True |
| `extensions` | 不设置 | 普通键为诊断信息；`response_mode` 是受校验的响应长度契约（`full_chunk` / `variable_chunk`） |

`p.build_metadata()` 只构造和校验声明。直接生成 v3，可显式传入 `action_spec` / `observation_spec` / `timing` / `reset_supported`。部署参数、实际预处理和推理逻辑必须使用一致的配置。

新模型服务使用 v3：

```python
from g2_policy_protocol import protocol as p
from g2_policy_protocol.server import PolicyServer

metadata = p.build_metadata(
    policy_hz=30,
    action_horizon=50,
    arm="right",
    required_cameras=["head_color", "hand_right_color"],
    image_encoding="jpeg",
    image_sizes={"head_color": [224, 224], "hand_right_color": [224, 224]},
    image_resize="letterbox",
    rtc_supported=False,
    extensions={"response_mode": "full_chunk"},
    requires_joint_state=False,
)
```

未传入的 `action_spec` 默认为 `pose_target`（20 维绝对位姿）。其他控制量用 `protocol.build_action_spec("joint_position")` 等构造，校验拒绝错配的维度、字段、单位和坐标系。完整字段见 [v3 协议](protocol.md)。

`build_response` / `parse_response` / `parse_request` 都必须传入同一份 metadata。声明 `reset_supported=True` 时 backend 必须实现 `reset()`；`g2_policy_protocol.server.PolicyServer` 会在握手后处理 `reset` 消息。

模型加载和 metadata 构造在监听前完成。调用方提供的 `policy.infer(request)` 返回 `{"actions": array}`：

```python
PolicyServer(policy, metadata=metadata, host="0.0.0.0", port=8000).serve_forever()
```

### 状态与动作转换

安装 [geometry 可选依赖](guide.md#安装与版本)后，共享编解码器的使用方式见 [`examples/model_adapter.py`](../examples/model_adapter.py)：

```bash
python examples/model_adapter.py
```

示例在加载模型时解析一次 policy 并创建 `ActionCodec`，每次推理调用 `codec.decode()`。
配置显式声明右臂、相对于本块初始实测位姿的 xyz/rot6d 分量差、绝对夹爪；
解码器还原 `(T,20)` 绝对动作，并用实测值补齐未执行的左臂。
需要 SE(3)、逐步增量或其他旋转表示时，应在同一份 policy 中声明相应语义。

四元数的输入顺序在示例中显式从 wxyz 重排为 xyzw，再调用共享的旋转转换。
共享工具使用主动旋转；输出 rot6d 为旋转矩阵前两列按列拼接。示例不再独立实现转换公式。

在输出之前完成反归一化、去除模型填充维度、单位及坐标转换。关节动作模型可声明 `control_quantity=joint_position`，由域控按已绑定的关节执行器处理；未绑定的控制量会在打开硬件前被拒绝。仅声明 `requires_joint_state` 不会改变动作空间。模型不使用另一臂，也不能把它的旋转填零。

### 会话与并发

无状态 backend 可以直接复用；有 temporal ensemble、历史图像、RTC 缓存等可变状态时，实现 `new_session()` 为每条连接创建独立会话，权重可共享。backend 自行管理会话状态的资源生命周期。

声明 `reset_supported` 后，client 发 `reset`，server 调用该会话的 `reset()` 后回 `reset_ack`；reset 与机器人归位无关。episode 边界由 reset 明确表达，不从 request_id 推断。通用验收工具的两种编码使用两个连接，可能触发两次初始化。

共享 `PolicyServer` 使用一个模型工作线程串行执行所有会话的 infer/reset，网络事件循环继续处理连接与健康检查；等待模型执行权的请求不会预先提交到线程池。`new_session()` 只创建会话状态，模型权重可共享。服务关闭时等待工作线程结束。

需要自定义传输时，使用[协议 API](protocol.md#协议-api)；共享 WebSocket 边界的完整实现见 [`server.py`](../src/g2_policy_protocol/server.py)。

## 接入 wholebody

wholebody 的推理 Shell 共用 `app/policy_launcher.py`，通过 `--config` 选择部署配置；原有模型命名的 Python launcher 只选择默认配置。它们共用 `external_services.policy.runtime.transport.G2WebSocketPolicyClient`，封装连接、图像准备、消息编解码和请求 ID 管理；`parse_response` 会传入握手 metadata，按 `action_spec.dim` 校验动作宽度。

推理 YAML 的 `wire_encoding` 或 CLI `--wire-encoding json` 选择编码，默认是 MessagePack。连接地址可以使用 `ws://<gpu-host>:8000`；实际主机和端口以 GPU server 配置为准。JSON 连接使用 `?encoding=json`，wholebody 客户端会根据编码配置设置连接参数。

一次连接和推理按以下顺序进行：

1. 连接 server，接收首条 `metadata`。
2. 校验协议与服务能力；比较本地配置的 `policy_hz`，需要 RTC 时确认 server 支持。握手必须为 `g2.policy.v3`，执行器按能力选择。
3. 按握手采集观测：只发送 `observation_spec` 要求的 `state` / `joint_state` / `prompt` / `context`。图像按声明的编码、尺寸及 resize 方式准备。
4. 发送带唯一 `request_id` 的 `obs`，接收 `action` 或 `error`。v3 且 `reset_supported` 时，episode 边界可另发 `reset`。
5. 用同一份 metadata 校验响应 ID、动作维度和有限值，通过后交给控制层执行。

`metadata.arm` 决定执行臂，动作宽度为 `action_spec.dim`。单臂模型由 server 裁切/补齐完整双臂控制量；坐标、单位和字段顺序见[通信协议](protocol.md#动作控制量)。

当前 OpenPI G2 server 使用 `g2.policy.v3`，从 checkpoint 的 policy 配置推导 `control_quantity`；默认位姿配置为 `pose_target`（20 维绝对位姿）。域控按握手绑定执行器，并按 `arm` 只向所选侧发令。

### 校验与执行边界

- `validate_metadata` 在连接时检查协议和能力；`protocol == g2.policy.v3` 时走 v3 字段与 `action_spec` 布局校验。client 不依赖模型名称判断能否执行。
- `build_request` 按握手检查观测：按 `observation_spec` 检查 pose/joint/prompt、必需相机和 context。
- `parse_response(..., metadata=metadata)` 检查对应的 `request_id` 和有限非空 `(T, D)` 动作（D=`action_spec.dim`）；收到 `error` 时抛出 `RuntimeError`，非法响应抛出 `ValueError`。
- 校验失败或收到服务端错误时，不将该响应作为动作执行。wholebody 的 `G2WebSocketPolicyClient` 在响应解析抛出上述异常时会关闭连接并向上抛出异常。

协议校验不检查机器人动作的可达性或运动约束。动作限速、可达性和关节规划属于 G2 控制层。`control_quantity` 还必须有已绑定的执行器，否则域控在打开硬件前拒绝。

当前 wholebody 应用绑定 `pose_target`（EE 安全门）和 `joint_position`（`wholebody.arm.joint_command` 的关节限位、rad/s 速度上限及发送参数）。关节速度、力矩执行器尚未绑定；非零 `first_target_offset_s` 也未接入调度，均在打开硬件前拒绝。

历史策略通过独立线程按 `history_hz` 采样，episode 开头重复首帧，之后的采样缺口报错。默认应用支持历史和 episode 起点；命令、物体等额外参考需要显式 provider，在 `PolicyControlLoop(context_provider=...)` 接入，不能从当前实测末端伪造。

## 独立验收

工具根据 metadata 选择相机、缩放尺寸、raw/JPEG 编码和可选 joint_state，检查 metadata、响应类型、请求 ID、非空 `(T, D)` 动作及有限值（D=`action_spec.dim`）。额外检查夹爪在 `[0,1]` 内、rot6d 两列非退化；这些是工具附加诊断，不是核心解析器的新规则。`--protocol-only` 可跳过附加诊断。声明 `full_chunk` 的服务必须返回 T=`action_horizon`，不符时公共响应解析直接报错；可变长度服务返回长度不符时记录警告，需结合实际调度确认。

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

模型交付所需的版本、配置、样本与报告见[开发维护](development.md#对外交付)。
