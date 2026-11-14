# 开发 G2 policy 客户端与服务端

[README](../README.md) · [完整通信协议](protocol.md)

先读这条路径：**服务声明 → 机器人观测 → 模型适配与推理 → 绝对动作**。
`types.py` 解释每份数据；`protocol.py` 构造、解析和校验消息；
`runtime` 负责传输。训练配置、数据集和模型内部表示按需接入。

## 先跑通一对示例

安装本包后，在两个终端分别执行：

```bash
python examples/minimal_server.py --port 8000
python examples/minimal_client.py --url ws://127.0.0.1:8000
```

客户端应打印 `(2, 20)`、`float32`。示例使用合成观测，服务端返回保持当前位姿的动作，
不加载模型、不连接机器人，也不执行动作。

服务端加 `--jpeg`、客户端加 `--encoding json` 可检查另一条图像和消息编码路径。
两个编码选项独立：JPEG/raw 是图像表示，JSON/MessagePack 是整条消息的帧格式。

## 握手和消息按什么顺序传递

WebSocket 建立后，服务端主动发送 metadata。客户端校验版本、观测需求和动作布局，
校验通过即可发送 obs；应用协议没有另一个握手确认消息。JSON/MessagePack 由连接 URL
的 `encoding` 参数选择，图像 raw/JPEG 则由 metadata 声明。

```text
客户端 PolicyClient                         服务端 PolicyServer
connect() ─── 建立 WebSocket ──────────────▶ _handler()
          ◀── metadata ──────────────────── 发送已校验的服务声明
validate_metadata()
infer()   ─── obs，request_id=N ──────────▶ _handle_request()
          ◀── action 或 error，同一个 N ─── 校验 → 工作线程解码/推理 → 校验结果
parse_response()
reset()   ─── reset，request_id=N+1 ───────▶ 工作线程清理本会话
          ◀── reset_ack 或 error，N+1 ───── 清理完成后回复
```

字段全集在 `types.py`；消息规则按相同顺序排列在 `protocol.py`。
`parse_context/parse_rtc/parse_actions` 已收为内部辅助函数，调用方使用完整消息入口。
诊断工具也复用 `PolicyClient`，握手、超时和响应校验只有一份运行时实现。

## 数据经过哪些边界

所有记录仍是普通字典，类型定义只说明字段；数值校验由 `protocol` 执行。

| 数据 | 定义 | 谁负责产生 | 含义 |
|---|---|---|---|
| 服务声明 | `Metadata` | 服务端 | 需要哪些观测、输出什么动作、时间轴和可选能力 |
| 线上观测 | `ObservationMessage` | 客户端 | `obs` 消息；必需字段取决于声明 |
| 解析后的线上观测 | `ObservationRequest` | `parse_request` | 已校验、数值数组已规范化；JPEG 仍是 bytes |
| 推理输入 | `Observation` | 服务端运行时 | 图像统一为 RGB 数组，当前帧/历史已选择 |
| 推理输出 | `InferenceResult` | backend | 已还原的绝对动作及可选诊断；不含消息类型和请求 ID |

`ActionResponse` 是服务端给推理输出添加类型和请求 ID 后的线上响应。
其他消息由 `ErrorResponse`、`ResetRequest`、`ResetAcknowledgement` 描述。

不要混淆这些量：

| 量 | 形状 / 约定 |
|---|---|
| `state` | 实测 pose20；每臂 xyz（米）、rot6d、夹爪，左臂在前 |
| `joint_state` | 实测 joint16；每臂七个关节角（弧度）与夹爪，左臂在前 |
| 位姿与动作的坐标 | `base_link` 下的 `flange`；rot6d 为旋转矩阵前两列按列拼接 |
| 夹爪 | 绝对闭合比例，0 全开、1 全闭 |
| `actions` | `(T,20)` 位姿或 `(T,16)` 关节绝对目标；单臂模型也返回双臂完整布局 |
| `command_pose/joint` | 下发的命令参考，与实测 state 区别开 |
| 状态/图像历史 | 从旧到新，最后一项是当前帧；长度为 `history_steps` |
| 当前图像 | raw 为 RGB uint8 `(H,W,3)`；JPEG 为压缩 bytes |
| 图像历史 | raw 为 uint8 `(L,H,W,3)`；JPEG 为 L 个 bytes 的列表 |
| 动作时间轴 | 第 i 项偏移 `first_target_offset_s + i * action_dt`，单位秒 |
| `observation_time_ns` | 客户端单调时钟，标记最新观测；不与服务端时钟比较 |

历史采样间隔为 `1/history_hz`，动作间隔为 `1/policy_hz`，两者含义不同。
模型内部的增量、其他坐标系、归一化、tensor 和 padding 由模型适配器处理。
上表图像格式指线上表示。backend 的 `Observation.images` 总是 RGB uint8 数组：
当前帧为 `(H,W,3)`，图像历史为 `(L,H,W,3)`。图像历史从 wire context 移入 images，
backend 的 context 只保留数值参考和状态历史，不再需要判断 JPEG 或选择历史分支。
坐标、context 与 RTC 的完整定义见[通信协议](protocol.md)。

## 写服务端

最小 backend 只需实现一个方法：

```python
from g2_policy_protocol.types import InferenceResult, Observation

class Backend:
    def infer(self, request: Observation) -> InferenceResult:
        # request 已校验且图像已解码；只需转换为模型输入。
        model_inputs = to_model_inputs(request)
        model_outputs = model.predict(model_inputs)
        # 还原坐标、delta 和 padding，补齐非执行臂的实测值。
        absolute_targets = to_robot_targets(model_outputs, request)
        return {"actions": absolute_targets}
```

`infer()` 接收的 `prompt`、`context`、`rtc_enabled`、`inference_delay`、
`prev_chunk_left_over` 已经补默认值；未提供且非必需的 `state/joint_state` 保持缺省。
`images` 总是存在；有历史时直接提供历史数组，包含最后的当前帧。
共享运行时在模型工作线程解码图像并检查尺寸，backend 不再调用 JPEG 解码工具。
观测数组可能与解析后的请求共享存储，模型适配器应避免原地修改它们。

完整请求路径集中在 `PolicyServer._handle_request`：

```text
网络帧 → runtime.codec.loads
       → protocol.parse_request           校验结构、需求和数值，生成 ObservationRequest
       → runtime.images.decode_observation 工作线程内解码、选择历史，生成 Observation
       → backend.infer                    适配模型、推理、还原动作
       → protocol.build_response          校验绝对动作，添加原 request_id
       → runtime.codec.dumps → 网络帧
```

消息/图像不合法返回 `invalid_request`；推理异常或模型结果不合法返回 `inference_error`。
backend 不构造响应信封，也不分配或修改请求 ID。
`server_timing.decode_ms/infer_ms/unpack_ms/process_ms` 由共享运行时记录；
模型内部计时放在 `policy_timing`，其他附加诊断可由 backend 返回。

### 会话状态放在哪里

| 情况 | 实现方式 |
|---|---|
| 无状态推理 | 实现 `PolicyBackend.infer`，同一 backend 可被连接共享 |
| 每连接有历史/缓存 | 实现 `PolicySessionFactory.new_session`，返回拥有独立 episode 状态的 backend |
| 支持新 episode | 返回的 backend 实现 `ResettablePolicyBackend.reset`，并声明 `reset_supported=True` |

这些接口是结构类型，不要求继承。只有 `infer` 是最小 backend 的必需方法。
服务端检查实际返回的 session 是否满足声明；factory 本身无需实现 `infer/reset`。
模型权重可由多个 session 共享，episode 缓存由各自 session 持有。
每次连接创建一个 session；reset 清理该 session 的 episode 状态，不负责机器人归位。
所有连接的推理和 reset 在同一工作线程串行执行，网络循环保持响应。

## 写客户端

使用 `runtime.client.PolicyClient`：

```python
from g2_policy_protocol.runtime.client import PolicyClient

with PolicyClient("ws://127.0.0.1:8000") as client:
    metadata = client.metadata  # 握手已校验
    actions = client.infer(prompt="拿起饮料", state=pose20, images=rgb_images)
```

首次成功 `connect()` 前读取 `metadata` 会抛 `ConnectionError`，不会返回不完整声明。
当前 `images` 传 RGB uint8 数组，客户端按声明缩放和编码。
`context.image_history` 由调用方按声明采样并预先编码，便于缓存每个历史帧；
提供它时客户端跳过当前帧准备和发送，避免重复工作。
`state` 是 pose20，`joint_state` 是 joint16；客户端只发送服务声明需要的状态量。
`reset()` 等待当前会话完成清理，`cancel_pending()` 可从控制线程中断等待并禁止该实例重连。

`examples/minimal_client.py` 是独立可运行示例。共享客户端内部的完整过程是：

1. 建立 WebSocket，接收服务端先发的 metadata，用 `validate_metadata` 校验。
2. 按声明采集真实观测，准备图像尺寸/编码及所需历史和参考量。
3. 分配递增的请求 ID，`build_request` 后通过 codec 编码发送。
4. 接收并解码响应，`parse_response` 检查 ID、错误、动作维度和长度。
5. 只将校验通过的动作交给调用方调度和执行。

连接尝试在 `connect_timeout_s` 内重试；单次推理/reset 请求只发送一次。
收到错误、响应不匹配、连接中断、消息无法解码或超时后关闭当前连接并抛异常。
真实机器人由控制层负责历史采样、恢复策略、动作调度和执行。
wholebody 的适配类复用共享传输，只保留 `task/pose` 参数名及 OpenCV 图像优化。

## 已有 backend 的迁移

`PolicyBackend.infer` 现在接收 `Observation`。删除 backend 内的 `decode_images`、
`decode_image_history` 和当前帧/历史选择分支，直接使用 `request["images"]`。
其余数值状态、参考量、RTC 和动作布局不变。线上协议仍为 `g2.policy.v3`；
共享运行时与 backend 的 Python 接口需同步更新。

## 阅读与修改顺序

1. `types.py`：输入输出的字段、形状、单位和默认值保证。
2. `protocol.py`：每种消息的构造/解析入口，按交互顺序阅读。
3. `runtime/client.py` / `runtime/server.py`：连接、session、模型线程和错误分类。
4. `validation.py`：基础标量、字段集合和有限数组检查；具体消息规则在 `protocol.py` 的对应入口中。

`constraints.py` 是标准库实现的基础几何检查，两类调用方都可使用：
通信核心与训练 `contracts`。通信核心不导入训练配置；`contracts` 不加载数值、图像和网络依赖。
`capabilities` 单向地把已解析 policy 语义转换成服务声明。


## 模块边界

一个顶层包 `g2_policy_protocol` 内按职责分层。`contracts.policy` 定义 policy 语义，`contracts.dataset` 定义磁盘契约；`processing` 按这些契约采样、编码、统计和混合；`runtime` 负责客户端、服务端、序列化和图像传输。

模型适配器负责归一化、tensor、padding、token 和模型专属图像布局。共享库不加载模型框架，也不提供 PI0.5 专用模块。通信核心不依赖 `contracts` 或 `processing`，因此只做客户端或服务端时无需阅读训练算法。

修改顺序建议是：先改 `types.py` 和 `protocol.py` 的语义与校验，再改 `runtime` 的交互流程，最后改 `contracts` / `processing` 或模型适配器。线上字段或含义变化必须同时更新 [安装文档中的兼容性规则](install.md#兼容性如何保证)。
