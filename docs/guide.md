# 安装与接入

[文档入口](../README.md#文档导航) · [通信协议](protocol.md) · [policy 配置](policy-config.md)

从安装、模拟服务开始，再接入模型或 wholebody，最后使用同一套工具验收。未指定目录的命令在本仓库根目录执行；`/path/to/python` 等路径需要换成目标机器的实际路径。

[安装](#安装与版本) · [模拟服务](#启动模拟服务) · [模型接入](#接入模型) · [wholebody](#接入-wholebody) · [验收](#独立验收) · [排错](#排错)

## 安装与版本

本节是 **g2-policy-protocol 的唯一安装教程**。域控和 GPU 环境都要安装协议包，并使用同一份源码或同一个 wheel；模型框架（JAX、Torch、LeRobot）仍由 OpenPI 或 LeRobot 自己管理。当前协议包版本是 `0.7.0`，要求 Python ≥3.10，运行依赖为 NumPy、MessagePack、SciPy、websockets 和 Pillow。

包版本 `0.7.0` 和线上消息协议 `g2.policy.v3` 是两件事。历史上同一个包版本号存在不同构建，按[更新和交付一致性](#更新和交付一致性)核对 SHA256；不能只比较 `0.7.0`。

### 选择安装方式

| 场景 | 推荐方式 | 说明 |
|---|---|---|
| OpenPI GPU 训练/服务 | OpenPI 的 `scripts/g2/bootstrap.sh` | 它同时安装 OpenPI、模型依赖和协议包 |
| LeRobot GPU 训练/服务 | LeRobot 的 `scripts/bootstrap.sh` | 它同时安装 LeRobot、项目依赖和协议包 |
| wholebody 域控 | 安装到 wholebody 使用的 venv | 只需要协议通信层，不装 JAX、Torch 或 LeRobot |
| 独立协议开发/模拟服务 | 本仓库自己的 venv | 适合运行 examples、`g2-policy-check` 和协议测试 |

源码方式适合同一台机器上的开发和联调；新服务器和跨机器交付推荐 wheel：先按[更新和交付一致性](#更新和交付一致性)构建、分发文件，再按所属项目安装，最后执行[验证安装](#验证安装)。

### 前置条件

需要 Python ≥3.10、`uv`，以及可访问 Python 包源的网络。Ubuntu 可先安装：

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

如果使用 OpenPI/LeRobot 训练，还要按对应仓库文档准备 GPU 驱动、Git LFS 和视频解码环境。
这些不是协议包的安装依赖。

### 独立环境安装（开发、模拟服务和协议测试）

在本仓库根目录执行：

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e .
.venv/bin/python -c "import g2_policy_protocol; print(g2_policy_protocol.__file__)"
```

要运行协议测试，再安装开发依赖：

```bash
uv pip install --python .venv/bin/python pytest jsonschema
```

如果机器已有项目 venv，把 `.venv/bin/python` 换成实际解释器路径即可。不要把另一个项目的 `.venv` 复制过来；editable 安装记录了本机路径。

### OpenPI GPU 环境

OpenPI 使用相邻源码时，目录应为：

```text
~/openpi/
~/g2-policy-protocol/
```

首次安装由 OpenPI 自己的 bootstrap 一次完成，协议包装在 OpenPI 的 `.venv` 中：

```bash
cd ~/openpi
bash scripts/g2/bootstrap.sh
# 开发测试环境：
# bash scripts/g2/bootstrap.sh --dev
```

如果只拿 wheel，不放相邻协议源码：

```bash
cd ~/openpi
G2_PROTOCOL_WHEELS="$HOME/g2-wheels" bash scripts/g2/bootstrap.sh
```

`G2_PROTOCOL_WHEELS` 目录必须包含目标 wheel，例如
`g2_policy_protocol-0.7.0-py3-none-any.whl`。OpenPI bootstrap 负责安装项目本身、
OpenPI 依赖和协议包；不要只执行 `uv venv` 或只安装第三方依赖。
若目标环境已有旧协议包，bootstrap 结束后按[更新和交付一致性](#更新和交付一致性)显式刷新并验证。

### LeRobot GPU 环境

LeRobot 项目使用自己的 bootstrap。相邻源码方式：

```bash
cd ~/lerobot-finetune
bash scripts/bootstrap.sh
```

wheel 方式：

```bash
cd ~/lerobot-finetune
G2_PROTOCOL_WHEELS="$HOME/g2-wheels" bash scripts/bootstrap.sh
```

### wholebody 域控环境

将协议安装到 wholebody 实际启动脚本使用的 venv；下面的路径只是示例：

```bash
uv pip install --python /path/to/wholebody/.venv/bin/python -e /path/to/g2-policy-protocol
```

跨机器交付时安装 wheel：

```bash
uv pip install --python /path/to/wholebody/.venv/bin/python \
  /path/to/g2_policy_protocol-0.7.0-py3-none-any.whl
```

wholebody 自身的运行依赖由 wholebody 管理；不要把 OpenPI、Torch 或 JAX 依赖装进域控环境。
wheel 不包含 examples 和 Markdown；完整源码交付见[开发维护](development.md#对外交付)。

### 验证安装

在每台机器上都使用实际启动服务或客户端的同一个解释器：

```bash
/path/to/python - <<'PY'
from importlib.metadata import version
import g2_policy_protocol
from g2_policy_protocol.data.sources import resolve_sources

print("package:", version("g2-policy-protocol"))
print("module:", g2_policy_protocol.__file__)
assert version("g2-policy-protocol") == "0.7.0"
sources = resolve_sources({"sources": [
    {"path": "probe-a", "weight": "auto"},
    {"path": "probe-b", "weight": "auto"},
]}, base_dir=".")
assert [source["weight"] for source in sources] == [None, None]
print("weight: auto supported")
PY
uv pip check --python /path/to/python
```

`module` 应指向预期的源码目录或 wheel 的 `site-packages`，而不是另一份旧 venv。
`weight: auto supported` 验证当前数据混合功能可用，不访问数据集文件，也不能代替 wheel 内容一致性校验。
OpenPI 使用 `.venv/bin/python` 执行这段验证，再运行 `bash scripts/g2/train/train_pipeline.sh --dry-run` 检查实际配置；多源配置需要本地数据元数据。

### 更新和交付一致性

仅以 `-e` 安装的协议源码会直接反映文件变化；OpenPI 的相邻路径依赖默认会构建后安装，修改源码不代表已安装内容同步变化。
若使用相邻源码，同版本更新可在对应项目环境中显式刷新：

```bash
uv pip install --python /path/to/python --no-deps --reinstall /path/to/g2-policy-protocol
```

`--no-deps` 仅用于依赖未变化的协议包刷新；版本或依赖要求有变化时，先更新各项目的依赖声明和锁文件，再使用各项目 bootstrap 完整同步。
发布新的代码变更时更新包版本及调用方约束，避免继续覆盖同版本发行文件。

wheel 在保存了当前改动的源码目录中构建。使用新的输出目录，并让 uv 先生成 sdist、再从 sdist 构建 wheel，避免复用本地 `build/` 中已删除模块的残留：

```bash
cd ~/g2-policy-protocol
G2_PROTOCOL_BUILD_DIR="$(mktemp -d /tmp/g2-protocol-wheel-XXXXXX)"
uv build --out-dir "$G2_PROTOCOL_BUILD_DIR"
sha256sum "$G2_PROTOCOL_BUILD_DIR/g2_policy_protocol-0.7.0-py3-none-any.whl"
```

把上一步生成的**同一个 wheel 文件**复制到域控和 GPU 服务器的 `~/g2-wheels/` 目录，核对 SHA256 与构建机一致。
此目录只放当前要安装的协议 wheel，不混放旧构建：

```bash
sha256sum ~/g2-wheels/g2_policy_protocol-0.7.0-py3-none-any.whl
```

更新同版本 wheel 时必须显式 `--reinstall`；否则已有安装可能被保留：

```bash
uv pip install --python /path/to/python --no-deps --reinstall \
  ~/g2-wheels/g2_policy_protocol-0.7.0-py3-none-any.whl
```

升级顺序：先在两端安装同一个 wheel，再重启 GPU server，最后启动 wholebody client。
不能只替换一端。新协议代码和新训练配置必须成套更新。

### 常见安装问题

| 现象 | 原因和处理 |
|---|---|
| `No module named 'g2_policy_protocol'` | 安装到了另一份 Python；用实际启动脚本的解释器执行验证和安装命令 |
| `No module named 'openpi'` | OpenPI 本身没有安装；回到 OpenPI 仓库执行 `scripts/g2/bootstrap.sh` |
| `weight: auto` 被报成必须是正数 | 使用了旧的同名/同版本 wheel；重新构建并用 `--reinstall` 安装当前 wheel |
| 版本显示相同但行为不同 | 同版本 wheel 内容不同；比较 SHA256，并重新分发同一个文件 |
| 协议握手不兼容 | 两端协议或接入代码不一致；同时更新并按升级顺序重启 |
| `uv` 找不到 | 安装 uv 后重新加载 `$HOME/.local/bin/env`，或使用 uv 的绝对路径 |

## 启动模拟服务

后续 `python` 和 `g2-policy-check` 命令使用已验证的目标环境。独立环境安装后可在协议仓库根目录激活：

```bash
source .venv/bin/activate
```

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

接入真实模型后仍使用同一条验收命令，工具不要求返回动作等于输入。跨机联调时 server 使用 `--host 0.0.0.0`，检查命令改为 `--url ws://<gpu-host>:8000`；实际端口以部署配置为准。

## 检查 JPEG 传输

停止占用 8000 端口的其他示例后：

```bash
python examples/minimal_server.py --jpeg
```

另一个终端运行：

```bash
g2-policy-check --report jpeg-report.json
```

该示例要求头部和右腕两张 JPEG 图像、letterbox 到 224×224，声明右臂执行。解码使用 [`images.decode_images()`](../src/g2_policy_protocol/images.py)，得到 RGB uint8 数组后再检查尺寸。核心协议库不会主动解码图像。若使用 OpenCV 的 `imdecode()`，结果为 BGR，必须通过 `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` 转为 RGB 后再交给模型。

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

共享编解码器的使用方式见 [`examples/model_adapter.py`](../examples/model_adapter.py)：

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

协议支持 `pose_target` 和 `joint_position`，wholebody 按控制量绑定执行器；非零 `first_target_offset_s` 在打开硬件前拒绝。执行器参数、历史采样和额外 context 的接入方式由 wholebody 维护，见 [wholebody policy 通信契约](../../wholebody/docs/dev/policy-protocol.md)。

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

不传 `--sample` 时工具使用内置合成样本：三张 240×320 彩色图案、单位旋转、示例位置和零关节状态。它们不是机器人录制数据，示例位置与关节状态也不保证符合真实机器人运动学。若模型需要真实场景，请由有权限的数据持有人导出并分发脱敏样本。本仓库未附真实录制。

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

## 排错

错误信息来自 `protocol.py` 的 `ValueError`。server 端的错误会以 `error` 消息返回给 client，message 为 `异常类型: 原文`。

| 错误信息（节选） | 发生阶段 | 常见原因与处理 |
|---|---|---|
| `metadata.protocol has an unsupported value` | 握手 | 一端仍是旧版本。两端换成同一个 wheel，先重启 server |
| `server policy_hz disagrees with configured policy_hz` | 握手 | client YAML 写死的 `policy_hz` 与模型训练频率不同；改 client 配置或设为 null 接受 server 声明 |
| `server does not support RTC` | 握手 / 请求 | client 开了 RTC，server 声明 `rtc_supported=false`；关闭 client RTC 或换支持 RTC 的服务 |
| `action_dt disagrees with policy_hz` | 握手 | 手工构造的 `timing.action_dt` 不等于 1/policy_hz；改用 `build_metadata` 默认值 |
| `action_spec must use the canonical layout, units and frames` | 握手 | 手写了 action_spec；改用 `build_action_spec(quantity)` |
| `requires_joint_state disagrees with observation_spec.joint` | 握手 | 两处声明不一致；只设其中一个，让 `build_metadata` 推导另一个 |
| `missing required cameras` | 请求 | client 没有发送 `required_cameras` 中的相机；检查相机名和 client 相机配置 |
| `... size disagrees with metadata image_sizes` | 请求 | 图像尺寸与声明不符；client 应按握手缩放，server 解码 JPEG 后再校验 |
| `server requires a nonempty prompt` | 请求 | `observation_spec.prompt=true` 但未给任务文本 |
| `history requires observation_time_ns for the latest sample` | 请求 | history_steps>1 时未带观测时间 |
| `response request_id does not match observation` | 响应 | server 没有原样返回请求 ID，或 client 复用了连接上的旧响应 |
| `full_chunk response must contain action_horizon actions` | 响应 | 声明 `full_chunk` 却返回了截短的动作；返回完整 H 步或改声明 `variable_chunk` |
| `actions exceeds action_horizon` | 响应 | 返回步数大于 H；检查模型输出是否多了维度或未裁切 |
| `inference_delay exceeds ...` | 请求 | RTC delay 超过 H 或前缀长度；检查 client 的延迟估计 |
| `actions must be finite nonempty (T, D), got ...` | 响应 | 模型输出未还原到完整双臂 D 维（例如单臂未补齐、padding 维未去掉）；也可能含 NaN/Inf；见[状态与动作转换](#状态与动作转换) |

握手和协议都通过、但机器人动作异常时，问题通常不在协议层：先核对坐标系（base_link/flange）、rot6d 列顺序、夹爪 0 开 1 闭、单位，再核对训练与部署的图像预处理是否一致。
