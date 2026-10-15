# g2-policy-protocol

框架无关的 G2 policy 通信契约。**唯一结构定义是 [`g2.schema.json`](src/g2_policy_protocol/g2.schema.json)**，ACT、LeRobot π₀.₅、OpenPI 和第三方模型都使用 `g2.policy.v2`。

```text
src/g2_policy_protocol/
  g2.schema.json   JSON Schema SSOT（消息、字段、布局、图像、能力）
  _generated.py    由 schema 生成的常量与静态校验，禁止手改
  _validation.py   生成代码复用的轻量类型/范围检查
  protocol.py      消息构造、解析及动态语义校验
  codec.py         JSON / MessagePack；NumPy / Base64
```

Schema 按阅读顺序组织：先看 `metadata / obs / action / error` 四类消息，再看 pose、joint、actions 和 image 的数据表示；末尾是共用的尺寸、计时、请求 ID 和数值类型。`x-g2` 中列出向量字段顺序和单位。

运行依赖只有 NumPy、MessagePack；Python ≥3.10。不依赖机器人、模型、OpenCV、WebSocket 库。JSON Schema 校验只用于开发测试；运行时不读取 Schema 文件，不逐帧执行完整 Schema 验证。

静态规则由 `tools/generate_protocol.py` 在开发阶段编译成普通 Python：字段、必填项、枚举、类型、范围、集合长度与唯一性，以及计时/错误文本检查。NumPy 向量维度和原始图像规格也从 Schema 提取，运行时不遍历像素或逐元素解释 Schema。`protocol.py` 保留数组归一化/有限值、非空白 prompt、能力匹配、RTC 和 request_id 对应关系等动态检查；`codec.py` 保留编解码和数据完整性检查。

生成器只支持当前使用的静态关键字，遇到新的不支持规则会报错。修改 Schema 后必须重新生成并运行测试；测试也检查修改边界、枚举或必填字段会改变生成代码的行为。0.4.1 保持 `g2.policy.v2` 通信格式。

## 谁要装

**域控（client）和 GPU（server）都要装同一版本**，不能只装一端。

| 端 | 环境 | 用法 |
|----|------|------|
| 域控 | wholebody 的 venv（如 `/home/agi/.venv`） | `PolicyClient` 编解码 obs/action |
| GPU · G2_pi | `G2_pi/.venv` | `serve_pi05` / `serve_act` |
| GPU · OpenPI | openpi 的 uv 环境 | `G2PolicyServer` |

OpenPI 非 G2 的通用原生 server 不依赖本包。域控不要把 Torch / JAX / lerobot 全量装进遥操环境，只装本协议包。

## 安装与修改契约

两端固定 `g2-policy-protocol==0.4.1`。任选源码或 wheel，版本必须一致，装完后重启 GPU server 再启域控 client。

```bash
# 源码（可编辑）
python -m pip install /path/to/g2-policy-protocol

# 或同一 wheel（推荐跨机分发）
python -m pip install dist/g2_policy_protocol-0.4.1-py3-none-any.whl
```

各仓常用方式：

```bash
# 域控
source /home/agi/.venv/bin/activate
python -m pip install /path/to/g2-policy-protocol   # 或上述 wheel

# G2_pi：bootstrap 若发现相邻 ../g2-policy-protocol 会自动 editable 安装
bash scripts/bootstrap.sh
# 或手动：uv pip install -e ../g2-policy-protocol

# OpenPI：pyproject 的 tool.uv.sources 指向 ../g2-policy-protocol，uv sync 安装
# 仅部署 wheel 时，把 path 改成本地 wheel 路径，再 uv lock && uv sync
```

修改契约后在本仓生成、测通并打 wheel：

```bash
# 修改 g2.schema.json 后生成并验证常量
python tools/generate_protocol.py
python tools/generate_protocol.py --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
python -m pip wheel --no-index --no-deps --no-build-isolation . -w dist
```

测试依赖 `pytest`、`jsonschema`；发布包不依赖它们。

## 连接与消息

WebSocket 默认二进制 MessagePack；连接地址带 `?encoding=json` 时双方整条连接使用 JSON text frames。server 首先发送 metadata，client 完成能力校验后才发观测。编码方式只由连接参数确定，不在 metadata 中重复声明；不混发或自动猜测帧类型。

| 消息 | 内容 |
|---|---|
| `metadata` | protocol、Hz、horizon、执行臂、相机要求、图像编码/尺寸/resize、RTC、joint state 要求 |
| `obs` | request_id、prompt、实测 pose20、images；可选 joint_state、RTC prefix/delay、观测时间 |
| `action` | 原样回传 request_id、绝对 `(T,20)` 动作、可选计时和 RTC 信息 |
| `error` | 原样回传 request_id、code、message；不执行动作 |

metadata 只声明服务端可变的能力，client 不检查模型名称。pose20、base_link、米/rot6d/绝对动作由 `g2.policy.v2` 固定；向量顺序和固定语义记录在 Schema 的 `x-g2` 中。模型名称等诊断信息可放在 `extensions`，不影响执行。ACT 的 `execution_steps` 可小于模型 `action_horizon`，保留 temporal ensemble 的单步执行行为。

## 两端共用的 Python API

```python
import numpy as np
from g2_policy_protocol import codec, protocol as p

# server：支持任意模型。这里只展示消息，不负责监听端口。
meta = p.build_metadata(policy_hz=30,
                        required_cameras=["head_color"])
# client 收到 metadata 后校验；G2 会比较本地配置的 policy_hz。
p.validate_metadata(meta, policy_hz=30)
request = p.build_request(
    request_id=42, prompt="拿起饮料", state=np.zeros(20, np.float32),
    images={"head_color": np.zeros((224, 224, 3), np.uint8)}, metadata=meta,
)
# server：根据模型适配 observation，再推理/还原成绝对 pose20。
observation = p.parse_request(codec.loads(codec.dumps(request)), metadata=meta)
# actions = model_adapter.predict(observation)
response = p.build_response(request_id=observation["request_id"], actions=actions)
# client：校验关联 ID 与有限值，之后才交给动作控制层。
actions = p.parse_response(codec.loads(codec.dumps(response)), request_id=42)
```

JSON 使用相同 API，只有 `codec.dumps(message, encoding="json")` / `codec.loads(text, encoding="json")` 不同。底层 `packb/unpackb/json_dumps/json_loads` 也保留。普通 JSON 数值列表可直接表示 state/action；NumPy 数组与 bytes 的跨语言表示如下：

```json
{
  "state": {"$g2":"ndarray","dtype":"<f4","shape":[2],"data":[1.0,2.0]},
  "image": {"$g2":"bytes","encoding":"base64","data":"/9j/2Q=="}
}
```

这是类型表示示例，两个数不是合法 pose20，图片仅占位。数组 data 为与 shape 对应的嵌套列表；NumPy 标量用 `{"$g2":"scalar","dtype":"<f4","data":1.0}`。`$g2` 为保留键；JSON 拒绝非有限数字、错误 shape/Base64 和超范围整数。MessagePack 保持 OpenPI 兼容的 NumPy 标记，不使用 pickle。

## 动作与图像含义

- `state` / `actions`：左臂 xyz(3)、rot6d(6)、close_01(1)，再右臂，共 20 维；`base_link` 下绝对位姿，位置米。
- rot6d：`R[:, :2].reshape(-1, order="F")`；夹爪 `0=open`、`1=closed`。
- `joint_state`：可选 16 维，左右各 7 个弧度关节角及夹爪；用于关节 checkpoint。它不替代实测 pose20。
- 单臂模型仍接收完整 pose20，返回完整绝对动作；server 负责裁切/扩展，G2 执行端只向 metadata.arm 指定的臂发令。
- images 使用 `head_color`、`hand_left_color`、`hand_right_color`。仅 metadata.required_cameras 必须存在；不需要的相机可省略或为 null。
- 原始图像为 RGB uint8 H×W×3；JPEG 为 bytes。`image_sizes` 按相机声明 `[H,W]`，`image_resize` 为 `resize` 或 `letterbox`。编码、缩放和解压由适配层负责。
- JPEG 解码后 server 可调用 `validate_images(..., decoded=True)` 再检查实际尺寸。包不解压图像。
- RTC prefix 为有限非空 `(T,20)` 绝对动作；delay 为非负整数策略步数。仅 rtc_supported=true 时可以启用。
- schema 描述静态结构，Python 校验动态 shape、有限值、相机/RTC 能力、频率和 request_id。动作限速、可达性和关节规划属于 G2 控制层。

## 接入与版本边界

wholebody 的三个启动入口共用 `policy.runtime.transport.PolicyClient`；推理 YAML 的 `wire_encoding` 或 CLI `--wire-encoding json` 选择编码。G2_pi `policy_server`、OpenPI `G2PolicyServer` 使用同一包。OpenPI 非 G2 机器人的通用原生 server 不受影响。

v2 删除握手中的 `schema_version / state_layout / action_layout / frame / rotation / action_semantics / encoding / policy_type`，不兼容 `g2.policy.v1` 和更早的两套 G2 消息。**域控和 GPU 两端必须同步安装 0.4.1、更新代码并重启**，不能只替换一端。没有真实权重/GPU 的回环测试只验证协议和模型边界，不代表模型动作质量。
