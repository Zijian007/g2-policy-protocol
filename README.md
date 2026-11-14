# g2-policy-protocol

`g2-policy-protocol` 是 G2 机器人和模型服务之间的共享协议与数据语义库。它定义线上 `g2.policy.v3` 消息、客户端、服务端、机器人状态/动作语义，以及训练和部署共用的通用数据处理。

模型仓库只负责模型输入、归一化和 checkpoint；机器人仓库只负责采集、调度、安全和硬件执行。协议库不包含 Torch、JAX、OpenPI、LeRobot 或 PI0.5 专用代码。

## 从哪里开始

| 目标 | 先读 | 然后看 |
|---|---|---|
| 开发一个客户端或服务端 | [开发入口](docs/developing.md) | [最小客户端](examples/minimal_client.py)、[最小服务端](examples/minimal_server.py) |
| 自定义线上消息或排查兼容性 | [通信协议](docs/protocol.md) | `src/g2_policy_protocol/types.py`、`protocol.py` |
| 配置状态、动作、历史和坐标 | [Policy 配置](docs/policy-config.md) | `contracts/policy.py`、`contracts/dataset.py` |
| 安装到 wholebody、OpenPI 或 LeRobot | [安装与兼容性](docs/install.md) | 对应仓库的 bootstrap 和部署文档 |
| 修改共享实现 | [开发入口](docs/developing.md) | `runtime/client.py`、`runtime/server.py`、`processing/encoding.py` |

## 三分钟跑通

```bash
python -m pip install -e .
python examples/minimal_server.py --port 8000
# 另一个终端
python examples/minimal_client.py --url ws://127.0.0.1:8000
```

客户端会收到形状为 `(2, 20)` 的双臂绝对位姿目标。示例使用合成数据，不加载模型、不连接机器人，也不执行动作。`--jpeg` 和 `--encoding json` 可以分别检查图像编码和消息编码路径。

## 一次交互

```text
client                         server
  │── 建立 WebSocket ───────────▶│
  │◀─ metadata（服务声明）───────│  server 先发 metadata
  │   校验声明                    │
  │── obs(request_id=N) ────────▶│  解析 → 解码 → backend.infer
  │◀─ action / error（同一个 N）──│  校验结果 → 回包
  │── reset（可选）─────────────▶│  清理本连接 session
  │◀─ reset_ack / error ─────────│
```

协议没有额外的握手确认消息。推理和 reset 不自动重发；连接尝试可以在超时内重试。服务端在单独的模型工作线程中解码图像和执行推理，网络事件循环只负责收发。

## 当前源码结构

```text
src/g2_policy_protocol/
├── types.py              # 字段、固定向量布局、单位、历史和会话语义
├── protocol.py           # metadata、obs、action、error、reset 的构造与校验
├── capabilities.py       # 已解析 policy → metadata 能力声明
├── validation.py         # 通用标量、字段和数组校验
├── constraints.py        # 标准库实现的刚体变换约束
├── diagnostics.py        # 合成样本、服务检查和 g2-policy-check
├── contracts/
│   ├── policy.py         # policy 类型、默认值、语义查询和组合校验
│   └── dataset.py        # 磁盘列名、向量布局、相机键和数据集检查
├── runtime/
│   ├── client.py         # 握手、请求、响应校验、超时、取消和 reset
│   ├── server.py         # WebSocket、session、模型线程和错误分类
│   ├── codec.py          # MessagePack / JSON 帧编解码
│   └── images.py         # RGB 准备、JPEG 解码和历史图像解码
└── processing/
    ├── encoding.py       # encode_state、ActionCodec、动作/状态维度
    ├── geometry.py       # 旋转、位姿和坐标系转换
    ├── temporal_dataset.py # 按时间采样和有界统计读取
    ├── dataset_sources.py  # 数据源类型、路径和权重
    ├── mixture.py        # episode 重复计划和虚拟索引
    ├── statistics.py     # 流式归一化统计
    └── fingerprints.py   # 数据内容和共享源码摘要
```

依赖方向保持单向：

```text
types / protocol / runtime
          ↑
      contracts
          ↑
      processing
          ↑
  OpenPI / LeRobot adapters
```

通信核心不导入 `contracts` 或 `processing`。`processing` 可以使用契约；模型适配器决定如何把 canonical 数据变成具体模型输入。PI0.5 的布局标识、历史图像拼接和模型维度补齐位于 OpenPI/LeRobot 各自的 `adapters/g2/pi05_inputs.py`。

## 四个协议阅读入口

1. `types.py`：传递哪些字段、形状、单位和默认语义。
2. `protocol.py`：每种消息如何构造、解析和拒绝。
3. `runtime/client.py`：客户端如何握手、准备观测、检查响应和处理断线。
4. `runtime/server.py`：服务端如何声明能力、建立 session、调用 backend 和回包。

backend 最小接口是：

```python
from g2_policy_protocol.types import InferenceResult, Observation

class Backend:
    def infer(self, observation: Observation) -> InferenceResult:
        # observation.images 已是 RGB uint8；actions 必须是完整双臂绝对目标
        return {"actions": model_predict(observation)}
```

## 安装与验证

要求 Python 3.10+。源码开发：

```bash
python -m pip install -e . pytest
python -m pytest -q
```

跨仓库联调在四个仓库的共同父目录执行：

```bash
python g2-policy-protocol/tools/check_stack.py \
  --stack-root /home/zijianwang \
  --lerobot-python /home/zijianwang/.venvs/g2-policy-test/bin/python \
  --openpi-python /home/zijianwang/openpi/.venv/bin/python
```

命令覆盖两个 backend、pose/joint、history 1/2、raw/JPEG 和 JSON/MessagePack。它只验证协议和适配流程，不代表模型质量或真机安全。

完整安装、editable/wheel 选择和版本兼容规则见[安装文档](docs/install.md)。
