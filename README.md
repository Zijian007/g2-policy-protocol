# g2-policy-protocol

连接 G2 域控（wholebody）与 GPU 模型服务（OpenPI、LeRobot 或第三方）的共享 Python 包：

- **通信层**（`g2_policy_protocol`）：`g2.policy.v3` 消息构造与校验、MessagePack/JSON 编解码、WebSocket `PolicyServer`。
- **数据层**（`g2_policy_protocol.data`）：两个训练框架共用的数据契约、policy 语义、状态/动作编码和时序采样，保证训练与部署语义一致。

不依赖 Torch、JAX、LeRobot 或机器人运行时。

```text
域控（G2 真机）                                    GPU 服务器
  wholebody                                          openpi / lerobot-finetune
    采集 · 调度 · 安全门 · HAL 执行                    训练 · 推理 · backend
    G2WebSocketPolicyClient ──── obs ─────────▶      PolicyServer
                            ◀── action / error ──
  g2-policy-protocol（通信层）                       g2-policy-protocol（通信层 + data）
                     WebSocket · g2.policy.v3 · MessagePack 或 JSON
```

两端以 editable 方式安装相邻 clone 的本仓库，各自 `git pull` 更新；不兼容会在握手时报错，改变线上语义时须升级 `PROTOCOL` 版本，见[兼容性如何保证](docs/install.md#兼容性如何保证)。

## 文档

| 文档 | 内容 |
|---|---|
| [安装](docs/install.md) | 域控侧、GPU 服务器侧安装，兼容性规则 |
| [通信协议](docs/protocol.md) | 消息、握手、控制量、context、RTC、接入模型、验收工具、常见错误 |
| [policy 配置](docs/policy-config.md) | 训练 YAML 中 `policy` 字段的语义（状态/动作、坐标、差分、采样、数据集） |
| [CHANGELOG](CHANGELOG.md) | 版本变更 |

训练 YAML 和命令由模型仓库维护：[OpenPI](../openpi/scripts/g2/docs/training.md) · [LeRobot](../lerobot-finetune/scripts/g2/docs/training.md)。跨仓库链接要求各仓库位于同一父目录。

## 职责划分

| 层 | 归属 |
|---|---|
| 模型与训练 | 各模型仓库：训练 YAML、配置版本、模型加载、checkpoint；把 canonical 请求转成模型输入，再把输出还原为完整双臂绝对动作 |
| 数据与几何 | `data.contract`（列名、向量布局、相机键）、`data.policy`（`resolve_policy`、`policy_capabilities` 及共享查询）、`data.geometry`（旋转/位姿/坐标系转换）、`data.encoding`（`encode_state`、`ActionCodec`）、`data.dataset`（`validate_dataset_contract`、`TemporalDataset`）、`data.sources`（多数据集混合） |
| 通信 | `protocol`、`codec`、`server`；固定布局与枚举只在 `_constants.py` 定义一次 |
| 采集与执行 | wholebody：保存原始 state/command 并导出 LeRobot 数据；按握手准备观测、调度动作队列，经安全门和 HAL 执行 |

协议不传输模型内部设置（归一化、token、LoRA、padded action_dim 等）。

## 开发

```text
src/g2_policy_protocol/
  _constants.py      固定向量布局、相机、枚举、消息类型（仅标准库）
  protocol.py        公开 API：按握手构造/解析消息，校验形状、有限值与能力
  _messages.py       每种消息的静态结构：字段、类型、枚举、范围
  validation.py      通用数值检查与刚体变换校验
  codec.py           MessagePack / JSON 编解码、连接 URL 的编码参数
  images.py          缩放（letterbox / 拉伸）与 JPEG 编解码
  server.py          WebSocket PolicyServer：握手、错误码、单模型线程
  diagnostics/       验收工具，不参与线上运行
    samples.py       观测样本的生成、保存与读取
    check_server.py  g2-policy-check 命令
  data/              训练与部署共用的数据语义
    contract.py      LeRobot 磁盘列名、向量布局、相机键
    policy.py        policy 配置的默认值、校验、共享查询与线上能力推导
    geometry.py      旋转 / 位姿 / 坐标系转换
    encoding.py      ActionCodec、encode_state：线上值 ↔ 模型空间
    dataset.py       数据集契约检查与按时间采样
    sources.py       多数据集描述与按 episode 混合
```

```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -e . pytest
.venv/bin/python -m pytest          # 约 1 秒，无需 GPU/模型/机器人
```

改布局或枚举只改 `_constants.py`；新增消息字段时同步改 `_messages.py` 的字段表与检查函数、[通信协议](docs/protocol.md) 和测试。"某个 policy 需要哪些列 / context"只由 `data.policy` 的查询函数（`uses_kind`、`command_reference_kinds`、`uses_arm_base`）回答，训练和部署都调用它们。**结构不变但线上含义改变时，必须升级 `_constants.PROTOCOL`**，并记入 CHANGELOG。

### 跨仓库验证

用真实 wholebody client 与 LeRobot / OpenPI backend 联调（替换模型数值输出，不加载权重、不用 GPU/HAL），覆盖两种编码、pose/joint、history 1/2 和 reset：

```bash
# 在四个仓库的共同父目录；各仓库需已有自己的 .venv
lerobot-finetune/.venv/bin/python g2-policy-protocol/tools/check_stack.py --stack-root "$PWD"
```

输出的 `stack_manifest` 记录各仓库提交和源码摘要。这不等同于真机轨迹验收。
