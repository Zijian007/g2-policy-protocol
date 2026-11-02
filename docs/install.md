# 安装

[README](../README.md) · [通信协议](protocol.md) · [policy 配置](policy-config.md)

协议包是纯 Python，没有编译步骤。每台机器把本仓库 clone 在模型/机器人仓库**旁边**，以 editable 方式装进各自的 venv。部署时两端各自 `git pull` 即可，不需要人工核对版本：不兼容会在握手或请求校验时报错（见[兼容性如何保证](#兼容性如何保证)）。要求 Python ≥3.10 和 [uv](https://docs.astral.sh/uv/)；运行依赖只有 NumPy、MessagePack、SciPy、websockets、Pillow。

```text
~/g2-policy-protocol/
~/wholebody/  或  ~/openpi/ ~/lerobot-finetune/
```

| 机器 | 装进哪个环境 | 怎么装 |
|---|---|---|
| 域控（G2 真机） | wholebody 启动脚本使用的 venv（真机为 `/home/agi/.venv`） | `uv pip install -e` |
| GPU 服务器 | openpi / lerobot-finetune 的 `.venv` | 各自 bootstrap 自动 editable 安装 |

editable 安装后，`git pull` / `git checkout` 协议仓库即生效，无需重装；只有协议依赖变化时才需要重跑安装。

## 域控侧

只装协议包，不要装 OpenPI、Torch 或 JAX：

```bash
cd ~ && git clone git@github.com:Zijian007/g2-policy-protocol.git   # 首次；之后 git pull
uv pip install --python /home/agi/.venv/bin/python -e ~/g2-policy-protocol
```

解释器换成 wholebody launcher 实际使用的那个。推理入口为 wholebody 的 `launcher/real/start_{openpi,lerobot}_pi05_infer.sh`，连接地址和编码在推理 YAML 中配置，见 [wholebody policy 通信契约](../../wholebody/docs/dev/policy-protocol.md)。

## GPU 服务器侧

GPU 驱动、Git LFS、FFmpeg 等见模型仓库文档。协议包由 bootstrap 从相邻的 `../g2-policy-protocol` 以 editable 方式安装：

```bash
cd ~/openpi && bash scripts/g2/bootstrap.sh           # OpenPI（JAX π₀.₅）
cd ~/lerobot-finetune && bash scripts/bootstrap.sh    # LeRobot（ACT / π₀.₅ 等）
```

完整环境说明：[OpenPI](../../openpi/scripts/g2/docs/environment.md) · [LeRobot](../../lerobot-finetune/README.md)。

## 验证安装

用**实际启动服务或客户端的解释器**确认导入的是相邻源码：

```bash
"$PY" -c "import g2_policy_protocol as g; print(g.__file__)"   # 应为 ~/g2-policy-protocol/src/...
```

## 兼容性如何保证

两端不需要逐字节相同，只需要线上契约相同：

- **结构变化**（增删字段、改枚举、改动作布局、改字段表示）：握手或请求的严格校验会拒绝不兼容格式，新校验器在格式不匹配时提示 `the client and server may run different g2-policy-protocol versions`；旧校验器可能只有格式错误。按提示更新落后的一端，重启后再连。
- **语义变化**（坐标约定、单位、夹爪方向、字段含义等结构不变但意义改变的改动）：严格校验发现不了，因此**必须把 `_constants.py` 的 `PROTOCOL` 升一版**（如 `g2.policy.v4`），握手会拒绝版本不同的另一端。
- **内部改动**（修 bug、重构、数据层工具）：两端可以各自更新。

更新时先更新并重启 GPU server，再启动域控 client。

JPEG 图像历史的格式已从 raw 数组改为 JPEG bytes 列表，`g2.policy.v3` 不变；混用新旧编码实现会在首个相关请求中失败，此编码格式变更不影响 raw 或无图像历史的请求。需要同时更新协议包、wholebody 客户端和对应模型服务的适配代码，再重启服务。

有图像历史时，客户端还会省略重复的 `images`，适用于 raw/JPEG。新服务端接受旧客户端仍发送 `images` 的请求；新客户端连接仍要求 `images` 的旧协议实现时，首个历史请求会报缺少字段。先更新 GPU 侧协议包及支持图像历史的 backend，再更新客户端。第三方 backend 需按 [接入模型示例](protocol.md#接入模型server-侧) 优先读取历史，不能假设每个请求都有 `images`。

## 独立开发环境

只跑协议测试、示例和 `g2-policy-check` 时，在本仓库根目录：

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e . pytest
.venv/bin/python -m pytest
```

## 无 git 权限时：wheel

给拿不到仓库的第三方交付时才用 wheel。在要交付的提交上构建：

```bash
cd ~/g2-policy-protocol && OUT="$(mktemp -d)" && uv build --out-dir "$OUT" && sha256sum "$OUT"/*.whl
```

接收方把 wheel 放进单独目录：

```bash
uv pip install --python "$PY" --no-deps --reinstall /path/to/g2_policy_protocol-*.whl   # 域控或独立环境
G2_PROTOCOL_WHEELS=/path/to/wheels bash scripts/g2/bootstrap.sh                         # OpenPI
G2_PROTOCOL_WHEELS=/path/to/wheels bash scripts/bootstrap.sh                            # LeRobot
```

同版本 wheel 更新必须 `--reinstall`。

## 常见问题

| 现象 | 处理 |
|---|---|
| `No module named 'g2_policy_protocol'` | 装到了另一个解释器；用实际启动脚本的解释器安装和验证 |
| `g.__file__` 指向 `site-packages` | 装的是拷贝（旧 bootstrap 或 wheel），改源码不生效；重跑 bootstrap，或 `uv pip install --python "$PY" --no-deps --reinstall -e ~/g2-policy-protocol` |
| 找不到 `../g2-policy-protocol` | 协议仓库没有 checkout 在模型仓库旁边 |
| 握手报错末尾带 `may run different g2-policy-protocol versions` | 两端协议不兼容；两端 `git pull` 到最新，先重启 server |
| `uv` 找不到 | 安装 uv 后 `source "$HOME/.local/bin/env"`，或用绝对路径 |
