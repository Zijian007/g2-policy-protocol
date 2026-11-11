# 安装

[README](../README.md) · [通信协议](protocol.md) · [policy 配置](policy-config.md)

本包可独立安装，不依赖 Torch、JAX、模型仓库或机器人运行时。
要求 Python 3.10+；运行依赖为 NumPy、MessagePack、SciPy、websockets、Pillow。
协议代码是纯 Python，无需 C++ 编译；第三方依赖优先使用匹配平台的预编译 wheel。

以下命令均从本仓库根目录执行，`python` 指向实际使用本包的环境。
已有应用环境时直接使用它；独立使用时可在仓库外创建环境：

```bash
python3 -m venv ~/.venvs/robot-dev
source ~/.venvs/robot-dev/bin/activate
```

## 普通安装

```bash
python -m pip install .
python -c "import g2_policy_protocol; print(g2_policy_protocol.__file__)"
```

## 源码开发

```bash
python -m pip install -e . pytest
```

本包的 pytest 声明在开发 dependency group 中，没有 `[dev]` extra，因此上面显式安装 pytest。
可编辑安装直接读取 Python 源码，修改或 `git pull` 后重启调用进程即可。
依赖或包元数据变化时重新安装。普通安装和 wheel 安装需要重新安装才能更新源码副本。
开发测试按变更范围执行 `python -m pytest`，无需 GPU、模型或机器人。

## wheel 构建与安装

```bash
python -m pip wheel --no-deps --wheel-dir dist .
python -m pip install /path/to/wheels/g2_policy_protocol-*.whl
```

安装目录只保留待安装的一个目标 wheel。构建时默认隔离安装
`pyproject.toml` 声明的构建依赖；接收方无需本仓库源码。
离线安装需准备全部依赖 wheel，使用
`python -m pip install --no-index --find-links /path/to/wheels g2-policy-protocol`。

当前包版本固定为 0.7.0。同版本 wheel 更新时，在已满足依赖的环境中执行
`python -m pip install --force-reinstall --no-deps /path/to/wheels/g2_policy_protocol-*.whl`。
线上兼容性由协议契约控制，见 [兼容性如何保证](#兼容性如何保证)。

## 指定环境与卸载

未激活环境时，用 `/path/to/env/bin/python -m pip ...` 指定目标解释器。
uv 管理且未提供 pip 的环境，可用 `uv pip install --python /path/to/env/bin/python .`
或 `uv pip install --python /path/to/env/bin/python -e . pytest`；uv 不是独立安装本包的前提。

```bash
python -m pip uninstall g2-policy-protocol
```

卸载只移除当前环境中的包或可编辑链接，不删除仓库和本地构建产物。

## 接入现有应用

现有模型 bootstrap 使用相邻 checkout，通常以可编辑方式安装：

```text
~/g2-policy-protocol/
~/wholebody/  或  ~/openpi/ ~/lerobot-finetune/
```

相邻目录是这些应用脚本的约定，不是协议包本身的安装限制。

| 机器 | 目标环境 | 应用接入方式 |
|---|---|---|
| 域控（G2 真机） | wholebody 启动脚本实际使用的环境 | 安装普通包、可编辑源码或 wheel |
| GPU 服务器 | openpi / lerobot-finetune 的环境 | 各自 bootstrap 默认安装相邻协议源码 |

### 域控侧

只装协议包，不要装 OpenPI、Torch 或 JAX：

```bash
git clone git@github.com:Zijian007/g2-policy-protocol.git
/path/to/env/bin/python -m pip install -e ./g2-policy-protocol
```

首次在应用仓库的父目录 clone；已有 checkout 时直接更新源码。
解释器路径换成 wholebody launcher 实际使用的那个。
推理入口为 wholebody 的 `launcher/real/start_{openpi,lerobot}_pi05_infer.sh`，连接地址和编码在推理 YAML 中配置，见 [wholebody policy 通信契约](../../wholebody/docs/dev/policy-protocol.md)。

### GPU 服务器侧

GPU 驱动、Git LFS、FFmpeg 等见模型仓库文档。协议包由 bootstrap 从相邻的 `../g2-policy-protocol` 以 editable 方式安装：

```bash
# 在 OpenPI 根目录执行
bash scripts/g2/bootstrap.sh
# 或在 LeRobot 根目录执行
bash scripts/bootstrap.sh
```

完整环境说明：[OpenPI](../../openpi/scripts/g2/docs/environment.md) · [LeRobot](../../lerobot-finetune/README.md)。

### 验证应用环境

用实际启动服务或客户端的解释器确认导入位置：

```bash
/path/to/env/bin/python -c "import g2_policy_protocol as g; print(g.__file__)"
```

可编辑安装应指向 checkout 的 `src/`；普通安装或 wheel 安装应指向目标环境的 `site-packages`。

使用 wheel 接入模型 bootstrap 时，将协议 wheel 目录传给应用脚本：

```bash
G2_PROTOCOL_WHEELS=/path/to/wheels bash scripts/g2/bootstrap.sh  # OpenPI 根目录
G2_PROTOCOL_WHEELS=/path/to/wheels bash scripts/bootstrap.sh     # LeRobot 根目录
```

## 兼容性如何保证

两端不需要逐字节相同，只需要线上契约相同：

- **结构变化**（增删字段、改枚举、改动作布局、改字段表示）：握手或请求的严格校验会拒绝不兼容格式，新校验器在格式不匹配时提示 `the client and server may run different g2-policy-protocol versions`；旧校验器可能只有格式错误。按提示更新落后的一端，重启后再连。
- **语义变化**（坐标约定、单位、夹爪方向、字段含义等结构不变但意义改变的改动）：严格校验发现不了，因此**必须把 `_constants.py` 的 `PROTOCOL` 升一版**（如 `g2.policy.v4`），握手会拒绝版本不同的另一端。
- **内部改动**（修 bug、重构、数据层工具）：两端可以各自更新。

更新时先更新并重启 GPU server，再启动域控 client。

JPEG 图像历史的格式已从 raw 数组改为 JPEG bytes 列表，`g2.policy.v3` 不变；混用新旧编码实现会在首个相关请求中失败，此编码格式变更不影响 raw 或无图像历史的请求。需要同时更新协议包、wholebody 客户端和对应模型服务的适配代码，再重启服务。

有图像历史时，客户端还会省略重复的 `images`，适用于 raw/JPEG。新服务端接受旧客户端仍发送 `images` 的请求；新客户端连接仍要求 `images` 的旧协议实现时，首个历史请求会报缺少字段。先更新 GPU 侧协议包及支持图像历史的 backend，再更新客户端。第三方 backend 需按 [接入模型示例](protocol.md#接入模型server-侧) 优先读取历史，不能假设每个请求都有 `images`。

## 常见问题

| 现象 | 处理 |
|---|---|
| `No module named 'g2_policy_protocol'` | 装到了另一个解释器；用实际启动脚本的解释器安装和验证 |
| 源码修改后未生效 | 检查实际解释器和导入路径；普通安装或 wheel 需重装，可编辑安装需重启调用进程 |
| 找不到 `../g2-policy-protocol` | 模型 bootstrap 使用相邻 checkout；将协议仓库放在模型仓库旁边，或用该脚本支持的 wheel 入口 |
| 握手报错末尾带 `may run different g2-policy-protocol versions` | 两端协议不兼容；两端 `git pull` 到最新，先重启 server |
| `No module named pip` | 使用带 pip 的独立环境，或在 uv 管理的应用环境中使用 `uv pip --python` |
