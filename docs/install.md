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

## G2 工作区

[统一安装器](../../g2_wzj_docs/ops/setup/environment.md#3-安装与更新)将协议包可编辑安装到 `AGI_PYTHON`。
域控侧的 Wholebody 使用这份环境。

## 独立项目

从本仓库根目录执行。用目标项目的 Python 环境，开发时带 `[dev]`，运行时只需 `-e .`。

```bash
python -m pip install -e '.[dev]'
python -I -c 'import g2_policy_protocol; print(g2_policy_protocol.__file__)'
python -m pytest -q
```

可编辑安装的导入路径应落在本仓库的 `src/`。修改 Python 源码后重启调用进程，
依赖或包元数据变化时重新安装。不需要 GPU、模型或机器人。

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

## 指定解释器

可使用 `/path/to/env/bin/python -m pip` 指定独立环境。由 uv 管理的环境可用
`uv pip install --python /path/to/env/bin/python -e '.[dev]'`。

## 接入应用

- Wholebody 使用工作区统一安装后的协议源码。推理入口与传输配置见 [Wholebody 协议说明](../../wholebody/docs/dev/policy-protocol.md)。
- OpenPI 和 LeRobot 由各自的 bootstrap 把相邻的 `../g2-policy-protocol` 可编辑安装到各自环境。
  离线部署也可以向 bootstrap 传入 `G2_PROTOCOL_WHEELS`。 [OpenPI 环境](../../openpi/scripts/g2/docs/environment.md) · [LeRobot 安装](../../lerobot-finetune/README.md#安装)

用实际启动服务或应用的解释器执行上面的导入检查，确认它读取的是预期 checkout 或 wheel。

## 兼容性如何保证

两端的 Python 源码可以不同，但连接两端必须理解同一个 `g2.policy.v3` 线上契约。

- **结构变化**：新增/删除字段、枚举、动作维度或图像表示会在握手或请求校验阶段失败，并提示两端可能使用了不同版本。
- **语义变化**：坐标系、单位、夹爪方向、历史顺序或字段含义发生变化时，必须升级 `src/g2_policy_protocol/types.py` 中的 `PROTOCOL`，例如改为 `g2.policy.v4`。
- **内部变化**：修复实现、整理文件或修改训练辅助算法，只需同步有实际依赖的一端。

当前图像历史规则是：当 metadata 声明 `context.image_history` 时，历史从旧到新排列，最后一帧是当前帧；raw 使用 `uint8 (L,H,W,3)`，JPEG 使用每帧一个 bytes 的列表。客户端可以省略重复的 `images`，服务端运行时把历史解码后交给 `Observation.images`。详情以[通信协议](protocol.md#数据约定)为准。

推荐更新顺序：先更新并重启 GPU server，再更新域控 client。升级后用 `g2-policy-check` 和 `tools/check_stack.py` 验证，最后再接入真实控制环。

## 常见问题

| 现象 | 处理 |
|---|---|
| `No module named 'g2_policy_protocol'` | 装到了另一个解释器；用实际启动脚本的解释器安装和验证 |
| 源码修改后未生效 | 检查实际解释器和导入路径；普通安装或 wheel 需重装，可编辑安装需重启调用进程 |
| 找不到 `../g2-policy-protocol` | 模型 bootstrap 使用相邻 checkout；将协议仓库放在模型仓库旁边，或用该脚本支持的 wheel 入口 |
| 握手报错末尾带 `may run different g2-policy-protocol versions` | 两端协议不兼容；两端 `git pull` 到最新，先重启 server |
| `No module named pip` | 使用带 pip 的独立环境，或在 uv 管理的应用环境中使用 `uv pip --python` |
