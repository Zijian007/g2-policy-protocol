# 开发维护

[文档入口](../README.md#文档导航) · [安装与接入](guide.md) · [通信协议](protocol.md) · [policy 配置](policy-config.md)

本页集中说明职责、数据边界、源码位置，以及修改、验证和交付方法。安装和接入命令见[安装与接入](guide.md)。

## 职责划分

各模型保留自己的训练配置、权重格式、网络、损失和推理算法。服务端适配器把这些差异封装在 `PolicyBackend` 后面；机器人端按动作控制量、观测需求和已绑定的执行器处理请求，不读取模型名称。

| 层 | 归属与接口 |
| --- | --- |
| 模型与训练 | 各 policy 仓库维护训练 YAML、配置版本、字段映射、模型加载和 checkpoint 快照 |
| G2 数据与几何 | `g2_policy_protocol.data.contract`、`encoding.ActionCodec`、`dataset.TemporalDataset`；无需导入 OpenPI、wholebody、PyTorch 或 JAX |
| 通信 | `g2_policy_protocol`：版本化 schema、生成校验、JSON/MessagePack、`server.PolicyServer` 和 `PolicyBackend` 接口 |
| 模型适配 | 各 policy 实现 canonical request → 模型观测 → canonical actions；OpenPI 用 `OpenPIBackend`，训练转换也使用同一个 `ActionCodec` |
| 采集与机器人执行 | wholebody 保存原始 state/command、连续段和采样时间并导出标准列；按握手准备观测、调度时间队列，经安全门和 HAL 执行 |

一个顶级 Python 包 `g2_policy_protocol` 提供通信接口和 `data` 子包。框架适配器保留在各自仓库，通信层不导入数据层、模型框架或机器人运行时。安装与版本要求见[安装与接入](guide.md#安装与版本)。

训练 YAML 的 `experiment / data / policy` 语义共用；模型超参、优化器和 checkpoint 由各自框架维护。

模型内部可以预测 delta、SE(3)、不同旋转表示、单臂或补齐维度；适配器必须还原到声明的完整双臂绝对控制量。协议不会传输 Pi05 token 设置、归一化参数、LoRA 或 padded action_dim=32。

`data.policy.resolve_policy` 在入口解析并校验 policy 结构和数值约束；状态和动作编码入口检查数组形状及有限值，内部坐标转换不重复扫描。单臂先裁切再转换，解码后补齐另一臂的实测位置；速度和力矩补零。共享包不定义训练 YAML 或配置版本。OpenPI 的字段翻译与快照位于 `openpi.adapters.g2.training_config`，LeRobot 的字段翻译位于 `lerobot_finetune.adapters.g2.config`；它们只将已解析的 `policy` 子树交给共享数据工具。

## 数据模块

| 模块 | 主要接口与职责 |
|---|---|
| `data.contract` | 固定列名、向量布局和相机键；仅依赖标准库 |
| `data.policy` | `resolve_policy` 解析语义；`policy_capabilities` 推导线上能力 |
| `data.encoding` | `encode_state`、`ActionCodec`；坐标/旋转转换与状态/动作编码 |
| `data.dataset` | `validate_dataset_contract`、`TemporalDataset`、`references_from_sample`、`action_label_column` |

导入路径均以 `g2_policy_protocol` 开头。依赖方向为 `dataset → encoding → policy`；能力推导使用通信层；整个包不导入模型框架或机器人运行时。

`TemporalDataset` 的 `source_fps` / `sample_hz` 分别表示源数据帧率和目标采样频率；`action_label_column()` 返回训练动作标签的磁盘列。完整源码树与定义来源见[开发维护](development.md#源码组织与定义来源)，policy 候选值与坐标语义见[配置参考](policy-config.md)。

## 连续段与监督

当前 wholebody 的 v2.1 / v3 导出把一次开录到停录的 raw 记录保留为一个 episode。多个连续 segment 分别重采样，再按顺序合入同一个 episode；暂停区间不补帧。导出不写 `segment_policy`，也不写 `source_segment_id`；保留原始记录来源，整集提交，中断后按同一录制重试和去重。具体规则见 [wholebody 导出文档](../../wholebody/docs/user/collect/export.md)。

新导出固定 `action_supervision: command`，不把 state 重命名成 action。测量监督由 `policy.action.pose.source: state` 或 `policy.action.joint.source: state` 选择。旧 `action_supervision: state` 导出配置会在入口被拒绝。

非空数据集的导出契约必须一致。当前导出可追加到契约一致、未声明 `segment_policy` 的既有数据集；遇到已按 `separate_episodes` 生成的库会拒绝追加，需要新的输出目录。不要修改 info.json 假装已迁移，也不要把两种 episode 边界写进同一数据集。历史原始记录和既有数据不自动重写。

两套训练入口调用 `validate_dataset_contract`：检查所需列、声明的向量维度、有效 fps、已声明的 base_link/flange/rot6d、command 监督一致性，以及显式标注的 concatenated 段策略。保留对历史数据缺省元信息的兼容；缺省不能证明坐标或段边界正确，需要先核对来源。此工具不猜测或补写历史元信息。

训练入口会拒绝显式声明 `segment_policy: concatenated` 的数据，但不会从缺省字段识别暂停拼接点。`TemporalDataset` 只按 episode 边界裁切，时序窗口可能跨越同一 episode 内的片段连接处。使用多片段录制训练前，应核对并准备所需的连续 episode；契约校验通过不代表时序连续。

## 预测块与执行预算

H 为模型预测长度，R 为响应帧数，K 为客户端提交给执行队列的步数。两套第一方 v3 服务声明 `extensions.response_mode: full_chunk`，固定 R=H；公共响应入口拒绝截短输出。RTC 的历史前缀允许短于 H，不受完整响应约束。

K 只由 wholebody 的 `infer.execution_steps` 决定，满足 K≤R。服务端不截取 execution_steps，也不把每次推理调用当成一拍。第三方服务可独立声明自己的响应模式。

`replan_after_steps` 表示从预测起点经过多少步后请求下一段；例如 H=50、K=20、重规划步数为 10 时，经过 10 步补段，最多提交 20 步。wholebody 保存完整预测，只向执行/RTC 队列提交 K 帧，过期前缀只裁切一次。

LeRobot G2 部署配置删除 `inference.execution_steps`；`temporal_ensemble_coeff` 非空会报错，原因是单步输出尚无目标时间对齐。模型原生动作索引必须连续并从当前观测开始；其他时间布局需先提供专属输出时间适配，离线预测和训练的原生能力不因此增加。

状态历史、图像历史、episode 起点和 command/object 参考由 wholebody 的采样时间提供。OpenPI 不再根据不均匀的推理调用缓存历史或猜测 episode 起点。两种服务共用 policy_capabilities 推导需求；缺少必要 context 会在请求边界报错。

pose_target / joint_position 可由现有 wholebody 执行器处理。关节速度、力矩仅有协议/训练语义，应用没有绑定执行器时启动前拒绝。LeRobot 尚不支持 RTC；OpenPI 保留自身 RTC 实现。

## 源码组织与定义来源

```text
src/g2_policy_protocol/
  g2.schema.json     v3 JSON Schema（消息、能力、向量布局和图像）
  _generated.py      由 Schema 生成的常量与静态校验，禁止手改
  _validation.py     生成代码复用的轻量类型/范围检查
  protocol.py        消息构造、解析和按能力检查的动态校验
  codec.py           JSON / MessagePack；NumPy / Base64
  validation.py      刚体变换校验
  server.py          WebSocket 边界与模型工作线程
  images.py          JPEG 编解码
  samples.py         验收用合成 / 真实观测样本
  check_server.py    独立服务验收命令
  data/
    contract.py      固定列名、向量布局和相机键
    policy.py        policy 解析、语义校验和能力推导
    encoding.py      状态/动作编码与坐标转换
    dataset.py       数据契约检查与时序采样
```

结构源是 [`g2.schema.json`](../src/g2_policy_protocol/g2.schema.json)，`x-g2` 记录固定向量布局，`action_spec` 声明控制量。`tools/generate_protocol.py` 生成并检查 `_generated.py`。

JSON Schema 校验只用于开发测试；运行时不读取 Schema 文件，不逐帧执行完整 Schema 验证。

静态规则由 [`tools/generate_protocol.py`](../tools/generate_protocol.py) 在开发阶段编译成普通 Python：字段、必填项、枚举、类型、范围、集合长度与唯一性，以及计时/错误文本检查。NumPy 向量维度和原始图像规格也从 Schema 提取，运行时不遍历像素或逐元素解释 Schema。

`protocol.py` 保留数组归一化/有限值、非空白 prompt、能力匹配、RTC 和 request_id 对应关系等动态检查；`codec.py` 保留编解码和数据完整性检查。

## 修改与验证

生成器只支持当前使用的静态关键字，遇到新的不支持规则会报错。修改 Schema 后必须重新生成并运行测试；测试也检查修改边界、枚举或必填字段会改变生成代码的行为。

完整测试另需 `pytest`、`jsonschema`。先在开发环境安装：

```bash
python -m pip install -e . pytest jsonschema
```

在本仓库根目录执行生成和验证：

```bash
python tools/generate_protocol.py
python tools/generate_protocol.py --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

没有真实权重/GPU 的回环测试只验证协议和模型边界，不代表模型动作质量。

## 跨仓库验证

开发使用相邻源码安装；跨机部署应构建一次 wheel，复制同一个文件，记录其 SHA256，并记录四个仓库的提交和未提交变更。相同版本号不能证明两份 editable 源码一致。测试输出中的 stack_manifest 包含提交、分支和源码变更摘要；有未提交变更时还需保存对应 patch 和新文件。

```bash
# 在四个仓库的共同父目录，客户端使用 LeRobot 环境；worker 自动选各自环境。
lerobot-finetune/.venv/bin/python g2-policy-protocol/tools/check_stack.py --stack-root "$PWD"
```

脚本用真实 wholebody 客户端、共享服务边界、LeRobot G2Backend 和 OpenPI G2Policy/OpenPIBackend，覆盖两种编码、pose/joint、L=1/2、显式参考、H=R=4/K=2 和 reset。只替换模型数值输出，不加载权重、不访问 GPU/HAL，不等同于真机轨迹验收。

wholebody 的 test_export_segments 覆盖真实 JPEG/ffmpeg/v2.1/v3 写盘、暂停区间不插值、整集去重、导出中断重试和拒绝混写 separate_episodes 库；共享 test_pipeline_contract 覆盖数据契约和短响应拒绝。训练恢复由 OpenPI scripts/train_test.py 的真实小模型测试验证；运行时设 JAX_PLATFORMS=cpu，以免被服务器 GPU 数影响测试批量。

## 打包与跨端升级

打包环境需已安装 setuptools 和 wheel。验证通过后执行：

```bash
python -m pip wheel --no-index --no-deps --no-build-isolation . -w dist
```

当前源码构建产物为 `g2_policy_protocol-0.6.0-py3-none-any.whl`。发布新版本时同步更新版本声明、调用方依赖和相关文档；涉及不兼容契约变更时，明确通信协议的兼容边界。当前源码只接受 `g2.policy.v3`，升级时同步更新客户端和服务端。

Python API 统一使用 `g2_policy_protocol.data`，并将图像工具移到 `g2_policy_protocol.images`；旧导入路径不保留。升级时同步更新调用方并重新构建 wheel，不能复用旧的同版本文件。本次同时移除 v2、版本选择参数和握手中的 `execution_steps`；`build_response` / `parse_response` 必须传入 metadata，动作布局用 `build_action_spec` 构造。磁盘列格式沿用现有契约。

部署时：

1. 为域控和 GPU 环境分发同一版本的源码或同一个 wheel。
2. 同步更新 wholebody、lerobot-finetune / OpenPI 中相应的接入代码和依赖。
3. 两端安装同一版本协议包。
4. 重启 GPU server 后再启动域控 client，验证能力握手和观测/动作往返。

不能只替换一端。OpenPI 非 G2 机器人的通用原生 server 不受本协议升级影响。

## 对外交付

wheel 的安装方式见[安装与接入](guide.md#安装与版本)。wheel 包含 `g2-policy-check` 命令及其模块；不包含 Markdown 文档和 examples 源码。使用当前源码重新构建 wheel，旧的同版本 wheel 没有新工具。

需要完整文档与示例时，从已提交的源码打包：

```bash
git archive --prefix=g2-policy-protocol/ -o dist/g2-policy-protocol-handoff.zip HEAD
```

产物包含源码、文档、示例、生成工具和测试；不包含 `.git`、未提交改动、构建缓存、模型权重或观测录制。接收者解压后可以独立安装、开发并验收，无需 wholebody、lerobot-finetune 或 OpenPI 仓库。版本要求仍与域控一致；当前仅交付 v3 契约。

交付给集成人员时提供 server 启动命令、模型/协议包版本、连接地址、metadata 配置、普通验收报告以及启用 RTC 时的专项报告。附上使用的样本 provenance、已确认的模型坐标与夹爪约定，并说明尚未进行真实机器人运行验证。
