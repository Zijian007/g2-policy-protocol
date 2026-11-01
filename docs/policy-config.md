# G2 训练 policy 字段速查

[文档入口](../README.md#文档导航) · [开发维护](development.md#职责划分) · [v3 协议](protocol.md)

LeRobot 和 OpenPI 使用同一套 `policy.state/action/timing`，约定如何把机器人观测和示教目标编码成模型输入与训练标签，并在部署时还原输出。训练 YAML 外层、配置版本和 checkpoint 格式由各模型仓库维护，见 [OpenPI 训练配置](../../openpi/scripts/g2/docs/training.md) 和 [LeRobot 训练配置](../../lerobot-finetune/scripts/g2/docs/training.md)。候选值与组合约束由 [data/policy.py](../src/g2_policy_protocol/data/policy.py) 校验。

## 先理解整体流程

`state` 决定模型看到的机器人状态，`action` 决定模型学习的目标，`timing` 决定历史和目标对应哪些时刻。图像由 `data.cameras` 选择，任务文本由模型的数据流程处理。

```text
训练：图像、实测状态、任务文本 + 示教目标 → 按 policy 编码 → 归一化 → 模型学习
部署：图像、实测状态、任务文本 → 同样的输入编码与归一化 → 模型预测
                                                      ↓
                                      反归一化 → 按 policy 解码 → 域控执行
```

对于位姿，先区分“预测哪个目标”和“用什么数字表达目标”：同一个实际目标可以表示为绝对位姿、相对参考的分量差，或参考坐标系中的相对变换。`coordinates`、`delta`、`reference`、`method` 共同决定这种表示；它们不改变所选示教轨迹本身。

阅读相对位姿配置时，沿着“取目标 → 取当前参考 → 在共同父系中表达 → 按 scope 和 method 编码”这条顺序理解。目标来源、参考来源与坐标表达可以分别选择。

阅读顺序：[字段速查](#按问题配置) → [配置例子](#例子绝对状态与相对动作) → [参考来源与序列](#参考来源与序列) → [坐标表达与作差](#坐标表达与作差)。多数据集属于训练 YAML 外层配置，见[多数据集与图像尺寸](#多数据集与图像尺寸)。

## 按问题配置

| 问题 | 配置 | 候选或规则 |
|---|---|---|
| 输入什么？ | `data.cameras`、`policy.state`、数据集任务文本 | State 至少启用 joint/pose 一支，也可同时启用；固定读取实测数据。任务文本仍由各模型的数据流程处理。 |
| 预测什么控制量、学习谁的轨迹？ | `policy.action`、`policy.action.*.source` | Action 恰好启用 pose/joint 一支，预测末端位姿或关节位置目标；source 选择 action.* 指令列或 state.* 实测列。 |
| 坐标系、末端点和旋转？ | `*.pose.coordinates` | `parent_frame` 选 base_link / arm_base_link；`ee_frame` 选 flange / 标定 TCP；`rotation` 选旋转编码。磁盘坐标声明是采集事实。 |
| 是否编码相对量？ | `*.delta` | YAML 布尔值 `true` / `false`。false 保留原控制量；true 必须指定 method 和 reference。 |
| 相对谁？ | `*.reference.source` | `state`（当前实测）或 `action`（当前指令）；它选择参考，不选择训练目标。 |
| 整个序列如何用参考？ | `*.reference.scope` | `chunk`：全部相对同一参考。`step`：首个相对 source，后续相对前一个目标或历史帧。仅 delta=true 时指定。 |
| 怎样计算相对量？ | `*.method` | `component`：编码后的分量差。`se3`：参考位姿到目标的相对变换，仅 pose。 |
| 看多少历史、预测哪些时刻？ | `policy.timing` | 历史长度、采样频率、预测长度、首目标偏移和目标间隔。 |

`*.pose.coordinates` 中的 `*` 指 `policy.state` 或 `policy.action`；`*.delta`、`*.method` 和 `*.reference` 中的 `*` 指具体的 pose/joint 分支。未启用分支保留配置，但不启用其数值语义；未知字段始终报错。夹爪始终是绝对闭合比例（0 开、1 闭），不参与坐标差分或序列累积。

`delta: true` 时，method 与 reference.source/scope 都必须有效；`delta: false` 时写法如下（reference 本身仍为映射）：

```yaml
delta: false
method: null
reference:
  source: null
  scope: null
```

## 例子：绝对状态与相对动作

以下是实验 YAML 的配置片段，模型、数据路径和训练参数仍需在各框架中配置。输入右臂实测绝对位姿，学习指令位姿相对当前实测的分量差，整个 chunk 共用参考（TeaPi 示例）：

```yaml
experiment:
  arm: right
policy:
  state:
    joint: {enabled: false}
    pose:
      enabled: true
      coordinates: {parent_frame: base_link, ee_frame: flange, rotation: rot6d}
      delta: false
      method: null
      reference: {source: null, scope: null}
  action:
    joint: {enabled: false}
    pose:
      enabled: true
      quantity: pose
      source: action       # T_target：取目标时刻的指令轨迹
      coordinates: {parent_frame: base_link, ee_frame: flange, rotation: rot6d}
      delta: true
      method: component
      reference:
        source: state      # T_current：取当前实测位姿作为初始参考
        scope: chunk
  timing:
    sample_hz: 30.0
    history_steps: 1
    action_horizon: 50      # LeRobot 须与所选模型的预测长度一致
    target_offset_steps: 0
    stride_steps: 1
```

这段配置的含义是：模型看当前右臂绝对位姿，预测从当前时刻开始、以 30 Hz 采样的 50 帧指令目标；每帧的非夹爪标签都减去本次观测的实测参考，而不是逐帧累加。

若采用 UMI 风格的相对末端表示，state.pose/action.pose 都设置 `delta: true`、`method: se3`、`reference: {source: state, scope: chunk}`、`rotation: rot6d`。State 历史和 Action 目标都变换到当前实测末端系；`parent_frame` 可以保持 `base_link`，原因见[父系与相对编码](#parent_frame-与-delta-的关系)。State 需要 `history_steps > 1`，否则位姿相对自身会变成单位变换，配置校验会拒绝。

要对应 UMI 从示教实测轨迹构造动作标签的方式，再设置 `action.pose.source: state`；保留 `source: action` 则仍学习指令轨迹。这里对照的是核心位姿表达；先将 TCP/flange 视为同一末端点，不涉及额外的 episode 起点朝向等输入。直接接入原始 UMI 数据还需转换数据格式、磁盘坐标和末端定义、rot6d 顺序与夹爪编码，不能只复制配置。

LeRobot 可以直接使用此片段。OpenPI 自定义 YAML 是对完整默认配置的覆盖；在默认 action.pose 上只写 `delta: false` 会清空继承的 method 和 reference。

## 参考来源与序列

两种 source 回答不同问题：

| 字段 | 对应位姿矩阵 | 决定什么 |
|---|---|---|
| `action.pose.source` | `T_target(i)`，第 i 个目标 | `action`：目标时刻的指令位姿；`state`：目标时刻的实测位姿。 |
| `action.pose.reference.source` | `T_current`，本次观测时刻的初始参考 | `state`：当前实测位姿；`action`：当前指令位姿。 |

因此，两个 `source` 不是重复字段：前者回答“要预测谁”，后者回答“相对于谁”。例如 `source: state` 与 `reference.source: state` 表示“预测未来实测轨迹，并以当前实测位姿为参考”；`source: action` 与 `reference.source: state` 表示“预测未来指令轨迹，并以当前实测位姿为参考”。

矩阵由配置选择的数据计算得到，并非 YAML 中直接保存的字段。目标时刻由 `policy.timing` 指定；`target_offset_steps=0` 时，首目标就是当前时刻。“未来轨迹”在这里包括这一首目标。

若当前实测 x=0.40、当前指令 x=0.41、未来指令目标 x=0.45，`source: action` 配合 component 对两种参考分别得到 +0.05 和 +0.04。`reference.source: action` 使用当前指令，不是未来目标自身。

| reference.source | 含义 | 数据 / 部署上下文 |
|---|---|---|
| `state` | 当前实测 | state.pose / state.joint |
| `action` | 当前指令 | action.pose / action.joint；线上为 command_pose / command_joint |

source 是“从哪里取初始参考”；scope 是“预测或历史序列如何使用它”，与学习 action/state 标签相互独立：

- `chunk`：全部目标或历史帧使用同一个参考。
- `step`：Action 首个目标相对 source，其余相对前一个目标；State 历史同样，首帧相对 source，其余相对前一历史帧。解码依次还原，不读取未来实测数据。

下文公式统一把实际使用的参考记为 `T_ref`。`chunk` 下整段均为 `T_ref = T_current`；`step` 下只有首项使用 `T_current`，后续参考按上面的序列规则更新。State 的待编码位姿来自实测历史，其参考同样由 State 分支自己的 `reference` 指定。

参考 x=0.40，目标 x=[0.42,0.45,0.47]，使用 component 时：

| 项目 | 第一个目标 | 第二个目标 | 第三个目标 |
|---|---|---|---|
| 绝对目标 | 0.42 | 0.45 | 0.47 |
| chunk 使用的参考 | 0.40 | 0.40 | 0.40 |
| chunk 标签 | +0.02 | +0.05 | +0.07 |
| step 使用的参考 | 0.40 | 0.42 | 0.45 |
| step 标签 | +0.02 | +0.03 | +0.02 |

chunk 的参考仅在本次样本或推理请求内固定；下一次观测重新取参考。step 后续参考的是上一目标，而不是未来时刻的实测位置。两种 method 都按此 scope 选择参考，区别只在于计算相对量的运算。服务端还原绝对目标后发送，客户端无需再次叠加。

`history_steps=1` 且 state 相对 `source: state` 会变成相对自身，配置阶段直接报错。夹爪仍为绝对值。线上历史和指令参考由客户端提供，服务端不按调用次数猜测。

## 坐标表达与作差

磁盘和 wire 使用 `base_link` 下的 flange/rot6d 绝对位姿。入口要求所用 pose 列的 `pose_frame` / `action_parent_frame` 为 base_link；不匹配的数据须先按真实采集语义转换。训练 YAML 的坐标设置只改变模型内部表示。目标和参考先在该分支自己的父坐标系中表达，再按 method 编码相对量；State 与 Action 的父系可以分别选择。

| delta | method | 模型学习的非夹爪分量 |
|---|---|---|
| `false` | `null` | 目标的绝对位置与旋转编码。 |
| `true` | `component` | `p_target − p_ref` 与 `e(R_target) − e(R_ref)`。rot6d 分量差不是角度差。 |
| `true` | `se3` | `inv(T_ref) @ T_target` 的位置和旋转编码。单位旋转：rot6d `[1,0,0,0,1,0]`，四元数 `[0,0,0,1]`，axis_angle `[0,0,0]`。 |

**两种 method 都有参考位姿，由 `reference` 指定；区别在于参考位姿如何参与计算。**

- `component` 把参考位姿当作数值相减的基准，位置差仍沿原父坐标系的轴表达，旋转编码逐分量相减。
- `se3` 把参考位姿当作局部坐标系，计算目标在该参考系中的位置和朝向，即刚体相对变换。

从变换角度看，`component` 使用 `目标编码 = 参考编码 + 差值编码` 还原非夹爪分量；
`se3` 使用 `ΔT = inv(T_ref) @ T_target` 编码、`T_target = T_ref @ ΔT` 还原位姿。
其中 `T_ref` 和 `T_target` 都是相对于同一父坐标系的位姿矩阵。`se3` 的位置为
`R_refᵀ @ (p_target − p_ref)`，相对旋转为 `R_refᵀ @ R_target`，再按所选 rotation 编码；
这里不是把相对变换转换成六维李代数。夹爪始终保留绝对值。

`p` 表示位置，`R` 表示旋转矩阵，`T` 表示包含两者的 4×4 位姿矩阵，`e(R)` 表示所选旋转编码。参考与目标姿态相同时，component 的旋转差为零；se3 的相对旋转为单位旋转，其 rot6d 编码为 `[1,0,0,0,1,0]`，不是六个零。

例如参考绕 z 轴旋转 90°，目标沿父系 x 正方向移动 0.1 米：component 的位置差是 `[0.1,0,0]`；se3 的位置是 `[0,-0.1,0]`。

### `parent_frame` 与 `delta` 的关系

`parent_frame` 选择编码前目标和参考共同采用的父坐标系；`delta` 选择是否将它们编码为相对量。父系是表达绝对位姿的坐标系，`se3` 产生的相对量则在参考末端系中表达。以 `parent_frame: base_link`、`method: se3`、`scope: chunk` 为例：

```text
T_current   = 当前实测或当前指令在 base_link 下的末端位姿
T_target(i) = timing 指定的第 i 个目标在 base_link 下的末端位姿
ΔT(i)       = inverse(T_current) · T_target(i)
```

这里的 `T_current` 由 `action.pose.reference.source` 选择，`T_target(i)` 由 `action.pose.source` 选择；两者都使用 Action 分支的坐标设置，不由 State 输入分支的编码方式决定。`ee_frame: flange` 表示末端点是法兰；命名 TCP 则先应用 `flange_to_tcp` 固定变换。

对于 `se3`，若对参考和目标共同应用同一个父系变换 `A`，该变换会抵消：

```text
inverse(A·T_current) · (A·T_target)
= inverse(T_current) · T_target
```

因此 `parent_frame: base_link` 配合 `se3` 和当前实测参考，就能得到 UMI 风格的当前末端局部相对目标。父系不必在整个运行期间固定：即使它随腰部运动，只要本次编码对参考、历史和所有目标共用当前时刻的 `A(t)`，抵消仍成立；若每个目标使用各自未来时刻的变换，则不能这样抵消。

这说明 `se3` 相对位姿对公共父系变换保持不变，不代表启用任意 delta 后父系都无关。绝对位姿和 `component` 标签一般依赖所选父系，State 与 Action 也可能采用不同编码。当前实现仍会校验父系、加载所需变换并按同一流程解码；选择 `arm_base_link` 仍要求相应数据和部署上下文。

### 坐标配置与数据要求

坐标配置：

```yaml
coordinates:
  parent_frame: arm_base_link # base_link（默认）| arm_base_link
  ee_frame: flange       # flange；命名 TCP 需要标定
  rotation: rot6d        # rot6d | quaternion（xyzw）| axis_angle
```

命名 TCP 仍须提供左右臂 `flange_to_tcp` 标定。父坐标系只接受 base_link / arm_base_link；训练 YAML 不接受 `base_to_parent`。

选择 arm_base_link 时，数据集必须包含 `state.arm_base`（9 维 xyz + rot6d），表示 base_link 下的实测腰部末端位姿。采样器在当前观测时刻 t 插值平移和 SO(3) 旋转，然后计算 `T_armbase←base(t) = inverse(T_base←armbase(t))`。整个历史输入、预测 chunk 和差分参考共用这一变换，不读取未来腰部状态。episode 起点之前仍重复首帧。

部署握手按需声明 `context.base_to_parent`，含义固定为 `T_armbase←base(t)`。wholebody 使用当前实测腰部关节和缓存的运动链计算它；服务端用同一次请求的变换编码输入、RTC 前缀并将输出还原到 base_link。动作执行期间不持续跟随腰部变化。base_link 模型不读取腰部列、不计算腰部 FK。

父坐标系保存在 checkpoint 配置中，并参与归一化缓存和续训语义检查。切换坐标系须新建实验并重算统计量；修改旧 checkpoint 的坐标名称不会转换模型权重。

每臂位姿维度是 xyz3 + 旋转编码 + 夹爪1：rot6d=10、quaternion=8、axis_angle=7；关节位置是 7 个关节 + 夹爪，共 8 维。

## 支持的控制量

| 字段 | 候选或限制 |
|---|---|
| `action.pose.quantity` | `pose`，末端位姿目标。 |
| `action.joint.quantity` | 仅 `position`，关节位置目标，取 action.joint 或 state.joint。 |
| `*.method` | 关节只能 `component`；`se3` 只允许 pose。 |

`delta: false` 时，method、reference.source 和 reference.scope 均为 null，reference 仍为映射。joint 分支没有 coordinates。模型可表达的控制量与域控已绑定的执行能力需分别确认，见各框架部署文档。

## 采样与恢复

| policy.timing 字段 | 候选 | 含义 |
|---|---|---|
| `sample_hz` | 有限正数 | 采样网格频率，Hz。 |
| `history_steps` | 正整数 | 历史帧数，须匹配模型输入能力。 |
| `target_offset_steps` | 非负整数 | 首目标距当前观测的采样步数。 |
| `stride_steps` | 正整数 | 相邻目标间隔的采样步数。 |
| `action_horizon` | 与模型一致的正整数 | 统一预测长度；LeRobot 映射为 chunk_size，OpenPI 映射为 model.action_horizon。 |

第 i 个目标时刻为 `t + (target_offset_steps + i * stride_steps) / sample_hz`，i 从 0 开始。修改坐标、delta、method、参考或采样定义后应新建实验并重算统计量。配置不接受 spatial/temporal 别名。

动作目标间隔为 `stride_steps / sample_hz` 秒，对应 `policy_hz = sample_hz / stride_steps`；历史间隔仍为 `1 / sample_hz` 秒。sample_hz 是数据采样频率，不是模型每秒必须完成的推理次数。

例如 sample_hz=30、stride_steps=1、target_offset_steps=0、action_horizon=50 时，首目标在 t，末目标在 t+49/30 秒（约 1.633 秒）。预测长度不等于客户端一次必须执行的长度；服务端返回完整 chunk，执行和重新规划由客户端调度。

历史超出 episode 起点时重复首帧，目标超出终点时重复末帧；低维位姿的平移线性插值、旋转在 SO(3) 上插值，图像取近邻帧，不跨 episode。

## 多数据集与图像尺寸

训练 YAML 的 `data` 支持以下写法，需要 `g2-policy-protocol==0.7.0` 及配套训练/部署代码。外层 `schema_version` 由模型仓库维护：当前 OpenPI 使用 v8，LeRobot 使用 v7；它们均独立于通信协议版本 `g2.policy.v3`。

以下路径以 OpenPI 的 LeRobot v2.1 数据为例；lerobot-finetune 使用 v3，需换成对应数据目录。

```yaml
data:
  root: ./data
  cameras: null
  image_size: [224, 224]   # [高度, 宽度]；可选，统一 letterbox
  sources:
    - path: vr/task_a/lerobot_v21
      weight: 3
    - name: task_b
      path: /mnt/data/task_b/lerobot_v21
      weight: 1
```

- `data.repo_id` 与 `data.sources` 互斥。旧单源配置保持有效；新列表使用 `path`。
- 相对 `path` 相对于 `data.root`，绝对路径直接使用；`root` 默认 `./data`，相对训练仓库根目录解析。名称默认取填写的 path，必须唯一。
- 所有源都不写 `weight`，或所有源都写 `weight: auto` 时，按有效样本数量自然混合；全部填写有限正权重时按完整 episode 重复近似加权，不能只填一部分。权重 3:1 表示目标采样比例约 75%:25%，不要求每个 batch 配平。启动/inspect 输出实际占比；长短 episode 和小数重复会影响精度。`auto` 不能和数字权重混用。
- 训练只复制索引，不合并或修改源数据。历史帧和动作 chunk 不跨 episode 或数据源。
- LeRobot 启用验证集时，在每个源内部按 episode 划分；验证集不重复、不加权，使用训练集归一化统计量。
- LeRobot 在样本进入 batch 前 resize；未设置 image_size 时要求各源对应相机形状兼容。OpenPI Pi05 的 image_size 仅接受 `[224, 224]`，沿用模型最终 resize。
- 模型依然使用同一套 state/action、相机与控制语义。Pi05 保留任务文本；ACT 不因数据集增加而获得语言条件能力。

多源 checkpoint 保存混合指纹与归一化信息，部署无需原始训练数据。数据内容、权重或策略改变时新建实验。LeRobot 复用原生数据顺序续训（相同 batch 和并行规模）；OpenPI 恢复训练状态，但不保证接续原来的数据顺序。

## Checkpoint 与部署

权重、实际生效的 policy 和归一化统计量共同确定模型输出的含义。训练保存配置快照，部署按该快照恢复语义和统计量；修改当前训练 YAML 不会改变已有 checkpoint 的语义。快照位置与格式由各模型仓库维护。

模型内部可以采用单臂、delta、TCP 或不同旋转编码，通信位姿仍按 base_link 下的 flange/rot6d 绝对目标传输；握手声明所需观测与上下文。参考由当前请求或其历史提供，服务端不依赖调用次数猜测参考。RTC 前缀也经同一套动作编码处理，客户端不重复作差或归一化。
