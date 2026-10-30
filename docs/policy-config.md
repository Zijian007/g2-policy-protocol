# G2 训练 policy 字段速查

[文档入口](../README.md#文档导航) · [开发维护](development.md#职责划分) · [v3 协议](protocol.md)

LeRobot 和 OpenPI 使用同一套 `policy` 树；填好的 YAML 就是运行时契约，没有另一套编译字段。共享包只定义 `policy.state/action/timing` 的数据语义；训练 YAML 外层、配置版本和 checkpoint 格式由各模型仓库维护，见 [OpenPI 训练配置](../../openpi/scripts/g2/docs/training.md)和 [LeRobot 训练配置](../../lerobot-finetune/scripts/g2/docs/training.md)。候选值与组合约束由 [data/policy.py](../src/g2_policy_protocol/data/policy.py) 校验。框架怎么跑训练见各自 `scripts/g2/docs/training.md`。

## 多数据集与图像尺寸

训练 YAML 的 `data` 支持以下写法（schema 仍为 7，需要 0.7.0 协议工具及配套训练/部署代码）：

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
- 所有源都不写 weight 时按有效样本数量自然混合；全部填写有限正权重时按完整 episode 重复近似加权，不能只填一部分。权重 3:1 表示目标采样比例约 75%:25%，不要求每个 batch 配平。启动/inspect 输出实际占比；长短 episode 和小数重复会影响精度。
- 训练只复制索引，不合并或修改源数据。历史帧和动作 chunk 不跨 episode 或数据源。
- 验证集在每个源内部按 episode 划分，不重复、不加权，使用训练集归一化统计量。
- LeRobot 在样本进入 batch 前 resize；未设置 image_size 时要求各源对应相机形状兼容。OpenPI Pi05 的 image_size 仅接受 `[224, 224]`，沿用模型最终 resize。
- 模型依然使用同一套 state/action、相机与控制语义。Pi05 保留任务文本；ACT 不因数据集增加而获得语言条件能力。

多源 checkpoint 保存混合指纹与归一化信息，部署无需原始训练数据。数据内容、权重或策略改变时新建实验。LeRobot 复用原生数据顺序续训（相同 batch 和并行规模）；OpenPI 恢复训练状态，但不保证接续原来的数据顺序。

## 按问题配置

| 问题 | 配置 | 候选或规则 |
|---|---|---|
| 输入什么？ | `data.cameras`、`policy.state`、数据集任务文本 | State 至少启用 joint/pose 一支，也可同时启用；固定读取实测数据。任务文本仍由各模型的数据流程处理。 |
| 学习谁的动作？ | `policy.action.*.source` | `action` 取 action.* 指令列；`state` 取 state.* 实测列。Action 恰好启用一支。 |
| 坐标系、末端点和旋转？ | `*.pose.coordinates` | `parent_frame` 选 base_link / arm_base_link；`ee_frame` 选 flange / 标定 TCP；`rotation` 选旋转编码。磁盘坐标声明是采集事实。 |
| 是否作差？ | `*.delta` | YAML 布尔值 `true` / `false`。false 为绝对值；true 必须再选 `method`。 |
| 怎么作差？ | `*.method` | `component`：在编码后的父坐标系里减分量。`se3`：`inv(T_ref) @ T`，仅 pose。 |
| 相对谁？ | `*.reference.source` | `state`（当前实测）或 `action`（当前指令）。delta=false 时为 null。 |
| 整个序列如何用参考？ | `*.reference.scope` | `chunk`：全部相对同一参考。`step`：首个相对 source，后续相对前一个目标或历史帧。State 与 Action 都要填。 |

未启用分支保留配置，但不启用其数值语义；未知字段始终报错。夹爪始终是绝对闭合比例（0 开、1 闭），不参与坐标差分或序列累积。`delta: true` 必须同时写 `method` 和 `reference`。`parent_frame` 决定位姿表达的父坐标系；相对末端用 `method: se3`。

## 一个完整例子

输入右臂实测绝对位姿，学习指令位姿相对当前实测的分量差，整个 chunk 共用参考（当前 TeaPi）：

```yaml
policy:
  state:
    joint: {enabled: false}
    pose:
      enabled: true
      coordinates: {parent_frame: base_link, ee_frame: flange, rotation: rot6d}
      delta: false
  action:
    joint: {enabled: false}
    pose:
      enabled: true
      quantity: pose
      source: action
      coordinates: {parent_frame: base_link, ee_frame: flange, rotation: rot6d}
      delta: true
      method: component
      reference:
        source: state
        scope: chunk
```

UMI 相对末端：state/action 都用 `method: se3`、`source: state`、`scope: chunk`、`rotation: rot6d`；state 需要 `history_steps > 1`，否则相对自身会变成单位位姿。

LeRobot 可以直接使用此片段。OpenPI 自定义 YAML 是对完整默认配置的覆盖；在默认 action.pose 上只写 `delta: false` 会清空继承的 method 和 reference。

## 坐标表达与作差

磁盘和 wire 使用 `base_link` 下的 flange/rot6d 绝对位姿。入口要求所用 pose 列的 `pose_frame` / `action_parent_frame` 为 base_link；不匹配的数据须先按真实采集语义转换。训练 YAML 的坐标设置只改变模型内部表示。作差在该分支自己的父坐标系里完成；State 与 Action 的父系可以分别选择。

| delta | method | 模型学习的非夹爪分量 |
|---|---|---|
| `false` | `null` | 目标的绝对位置与旋转编码。 |
| `true` | `component` | `p_target − p_ref` 与 `e(R_target) − e(R_ref)`。rot6d 分量差不是角度差。 |
| `true` | `se3` | `inv(T_ref) @ T_target` 的位置和旋转编码。单位旋转：rot6d `[1,0,0,0,1,0]`，四元数 `[0,0,0,1]`，axis_angle `[0,0,0]`。 |

例如参考绕 z 轴旋转 90°，目标沿父系 x 正方向移动 0.1 米：component 的位置差是 `[0.1,0,0]`；se3 的位置是 `[0,-0.1,0]`。

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

## 参考来源与序列

| reference.source | 含义 | 数据 / 部署上下文 |
|---|---|---|
| `state` | 当前实测 | state.pose / state.joint |
| `action` | 当前指令 | action.pose / action.joint；线上为 command_pose / command_joint |

source 是“从哪里取初始参考”；scope 是“预测或历史序列如何使用它”，与学习 action/state 标签相互独立：

- `chunk`：全部目标或历史帧使用同一个参考。
- `step`：Action 首个目标相对 source，其余相对前一个目标；State 历史同样，首帧相对 source，其余相对前一历史帧。解码依次还原，不读取未来实测数据。

参考 x=0.40，目标 x=[0.42,0.45,0.47] 时，component + chunk 标签为 [0.02,0.05,0.07]，step 标签为 [0.02,0.03,0.02]。服务端还原绝对目标后发送，客户端无需再次叠加。

`history_steps=1` 且 state 相对 `source: state` 会变成相对自身，配置阶段直接报错。夹爪仍为绝对值。线上历史和指令参考由客户端提供，服务端不按调用次数猜测。

## 关节与其他控制量

| 字段 | 候选或限制 |
|---|---|
| `action.pose.quantity` | `pose`，末端位姿目标。 |
| `action.joint.quantity` | `position` / `velocity` / `torque`，分别取 *.joint / *.joint_velocity / *.joint_torque。仅 position 支持 delta。 |
| `*.method` | 关节只能 `component`；`se3` 只允许 pose。 |

所有不使用参考的分支，其 method 与 reference 为 null。joint 分支没有 coordinates。

## 采样与恢复

| policy.timing 字段 | 候选 | 含义 |
|---|---|---|
| `sample_hz` | 有限正数 | 采样网格频率，Hz。 |
| `history_steps` | 正整数 | 历史帧数，须匹配模型输入能力。 |
| `target_offset_steps` | 非负整数 | 首目标距当前观测的采样步数。 |
| `stride_steps` | 正整数 | 相邻目标间隔的采样步数。 |
| `action_horizon` | 与模型一致的正整数 | 统一预测长度；LeRobot 映射为 chunk_size，OpenPI 映射为 model.action_horizon。 |

第 i 个目标时刻为 `t + (target_offset_steps + i * stride_steps) / sample_hz`，i 从 0 开始。修改坐标、delta、method、参考或采样定义后应新建实验并重算统计量。配置不接受 spatial/temporal 别名。
