# G2 训练 policy 字段速查

LeRobot 和 OpenPI 使用同一套 `policy` 树；填好的 YAML 就是运行时契约，没有另一套编译字段。OpenPI 训练配置外层为 `schema_version: 6`；LeRobot 的 `policy` 节点相同，不额外声明版本。候选值与组合约束由 [schema.py](../src/g2_policy_tools/schema.py) 校验。框架怎么跑训练见各自 `scripts/g2/docs/training.md`。

## 按问题配置

| 问题 | 配置 | 候选或规则 |
|---|---|---|
| 输入什么？ | `data.cameras`、`policy.state`、数据集任务文本 | State 至少启用 joint/pose 一支，也可同时启用；固定读取实测数据。任务文本仍由各模型的数据流程处理。 |
| 学习谁的动作？ | `policy.action.*.source` | `action` 取 action.* 指令列；`state` 取 state.* 实测列。Action 恰好启用一支。 |
| 末端点和旋转？ | `*.pose.coordinates` | 训练 YAML 只选 `ee_frame`、`rotation`。磁盘 `pose_frame` / `action_parent_frame` 是采集事实，不是训练选项。 |
| 是否作差？ | `*.delta` | YAML 布尔值 `true` / `false`。false 为绝对值；true 必须再选 `method`。 |
| 怎么作差？ | `*.method` | `component`：在编码后的父坐标系里减分量。`se3`：`inv(T_ref) @ T`，仅 pose。 |
| 相对谁？ | `*.reference.source` | `state`（当前实测）或 `action`（当前指令）。delta=false 时为 null。 |
| 整个序列如何用参考？ | `*.reference.scope` | `chunk`：全部相对同一参考。`step`：首个相对 source，后续相对前一个目标或历史帧。State 与 Action 都要填。 |

未启用分支保留配置，但不启用其数值语义；未知字段始终报错。夹爪始终是绝对闭合比例（0 开、1 闭），不参与坐标差分或序列累积。`delta: true` 必须同时写 `method` 和 `reference`。采集时的 `pose_frame` 留在数据集元数据里；相对末端用 `method: se3`，不要在训练 YAML 写 `parent_frame`。

## 一个完整例子

输入右臂实测绝对位姿，学习指令位姿相对当前实测的分量差，整个 chunk 共用参考（当前 TeaPi）：

```yaml
policy:
  state:
    joint: {enabled: false}
    pose:
      enabled: true
      coordinates: {ee_frame: flange, rotation: rot6d}
      delta: false
  action:
    joint: {enabled: false}
    pose:
      enabled: true
      quantity: pose
      source: action
      coordinates: {ee_frame: flange, rotation: rot6d}
      delta: true
      method: component
      reference:
        source: state
        scope: chunk
```

UMI 相对末端：state/action 都用 `method: se3`、`source: state`、`scope: chunk`、`rotation: rot6d`；state 需要 `history_steps > 1`，否则相对自身会变成单位位姿。

LeRobot 可以直接使用此片段。OpenPI 自定义 YAML 是对完整默认配置的覆盖；在默认 action.pose 上只写 `delta: false` 会清空继承的 method 和 reference。

## 坐标表达与作差

磁盘和 wire 仍是采集时的 `pose_frame`（默认 base_link）下 flange/rot6d。训练 YAML 的 `ee_frame` / `rotation` 只改变模型内部表示。作差在该分支自己的父坐标系里完成；State 与 Action 的父系可以不同。若采集 `pose_frame` 与 `action_parent_frame` 不同，对应分支必须提供 `base_to_parent`，缺变换则失败。

| delta | method | 模型学习的非夹爪分量 |
|---|---|---|
| `false` | `null` | 目标的绝对位置与旋转编码。 |
| `true` | `component` | `p_target − p_ref` 与 `e(R_target) − e(R_ref)`。rot6d 分量差不是角度差。 |
| `true` | `se3` | `inv(T_ref) @ T_target` 的位置和旋转编码。单位旋转：rot6d `[1,0,0,0,1,0]`，四元数 `[0,0,0,1]`，axis_angle `[0,0,0]`。 |

例如参考绕 z 轴旋转 90°，目标沿父系 x 正方向移动 0.1 米：component 的位置差是 `[0.1,0,0]`；se3 的位置是 `[0,-0.1,0]`。

日常训练配置只选择两个坐标字段：

```yaml
coordinates:
  ee_frame: flange       # flange；命名 TCP 需要标定
  rotation: rot6d        # rot6d | quaternion（xyzw）| axis_angle
```

自定义父坐标系或 TCP 仍须显式 `base_to_parent` / `flange_to_tcp`；不能只改名称就假定已经完成变换。两支同时使用动态父坐标系（`base_to_parent: observation`）时共用同一条 `observation/base_to_parent`。

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
| `action_horizon` | 与模型一致的正整数 | LeRobot 由模型读取，可省略；OpenPI 显式指定。 |

第 i 个目标时刻为 `t + (target_offset_steps + i * stride_steps) / sample_hz`，i 从 0 开始。修改坐标、delta、method、参考或采样定义后应新建实验并重算统计量。配置不接受 spatial/temporal 别名。
