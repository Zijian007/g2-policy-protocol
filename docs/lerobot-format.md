# G2 LeRobot 数据格式规范

本文规范 **G2 导出后的 LeRobot logical schema**：列名、维度、向量布局、相机 key、坐标系和 state/action 语义。

先看下面的完整 schema，再看后面的逐项解释。

## 1. 完整 schema

下面是 G2 LeRobot 格式的字段总览。`feature_catalog` 是完整字段集合；某个数据集实际需要哪些字段，由 policy 配置决定，见[必需字段](#7-必需字段)。

```yaml
g2_lerobot:
  # G2 logical schema id，不是 LeRobot 的物理存储版本
  schema_id: g2_lerobot_keys_v3
  storage: v2.1 | v3.0

  feature_catalog:
    state:
      joint:
        key: state.joint
        shape: [16]
        layout: [left_joint_1..7, left_gripper, right_joint_1..7, right_gripper]
      pose:
        key: state.pose
        shape: [20]
        layout: [left_xyz, left_rot6d, left_gripper,
                 right_xyz, right_rot6d, right_gripper]
      torso:
        key: state.torso
        shape: [5]
      head:
        key: state.head
        shape: [3]
      arm_base:
        key: state.arm_base
        shape: [9]
        layout: [arm_base_xyz, arm_base_rot6d]

    action:
      joint:
        key: action.joint
        shape: [16]
        layout: [left_joint_1..7, left_gripper, right_joint_1..7, right_gripper]
      pose:
        key: action.pose
        shape: [20]
        layout: [left_xyz, left_rot6d, left_gripper,
                 right_xyz, right_rot6d, right_gripper]
      torso:
        key: action.torso
        shape: [5]
      head:
        key: action.head
        shape: [3]

    images:
      head_color: observation.images.head_color
      hand_left_color: observation.images.hand_left_color
      hand_right_color: observation.images.hand_right_color

  vector_layout:
    joint_16:
      0..6: left_joint_1..7
      7: left_gripper
      8..14: right_joint_1..7
      15: right_gripper

    pose_20:
      0..2: left_xyz
      3..8: left_rot6d_0..5
      9: left_gripper
      10..12: right_xyz
      13..18: right_rot6d_0..5
      19: right_gripper

    arm_base_9:
      0..2: arm_base_xyz
      3..8: arm_base_rot6d_0..5

  conventions:
    pose_frame: base_link
    action_parent_frame: base_link
    ee_frame: flange / arm_*_end_link
    rotation: rot6d
    gripper:
      field: close_01
      range: [0, 1]
      0: open
      1: closed

  metadata:
    wholebody_export:
      ee_frame: flange
      rotation: rot6d
      pose_frame: base_link
      action_parent_frame: base_link
      action_supervision: command
```

这个 schema 中最关键的事实是：

```text
state.pose / action.pose
= base_link 坐标系下的 flange / arm_*_end_link 位姿
= xyz + rot6d + gripper
```

它们不是夹爪 TCP 位姿，也不是左右臂局部坐标系下的位姿。

## 2. 格式边界

G2 数据链路有三层格式：

```text
wholebody raw
  meta.json + frames.jsonl + images/<camera>/*.jpg
        │
        │ 解析和字段投影
        ▼
G2 LeRobot logical schema       ← 本文规范这一层
  state.* / action.* / observation.images.*
        │
        │ v2.1/v3.0 writer
        ▼
LeRobot physical storage
  meta/ + data/ + videos/
```

职责分别是：

- `wholebody raw`：保存原始 state、command、时间和图像路径。
- G2 logical schema：规定导出后的列名、维度、布局和语义。
- LeRobot physical storage：规定 v2.1/v3.0 的 Parquet、视频和 metadata 组织。

本仓库规范第二层。raw 的完整字段和物理写入实现位于 `wholebody`。

## 3. Feature 解释

### 3.1 双臂 joint

`state.joint` 和 `action.joint` 都是 16 维：

```text
0..6    left_joint_1 ... left_joint_7
7       left_gripper
8..14   right_joint_1 ... right_joint_7
15      right_gripper
```

关节单位是 rad。夹爪使用 `close_01`：

```text
0 = open
1 = closed
```

### 3.2 双臂 pose

`state.pose` 和 `action.pose` 都是 20 维：

```text
0..2    left_x, left_y, left_z
3..8    left_rot6d_0 ... left_rot6d_5
9       left_gripper

10..12  right_x, right_y, right_z
13..18  right_rot6d_0 ... right_rot6d_5
19      right_gripper
```

坐标和末端约定在 schema 中已经固定：

```text
position / rotation frame: base_link
EE frame:                  flange / arm_*_end_link
rotation representation:   rot6d
```

raw 中可以使用四元数 `orientation_wxyz`，但导出到 LeRobot 后使用 `rot6d`。

### 3.3 身体其他部件

```text
state.torso    [5]   腰部实测关节
action.torso   [5]   腰部目标
state.head     [3]   头部实测关节
action.head    [3]   头部目标
state.arm_base [9]   arm_base_link 在 base_link 下的 xyz + rot6d
```

`state.arm_base` 只在 policy 内部使用 `arm_base_link` 参考系时必需。

### 3.4 图像

raw 相机短名映射为：

```text
head_color       -> observation.images.head_color
hand_left_color  -> observation.images.hand_left_color
hand_right_color -> observation.images.hand_right_color
```

具体 policy 需要哪些相机，由 policy 配置决定；名称必须使用上述逻辑 key。

## 4. 坐标系和动作语义

### 4.1 磁盘上的 canonical 坐标系

包含 pose 的数据集必须按以下语义解释：

```text
pose_frame:          base_link
action_parent_frame: base_link
EE frame:            flange / arm_*_end_link
```

因此：

- `state.pose` 是 `base_link` 下的实测末端法兰位姿。
- `action.pose` 是 `base_link` 下的末端法兰目标。
- 位姿不是在左臂或右臂局部坐标系下表达的。
- 位姿不是夹爪 TCP 位姿。

wholebody 新录制也统一到 `base_link`：

- VR 末端目标写入前转换到 `base_link`。
- Gello joint command 先做 FK，再得到 `base_link` 下的末端目标。
- 实测 pose 是 `arm_*_end_link` 在 `base_link` 下的位姿。

### 4.2 模型内部的 `arm_base_link`

模型内部可以选择 `arm_base_link` 作为计算参考系，但不会改变磁盘上的 canonical schema：

```text
磁盘 pose frame:       base_link
模型内部参考系:        base_link 或 arm_base_link
```

使用 `arm_base_link` 时：

```text
dataset 需要 state.arm_base
deployment observation 需要 context.base_to_parent
server 负责模型空间与 base_link 之间的转换
wholebody 最终仍执行 base_link 下的动作
```

### 4.3 state 和 action

```text
state.*  = 机器人实际测量值
action.* = 录制时发出的目标 / command
```

导出默认使用：

```text
action_supervision = command
```

如果训练要使用 measured state 作为 label，应在 policy 配置中选择 `source: state`，不要修改磁盘列的原始语义：

```yaml
policy:
  action:
    pose:
      source: state
```

## 5. Metadata 约定

包含 pose 的数据集应在 `meta/info.json` 的 `wholebody_export` 中保留：

```json
{
  "wholebody_export": {
    "ee_frame": "flange",
    "rotation": "rot6d",
    "pose_frame": "base_link",
    "action_parent_frame": "base_link",
    "action_supervision": "command"
  }
}
```

`validate_dataset_contract()` 会根据 policy 检查相关字段、维度、fps 和这些 metadata 约定。

## 6. Raw 到 logical schema 的映射

wholebody raw 的主要字段映射如下：

```text
raw state.arm.left.joint
    -> state.joint[0:7]
raw state.gripper.left.close_01
    -> state.joint[7]
raw state.arm.right.joint
    -> state.joint[8:15]
raw state.gripper.right.close_01
    -> state.joint[15]

raw state.arm.left.pose + gripper
    -> state.pose[0:10]
raw state.arm.right.pose + gripper
    -> state.pose[10:20]

raw action.arm.*.joint + action.gripper.*.close_01
    -> action.joint
raw action.arm.*.pose + action.gripper.*.close_01
    -> action.pose

raw images/<camera>/*.jpg
    -> observation.images.<camera>
```

raw 的完整 schema 不在本仓库重复定义，见：

```text
wholebody/docs/user/collect/schema.md
```

实际投影和写入代码见：

```text
wholebody/external_services/data_collect/export/features.py
wholebody/external_services/data_collect/export/append_common.py
wholebody/external_services/data_collect/export/append_v21.py
wholebody/external_services/data_collect/export/append_v3.py
```

## 7. 必需字段

上面的 `feature_catalog` 是完整字段集合，不代表每个数据集都必须包含所有字段。`validate_dataset_contract()` 会根据 policy 计算必需字段。

### Pose policy

通常需要：

```text
state.pose
action.pose
所需的 observation.images.*
```

即使训练 label 选择 `action.pose.source: state`，`action.pose` command 列仍可能作为 command/reference 使用。

### Joint policy

通常需要：

```text
state.joint
action.joint
所需的 observation.images.*
```

### `arm_base_link` policy

在 pose/joint 和相机之外，还需要：

```text
state.arm_base
```

最终以以下共享校验为准：

```python
from g2_policy_protocol.contracts.dataset import validate_dataset_contract
```

## 8. Episode 和物理存储

wholebody 导出的训练数据遵循：

- 一次开录到停录对应一个 episode。
- 连续 segment 可以按顺序写入同一个 episode。
- 暂停区间不补帧。
- 历史帧和 action chunk 不跨 episode。
- `fps` 必须是正的有限值。

物理数据树通常包含：

```text
dataset/
├── meta/
├── data/
└── videos/
```

v2.1 和 v3.0 可以共享本文规定的 logical feature，但 Parquet、视频和 metadata 组织由对应版本的 writer 处理。

当前使用关系：

```text
OpenPI G2：主要读取 LeRobot v2.1
lerobot-finetune G2：以 v3 为目标，v2.1 可先转换到独立 v3 缓存
wholebody：可以导出 v2.1、v3.0 或 both
```

## 9. Source of Truth

如果实现和本文不一致，按以下顺序检查：

1. 固定字段名、field name、维度和相机 ID：
   `src/g2_policy_protocol/types.py`
2. LeRobot 列名、布局和 contract 校验：
   `src/g2_policy_protocol/contracts/dataset.py`
3. policy 如何选择 state/action/reference：
   `src/g2_policy_protocol/contracts/policy.py`
4. raw 到 LeRobot 的实际投影和 v2.1/v3 写入：
   `wholebody/external_services/data_collect/export/`

## 相关文档

- G2 policy 字段、坐标和采样语义：[`policy-config.md`](policy-config.md)
- wholebody raw 字段：`../wholebody/docs/user/collect/schema.md`
- wholebody raw 导出：`../wholebody/docs/user/collect/export.md`
- OpenPI G2 dataset adapter：`../openpi/src/openpi/adapters/g2/`
- LeRobot G2 dataset adapter：`../lerobot-finetune/src/lerobot_finetune/adapters/g2/`
