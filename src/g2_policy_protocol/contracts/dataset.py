"""G2 LeRobot on-disk contract. Standard library only.

Shared by dataset writers and policy adapters. Do not import wholebody or openpi.

Measured vectors are ``state.joint`` / ``state.pose``. Commands are
``action.joint`` / ``action.pose``. Camera videos stay ``observation.images.*``
(LeRobot video convention). Raw ``frames.jsonl`` paths are not in this file.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from g2_policy_protocol.types import (
    CAMERA_IDS,
    EE_FRAME,
    FRAME,
    JOINT_ARM_DIM,
    JOINT_DIM,
    JOINT_FIELDS,
    JOINT_SIDE_UNITS,
    POSE_ARM_DIM,
    POSE_DIM,
    POSE_FIELDS,
    POSE_SIDE_UNITS,
    SIDES,
)

from .policy import (
    KINDS,
    PolicySemantics,
    active_action,
    command_reference_kinds,
    uses_arm_base,
    uses_kind,
)


@dataclass(frozen=True, slots=True)
class VectorLayout:
    """Fixed-size float vector: ``names[i]`` is component ``i``.

    Concatenation order is ``sides`` (default left → right).
    """

    names: tuple[str, ...]
    per_side_dim: int
    gripper_indices: tuple[int, ...]
    sides: tuple[str, ...] = SIDES

    @property
    def dim(self) -> int:
        return len(self.names)

    def describe(self) -> dict:
        return {
            "dim": self.dim,
            "names": list(self.names),
            "sides": list(self.sides),
            "per_side_dim": self.per_side_dim,
            "gripper_index": list(self.gripper_indices),
        }


@dataclass(frozen=True, slots=True)
class DiskColumnKeys:
    """LeRobot parquet / MP4 column names (v2.1 and v3 share these strings)."""

    state_joint: str = "state.joint"
    state_pose: str = "state.pose"
    state_torso: str = "state.torso"
    state_head: str = "state.head"
    state_arm_base: str = "state.arm_base"
    action_pose: str = "action.pose"
    action_joint: str = "action.joint"
    action_torso: str = "action.torso"
    action_head: str = "action.head"
    qa_action_pose_minus_state_pose: str = "qa.action_pose_minus_state_pose"

    def vector_columns(self) -> tuple[str, ...]:
        return (
            self.state_joint,
            self.state_pose,
            self.action_pose,
            self.action_joint,
        )


@dataclass(frozen=True, slots=True)
class CameraKeys:
    """Camera short names (JPEG dir) → LeRobot video feature keys."""

    head: str = CAMERA_IDS[0]
    hand_left: str = CAMERA_IDS[1]
    hand_right: str = CAMERA_IDS[2]
    roles: tuple[str, ...] = ("cam_head", "cam_wrist_l", "cam_wrist_r")
    images_prefix: str = "observation.images"

    @property
    def shorts(self) -> tuple[str, ...]:
        return (self.head, self.hand_left, self.hand_right)

    def feature_key(self, short_name: str) -> str:
        return f"{self.images_prefix}.{short_name}"

    @property
    def feature_keys(self) -> tuple[str, ...]:
        return tuple(self.feature_key(short) for short in self.shorts)

    def role_map(self) -> dict[str, str]:
        return {role: self.feature_key(short) for role, short in zip(self.roles, self.shorts, strict=True)}


@dataclass(frozen=True, slots=True)
class DatasetColumn:
    """Numeric column name and component layout in G2 datasets."""

    role: str
    key: str
    names: tuple[str, ...]
    units: tuple[str, ...]
    dtype: str = "float32"
    frame: str | None = None


TORSO_FIELDS = (
    "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
    "idx04_body_joint4", "idx05_body_joint5",
)
HEAD_FIELDS = ("idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3")
ARM_BASE_FIELDS = ("arm_base_x", "arm_base_y", "arm_base_z", *(f"arm_base_rot6d_{i}" for i in range(6)))


@dataclass(frozen=True, slots=True)
class G2LerobotContract:
    """Column names + vector layouts + camera keys for one G2 LeRobot export."""

    schema_id: str = "g2_lerobot_keys_v3"
    rotation: str = "rot6d"
    pose_frame: str = FRAME
    ee_frame: str = EE_FRAME

    columns: DiskColumnKeys = DiskColumnKeys()
    joint: VectorLayout = VectorLayout(
        names=JOINT_FIELDS,
        per_side_dim=JOINT_ARM_DIM,
        gripper_indices=tuple(range(JOINT_ARM_DIM - 1, JOINT_DIM, JOINT_ARM_DIM)),
    )
    pose: VectorLayout = VectorLayout(
        names=POSE_FIELDS,
        per_side_dim=POSE_ARM_DIM,
        gripper_indices=tuple(range(POSE_ARM_DIM - 1, POSE_DIM, POSE_ARM_DIM)),
    )
    cameras: CameraKeys = CameraKeys()

    def numeric_columns(self) -> tuple[DatasetColumn, ...]:
        cols = self.columns
        joint_units = JOINT_SIDE_UNITS * len(self.joint.sides)
        pose_units = POSE_SIDE_UNITS * len(self.pose.sides)
        torso_units = ("rad",) * len(TORSO_FIELDS)
        head_units = ("rad",) * len(HEAD_FIELDS)
        return (
            DatasetColumn("state_joint", cols.state_joint, self.joint.names, joint_units),
            DatasetColumn("action_pose", cols.action_pose, self.pose.names, pose_units, frame=self.pose_frame),
            DatasetColumn("action_joint", cols.action_joint, self.joint.names, joint_units),
            DatasetColumn("state_pose", cols.state_pose, self.pose.names, pose_units, frame=self.pose_frame),
            DatasetColumn("state_torso", cols.state_torso, TORSO_FIELDS, torso_units),
            DatasetColumn("action_torso", cols.action_torso, TORSO_FIELDS, torso_units),
            DatasetColumn("state_head", cols.state_head, HEAD_FIELDS, head_units),
            DatasetColumn("action_head", cols.action_head, HEAD_FIELDS, head_units),
            DatasetColumn("state_arm_base", cols.state_arm_base, ARM_BASE_FIELDS, POSE_SIDE_UNITS[:-1], frame=self.pose_frame),
        )


def recorded_video_keys(
    features: Mapping[str, Any] | None, preferred: Sequence[str],
) -> tuple[str, ...]:
    """Read actual video features, keeping the preferred camera order when present."""
    if features is None:
        return tuple(preferred)
    video = tuple(key for key, spec in features.items() if isinstance(spec, Mapping) and spec.get("dtype") == "video")
    return tuple(key for key in dict.fromkeys((*preferred, *video)) if key in video)


G2 = G2LerobotContract()

__all__ = [
    "G2",
    "SIDES",
    "CameraKeys",
    "DiskColumnKeys",
    "DatasetColumn",
    "G2LerobotContract",
    "VectorLayout",
    "validate_dataset_contract",
    "recorded_video_keys",
]


_COLUMN_WIDTHS = {column.key: len(column.names) for column in G2.numeric_columns()}

def action_label_column(policy: PolicySemantics) -> str:
    """Disk column that holds the training action labels."""
    kind, branch = active_action(policy)
    return f"{branch['source']}.{kind}"


def validate_dataset_contract(
    features: Mapping[str, Mapping[str, Any]],
    dataset_metadata: Mapping[str, Any],
    policy: PolicySemantics,
    cameras: Sequence[str],
) -> None:
    """Check that a LeRobot dataset records everything the policy needs, in the expected frames."""
    action = policy["action"]
    _, label_branch = active_action(policy)
    required = {*cameras, action_label_column(policy)}
    required |= {f"state.{kind}" for kind in KINDS if uses_kind(policy, kind)}
    required |= {f"action.{kind}" for kind in command_reference_kinds(policy)}
    if uses_arm_base(policy):
        required.add(G2.columns.state_arm_base)
    missing = required - features.keys()
    if missing:
        raise ValueError(f"G2 dataset is missing required columns: {sorted(missing)}")
    for key in required & _COLUMN_WIDTHS.keys():
        if "shape" in features[key] and tuple(features[key]["shape"]) != (_COLUMN_WIDTHS[key],):
            raise ValueError(f"G2 dataset {key} must have shape [{_COLUMN_WIDTHS[key]}]")
    if "fps" in dataset_metadata:
        fps = dataset_metadata["fps"]
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Dataset fps must be a positive finite number")
    exported = dataset_metadata.get("wholebody_export", {})
    for field, expected in (("ee_frame", G2.ee_frame), ("rotation", G2.rotation)):
        if field in exported and exported[field] != expected:
            raise ValueError(f"Dataset wholebody_export.{field}={exported[field]!r} must match {expected!r}")
    pose_frame = exported.get("pose_frame", G2.pose_frame)
    action_parent = exported.get("action_parent_frame", pose_frame)
    if uses_kind(policy, "pose") and pose_frame != G2.pose_frame:
        raise ValueError(f"Dataset wholebody_export.pose_frame={pose_frame!r} must be {G2.pose_frame!r}")
    reads_command_pose = "pose" in command_reference_kinds(policy) or (
        action["pose"]["enabled"] and action["pose"]["source"] == "action"
    )
    if reads_command_pose and action_parent != G2.pose_frame:
        raise ValueError(f"Dataset wholebody_export.action_parent_frame={action_parent!r} must be {G2.pose_frame!r}")
    if label_branch["source"] == "action" and exported.get("action_supervision", "command") != "command":
        raise ValueError("Dataset wholebody_export.action_supervision must match action labels")
    if exported.get("segment_policy") == "concatenated":
        raise ValueError("Dataset concatenates capture segments; export separate continuous episodes")
