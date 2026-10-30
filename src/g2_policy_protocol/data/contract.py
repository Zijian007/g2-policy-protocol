"""G2 LeRobot on-disk contract. Standard library only.

Shared by dataset writers and policy adapters. Do not import wholebody or openpi.

Measured vectors are ``state.joint`` / ``state.pose``. Commands are
``action.joint`` / ``action.pose``. Camera videos stay ``observation.images.*``
(LeRobot video convention). Raw ``frames.jsonl`` paths are not in this file.
"""

from __future__ import annotations

from dataclasses import dataclass

SIDES: tuple[str, ...] = ("left", "right")


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

    head: str = "head_color"
    hand_left: str = "hand_left_color"
    hand_right: str = "hand_right_color"
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
class G2LerobotContract:
    """Column names + vector layouts + camera keys for one G2 LeRobot export."""

    schema_id: str = "g2_lerobot_keys_v3"
    rotation: str = "rot6d"

    columns: DiskColumnKeys = DiskColumnKeys()
    joint: VectorLayout = VectorLayout(
        names=tuple(f"{side}_{name}" for side in SIDES for name in (*(f"joint_{i}" for i in range(1, 8)), "gripper")),
        per_side_dim=8,
        gripper_indices=(7, 15),
    )
    pose: VectorLayout = VectorLayout(
        names=tuple(
            f"{side}_{name}" for side in SIDES for name in ("x", "y", "z", *(f"rot6d_{i}" for i in range(6)), "gripper")
        ),
        per_side_dim=10,
        gripper_indices=(9, 19),
    )
    cameras: CameraKeys = CameraKeys()


G2 = G2LerobotContract()

__all__ = [
    "G2",
    "SIDES",
    "CameraKeys",
    "DiskColumnKeys",
    "G2LerobotContract",
    "VectorLayout",
]
