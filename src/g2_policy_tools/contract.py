"""G2 LeRobot on-disk contract. Standard library only.

Shared by dataset writers and policy adapters. Do not import wholebody or openpi.

Measured vectors are ``state.joint`` / ``state.pose``. Commands are
``action.joint`` / ``action.pose``. Camera videos stay ``observation.images.*``
(LeRobot video convention). Raw ``frames.jsonl`` paths are not in this file.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field

SIDES: tuple[str, ...] = ("left", "right")


def _arm_joint_names(side: str) -> tuple[str, ...]:
    return (*(f"{side}_joint_{i}" for i in range(1, 8)), f"{side}_gripper")


def _arm_pose_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_x",
        f"{side}_y",
        f"{side}_z",
        *(f"{side}_rot6d_{i}" for i in range(6)),
        f"{side}_gripper",
    )


def _concat_sides(
    per_side: Callable[[str], Sequence[str]],
    sides: Sequence[str] = SIDES,
) -> tuple[str, ...]:
    out: list[str] = []
    for side in sides:
        out.extend(per_side(side))
    return tuple(out)


def _gripper_indices(names: Sequence[str]) -> tuple[int, ...]:
    return tuple(i for i, name in enumerate(names) if name.endswith("_gripper"))


_JOINT_NAMES = _concat_sides(_arm_joint_names)
_POSE_NAMES = _concat_sides(_arm_pose_names)


@dataclass(frozen=True, slots=True)
class VectorLayout:
    """Fixed-size float vector: ``names[i]`` is component ``i``.

    Concatenation order is ``sides`` (default left → right).
    """

    names: tuple[str, ...]
    per_side_dim: int
    sides: tuple[str, ...] = SIDES
    gripper_indices: tuple[int, ...] = dc_field(default_factory=tuple)

    def __post_init__(self) -> None:
        expected = self.per_side_dim * len(self.sides)
        if len(self.names) != expected:
            raise ValueError(
                f"layout has {len(self.names)} names, expected {expected} "
                f"({self.per_side_dim} per side x {len(self.sides)} sides)"
            )

    @property
    def dim(self) -> int:
        return len(self.names)

    def side_slice(self, side: str) -> slice:
        if side not in self.sides:
            raise KeyError(f"unknown side {side!r}; expected one of {self.sides}")
        start = self.sides.index(side) * self.per_side_dim
        return slice(start, start + self.per_side_dim)

    def describe(self) -> dict:
        payload: dict = {
            "dim": self.dim,
            "names": list(self.names),
            "sides": list(self.sides),
            "per_side_dim": self.per_side_dim,
        }
        if self.gripper_indices:
            payload["gripper_index"] = list(self.gripper_indices)
        return payload

    def delta_mask(self) -> tuple[bool, ...]:
        """True where that action dim is a delta. Gripper indices stay absolute."""
        grips = set(self.gripper_indices)
        return tuple(index not in grips for index in range(self.dim))


def _default_joint_layout() -> VectorLayout:
    return VectorLayout(
        names=_JOINT_NAMES,
        per_side_dim=8,
        gripper_indices=_gripper_indices(_JOINT_NAMES),
    )


def _default_pose_layout() -> VectorLayout:
    return VectorLayout(
        names=_POSE_NAMES,
        per_side_dim=10,
        gripper_indices=_gripper_indices(_POSE_NAMES),
    )


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

    columns: DiskColumnKeys = dc_field(default_factory=DiskColumnKeys)
    joint: VectorLayout = dc_field(default_factory=_default_joint_layout)
    pose: VectorLayout = dc_field(default_factory=_default_pose_layout)
    cameras: CameraKeys = dc_field(default_factory=CameraKeys)


G2 = G2LerobotContract()

__all__ = [
    "G2",
    "SIDES",
    "CameraKeys",
    "DiskColumnKeys",
    "G2LerobotContract",
    "VectorLayout",
]
