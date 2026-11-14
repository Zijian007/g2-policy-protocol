"""Robot state/actions <-> model vectors, using one resolved policy.

State encoding selects arms, converts coordinates/rotation, then applies deltas.
Action encoding follows the same rules; decoding restores absolute robot targets
in base_link/flange/rot6d and fills inactive arms from measured state.
Grippers stay absolute. Histories are oldest first, including the current frame.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from g2_policy_protocol.contracts.policy import (
    QUANTITIES,
    ArmSelection,
    Coordinates,
    PolicySemantics,
    active_action,
)
from g2_policy_protocol.types import JOINT_ARM_DIM, JOINT_DIM, POSE_DIM

from .geometry import (
    arm_slice,
    from_policy_frame,
    matrix_to_pose,
    pose_to_matrix,
    pose_width,
    to_policy_frame,
)

_WIRE_DIMS = {"pose": POSE_DIM, "joint": JOINT_DIM}
_WIRE_NAMES = {"pose": "EEF pose", "joint": "joint vector"}


References = TypedDict(
    "References",
    {
        "observation/pose": NDArray,
        "observation/joint": NDArray,
        "observation/pose_history": NDArray,
        "observation/joint_history": NDArray,
        "observation/command_pose": NDArray,
        "observation/command_joint": NDArray,
        "observation/base_to_parent": NDArray,
    },
    total=False,
)


def encode_state(
    policy: PolicySemantics,
    observation: Mapping[str, ArrayLike],
    arm: ArmSelection = "both",
    *,
    flatten: bool = True,
) -> NDArray:
    """Encode measured state histories; delta references are fixed at observation time.

    Each enabled modality uses its history when provided, otherwise repeats the
    current measured vector. Delta encoding also needs its declared reference.
    ``flatten=False`` keeps (history_steps, dims) for temporal policies.
    """
    history_steps = policy["timing"]["history_steps"]
    states = []
    for kind in ("pose", "joint"):
        branch = policy["state"][kind]
        if not branch["enabled"]:
            continue
        width = _WIRE_DIMS[kind]
        history_key = f"observation/{kind}_history"
        key = history_key if history_key in observation else f"observation/{kind}"
        values = np.asarray(observation[key], dtype=np.float32)
        if values.ndim == 1:
            values = np.tile(values, (history_steps, 1))
        if values.shape != (history_steps, width):
            raise ValueError(f"{kind} history must have shape {(history_steps, width)}")
        if not np.isfinite(values).all():
            raise ValueError(f"{kind} history must contain finite values")
        selection = arm_slice(arm, width)
        values = values[..., selection]
        if kind == "pose":
            values = _encode_pose_values(values, observation, branch, arm)
        elif branch["delta"]:
            reference = _read_reference_vector(observation, "joint", branch["reference"]["source"])[selection]
            values = _apply_joint_delta(values, reference, branch, inverse=False)
        states.append(values)
    result = np.concatenate(states, axis=-1)
    return result.reshape(-1) if flatten else result


def references_from_sample(sample: Mapping[str, Any]) -> References:
    """Map disk/sampler references to the shared geometry dictionary."""
    sources = {
        "state.pose": "pose",
        "state.joint": "joint",
        "g2.pose_history": "pose_history",
        "g2.joint_history": "joint_history",
        "g2.command_pose": "command_pose",
        "g2.command_joint": "command_joint",
        "g2.base_to_parent": "base_to_parent",
    }
    return {
        f"observation/{target}": np.asarray(sample[source]) for source, target in sources.items() if source in sample
    }


def state_width(policy: PolicySemantics, arm: ArmSelection = "both") -> int:
    arms = 2 if arm == "both" else 1
    width = 0
    if policy["state"]["pose"]["enabled"]:
        width += pose_width(policy["state"]["pose"]["coordinates"]["rotation"])
    if policy["state"]["joint"]["enabled"]:
        width += JOINT_ARM_DIM
    return width * arms * policy["timing"]["history_steps"]


def action_dimensions(policy: PolicySemantics, arm: ArmSelection = "both") -> tuple[int, int]:
    """(model_dim, wire_dim) of the active action branch for the selected arm."""
    kind, branch = active_action(policy)
    quantity = QUANTITIES[kind].get(branch["quantity"])
    arms = 2 if arm == "both" else 1
    if quantity == "pose_target":
        return pose_width(branch["coordinates"]["rotation"]) * arms, POSE_DIM
    if quantity == "joint_position":
        return JOINT_ARM_DIM * arms, JOINT_DIM
    raise ValueError(f"Unsupported G2 control quantity: {branch['quantity']!r}")


@dataclasses.dataclass(frozen=True)
class ActionCodec:
    """Encodes wire actions (labels, RTC prefixes) and decodes model outputs, for one arm selection."""

    policy: PolicySemantics
    arm: ArmSelection = "both"

    @property
    def dimensions(self) -> tuple[int, int]:
        """(model_dim, wire_dim)."""
        return action_dimensions(self.policy, self.arm)

    def encode(self, actions: ArrayLike, obs: Mapping[str, ArrayLike]) -> NDArray:
        """(T, wire_dim) absolute wire actions -> (T, model_dim) model targets."""
        kind, branch = active_action(self.policy)
        _, wire_dim = self.dimensions
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[-1] != wire_dim or not np.isfinite(values).all():
            raise ValueError(f"Actions must be finite (T, {wire_dim}) wire values")
        selection = arm_slice(self.arm, wire_dim)
        values = values[..., selection]
        if kind == "pose":
            return _encode_pose_values(values, obs, branch, self.arm)
        if branch["delta"]:
            reference = _read_reference_vector(obs, "joint", branch["reference"]["source"])[selection]
            return _apply_joint_delta(values, reference, branch, inverse=False)
        return values.copy()

    def decode(self, actions: ArrayLike, obs: Mapping[str, ArrayLike]) -> NDArray:
        """(T, ≥model_dim) model outputs -> (T, wire_dim) absolute wire actions.

        Extra trailing columns (model padding) are dropped. A single-arm policy
        fills the other arm with its measured values.
        """
        kind, branch = active_action(self.policy)
        model_dim, wire_dim = self.dimensions
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or not len(values) or values.shape[-1] < model_dim or not np.isfinite(values).all():
            raise ValueError(f"Model actions must be finite with at least {model_dim} columns")
        values = values[..., :model_dim].copy()
        selection = arm_slice(self.arm, wire_dim)
        measured = _read_reference_vector(obs, kind, "state") if self.arm != "both" else None
        if branch["delta"]:
            source = branch["reference"]["source"]
            reference = (
                measured if measured is not None and source == "state" else _read_reference_vector(obs, kind, source)
            )[selection]
        if kind == "pose":
            frame = _policy_frame(branch["coordinates"], obs)
            if branch["delta"]:
                reference = to_policy_frame(reference, frame, arm=self.arm)
                values = _apply_pose_delta(values, reference, branch, frame["rotation"], inverse=True)
            values = from_policy_frame(values, frame, arm=self.arm)
        elif branch["delta"]:
            values = _apply_joint_delta(values, reference, branch, inverse=True)
        if measured is None:
            return values
        full = np.broadcast_to(measured, (len(values), wire_dim)).copy()
        full[..., selection] = values
        return full


def _read_reference_vector(obs: Mapping[str, ArrayLike], kind: str, source: str) -> NDArray:
    """Full measured (source=state) or commanded (source=action) pose20 / joint16."""
    key = f"observation/{kind}" if source == "state" else f"observation/command_{kind}"
    if key not in obs:
        raise ValueError(f"Reference {source} requires {key}")
    width = _WIRE_DIMS[kind]
    result = np.asarray(obs[key], dtype=np.float32)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{key} must be a finite full ({width},) {_WIRE_NAMES[kind]}")
    return result


def _policy_frame(coordinates: Coordinates, obs: Mapping[str, ArrayLike]) -> dict[str, Any]:
    """Branch coordinates plus this request's base_to_parent when the parent is arm_base_link."""
    frame = dict(coordinates)
    if coordinates["parent_frame"] == "arm_base_link":
        if "observation/base_to_parent" not in obs:
            raise ValueError("arm_base_link requires observation/base_to_parent")
        frame["base_to_parent"] = obs["observation/base_to_parent"]
    return frame


def _encode_pose_values(values, obs, branch, arm):
    frame = _policy_frame(branch["coordinates"], obs)
    values = to_policy_frame(values, frame, arm=arm)
    if not branch["delta"]:
        return values
    reference = _read_reference_vector(obs, "pose", branch["reference"]["source"])[arm_slice(arm, POSE_DIM)]
    return _apply_pose_delta(
        values, to_policy_frame(reference, frame, arm=arm), branch, frame["rotation"], inverse=False
    )


def _apply_pose_delta(values, reference, branch, rotation, *, inverse):
    """Encode relative poses; inverse=True restores absolute poses in the same frame.

    component subtracts vector entries. se3 encodes inv(T_reference) @ T_target.
    chunk uses one observation-time reference; step uses each previous target.
    Coordinates and rotation representation must already match on both inputs.
    """
    width, scope = pose_width(rotation), branch["reference"]["scope"]
    if branch["method"] == "component":
        return _apply_component_delta(values, reference, width, scope, inverse=inverse)
    return _apply_se3_delta(values, reference, width, rotation, scope, inverse=inverse)


def _apply_joint_delta(values, reference, branch, *, inverse):
    """Subtract joint references, or add them back; retain absolute grippers."""
    return _apply_component_delta(values, reference, JOINT_ARM_DIM, branch["reference"]["scope"], inverse=inverse)


def _apply_component_delta(values, reference, arm_width, scope, *, inverse):
    """Subtract (inverse: add back) the reference on every component except the grippers.

    ``chunk`` scope uses one reference for all steps; ``step`` uses the previous target.
    """
    mask = np.ones(len(reference), dtype=bool)
    mask[arm_width - 1 :: arm_width] = False
    step = scope == "step"
    out = values.copy()
    if inverse:
        out[..., mask] = (np.cumsum(values[..., mask], axis=0) if step else values[..., mask]) + reference[mask]
    else:
        references = np.concatenate([reference[None], values[:-1]], axis=0) if step else reference
        out[..., mask] -= references[..., mask]
    return out


def _apply_se3_delta(values, reference, arm_width, rotation, scope, *, inverse):
    """Express each arm's pose in the reference pose's frame (inverse: compose back)."""
    if scope == "chunk" and len(values):
        # All timesteps share one reference per arm. Convert the entire chunk at
        # once instead of reconstructing/inverting that reference at every step.
        poses = values.reshape(len(values), -1, arm_width)
        targets = pose_to_matrix(poses, rotation)
        bases = pose_to_matrix(reference.reshape(-1, arm_width), rotation)
        transforms = (bases if inverse else np.linalg.inv(bases)) @ targets
        return matrix_to_pose(transforms, poses[..., -1], rotation).reshape(values.shape)

    # Step decoding is recurrent, including projection back onto SO(3) after each
    # decoded pose. Keep that sequence and its numerical behavior unchanged.
    out = []
    current = reference.copy()
    for value in values:
        parts = []
        for offset in range(0, len(reference), arm_width):
            pose = value[offset : offset + arm_width]
            target = pose_to_matrix(pose, rotation)
            base = pose_to_matrix(current[offset : offset + arm_width], rotation)
            transform = base @ target if inverse else np.linalg.inv(base) @ target
            parts.append(matrix_to_pose(transform, pose[-1], rotation))
        result = np.concatenate(parts)
        out.append(result)
        if scope == "step":
            current = result if inverse else value
    return np.asarray(out, dtype=np.float32)
