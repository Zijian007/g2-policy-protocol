"""Encode canonical wire states/actions into a policy's model space and decode them back.

Wire values are absolute full dual-arm pose20 / joint16. The model space follows
the policy branch: selected arm, coordinates, rotation and optional delta.
"""

import dataclasses

import numpy as np

from .._constants import JOINT_ARM_DIM, JOINT_DIM, POSE_DIM
from .geometry import arm_slice, from_policy_frame, matrix_to_pose, pose_to_matrix, pose_width, to_policy_frame
from .policy import QUANTITIES, active_action

_WIRE_DIMS = {"pose": POSE_DIM, "joint": JOINT_DIM}
_WIRE_NAMES = {"pose": "EEF pose", "joint": "joint vector"}


def action_dimensions(policy, arm="both"):
    """(model_dim, wire_dim) of the active action branch for the selected arm."""
    kind, branch = active_action(policy)
    quantity = QUANTITIES[kind].get(branch["quantity"])
    arms = 2 if arm == "both" else 1
    if quantity == "pose_target":
        return pose_width(branch["coordinates"]["rotation"]) * arms, POSE_DIM
    if quantity == "joint_position":
        return JOINT_ARM_DIM * arms, JOINT_DIM
    raise ValueError(f"Unsupported G2 control quantity: {branch['quantity']!r}")


def reference_vector(obs, kind, source):
    """Full measured (source=state) or commanded (source=action) pose20 / joint16."""
    key = f"observation/{kind}" if source == "state" else f"observation/command_{kind}"
    if key not in obs:
        raise ValueError(f"Reference {source} requires {key}")
    width = _WIRE_DIMS[kind]
    result = np.asarray(obs[key], dtype=np.float32)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{key} must be a finite full ({width},) {_WIRE_NAMES[kind]}")
    return result


def policy_frame(coordinates, obs):
    """Branch coordinates plus this request's base_to_parent when the parent is arm_base_link."""
    frame = dict(coordinates)
    if coordinates["parent_frame"] == "arm_base_link":
        if "observation/base_to_parent" not in obs:
            raise ValueError("arm_base_link requires observation/base_to_parent")
        frame["base_to_parent"] = obs["observation/base_to_parent"]
    return frame


@dataclasses.dataclass(frozen=True)
class ActionCodec:
    """Encodes wire actions (labels, RTC prefixes) and decodes model outputs, for one arm selection."""

    policy: dict
    arm: str = "both"

    @property
    def dimensions(self):
        """(model_dim, wire_dim)."""
        return action_dimensions(self.policy, self.arm)

    def encode(self, actions, obs):
        """(T, wire_dim) absolute wire actions -> (T, model_dim) model targets."""
        kind, branch = active_action(self.policy)
        _, wire_dim = self.dimensions
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[-1] != wire_dim or not np.isfinite(values).all():
            raise ValueError(f"Actions must be finite (T, {wire_dim}) wire values")
        selection = arm_slice(self.arm, wire_dim)
        values = values[..., selection]
        if kind == "pose":
            return _encode_pose(values, obs, branch, self.arm)
        if branch["delta"]:
            reference = reference_vector(obs, "joint", branch["reference"]["source"])[selection]
            return _joint_delta(values, reference, branch, inverse=False)
        return values.copy()

    def decode(self, actions, obs):
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
        measured = reference_vector(obs, kind, "state") if self.arm != "both" else None
        if branch["delta"]:
            source = branch["reference"]["source"]
            reference = (measured if measured is not None and source == "state"
                         else reference_vector(obs, kind, source))[selection]
        if kind == "pose":
            frame = policy_frame(branch["coordinates"], obs)
            if branch["delta"]:
                reference = to_policy_frame(reference, frame, arm=self.arm)
                values = _pose_delta(values, reference, branch, frame["rotation"], inverse=True)
            values = from_policy_frame(values, frame, arm=self.arm)
        elif branch["delta"]:
            values = _joint_delta(values, reference, branch, inverse=True)
        if measured is None:
            return values
        full = np.broadcast_to(measured, (len(values), wire_dim)).copy()
        full[..., selection] = values
        return full


def encode_state(policy, observation, arm="both", *, flatten=True):
    """Encode measured state histories; delta references are fixed at observation time.

    ``flatten=False`` keeps (history_steps, dims) for temporal policies.
    """
    history_steps = policy["timing"]["history_steps"]
    states = []
    for kind in ("pose", "joint"):
        branch = policy["state"][kind]
        if not branch["enabled"]:
            continue
        width = _WIRE_DIMS[kind]
        values = np.asarray(
            observation.get(f"observation/{kind}_history", observation[f"observation/{kind}"]), dtype=np.float32
        )
        if values.ndim == 1:
            values = np.tile(values, (history_steps, 1))
        if values.shape != (history_steps, width):
            raise ValueError(f"{kind} history must have shape {(history_steps, width)}")
        if not np.isfinite(values).all():
            raise ValueError(f"{kind} history must contain finite values")
        selection = arm_slice(arm, width)
        values = values[..., selection]
        if kind == "pose":
            values = _encode_pose(values, observation, branch, arm)
        elif branch["delta"]:
            reference = reference_vector(observation, "joint", branch["reference"]["source"])[selection]
            values = _joint_delta(values, reference, branch, inverse=False)
        states.append(values)
    result = np.concatenate(states, axis=-1)
    return result.reshape(-1) if flatten else result


def _encode_pose(values, obs, branch, arm):
    frame = policy_frame(branch["coordinates"], obs)
    values = to_policy_frame(values, frame, arm=arm)
    if not branch["delta"]:
        return values
    reference = reference_vector(obs, "pose", branch["reference"]["source"])[arm_slice(arm, POSE_DIM)]
    return _pose_delta(values, to_policy_frame(reference, frame, arm=arm), branch, frame["rotation"], inverse=False)


def _pose_delta(values, reference, branch, rotation, *, inverse):
    width, scope = pose_width(rotation), branch["reference"]["scope"]
    if branch["method"] == "component":
        return _component_delta(values, reference, width, scope, inverse=inverse)
    return _se3_delta(values, reference, width, rotation, scope, inverse=inverse)


def _joint_delta(values, reference, branch, *, inverse):
    return _component_delta(values, reference, JOINT_ARM_DIM, branch["reference"]["scope"], inverse=inverse)


def _component_delta(values, reference, arm_width, scope, *, inverse):
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


def _se3_delta(values, reference, arm_width, rotation, scope, *, inverse):
    """Express each arm's pose in the reference pose's frame (inverse: compose back)."""
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
