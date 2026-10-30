"""Shared G2 geometry and state/action encoding."""

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

from .policy import QUANTITIES, action_branch


def rotation_matrix(value, representation="rot6d"):
    """Project rot6d onto SO(3); quaternions use xyzw, axis angles use radians."""
    value = np.asarray(value, dtype=np.float64)
    shape = value.shape[:-1]
    if representation == "rot6d":
        a, b = value[..., :3], value[..., 3:6]
        norm = np.linalg.norm(a, axis=-1, keepdims=True)
        x = np.where(norm > 1e-8, a / np.maximum(norm, 1e-8), np.array([1.0, 0.0, 0.0]))
        b = b - np.sum(x * b, axis=-1, keepdims=True) * x
        fallback = np.eye(3)[np.argmin(np.abs(x), axis=-1)]
        fallback -= np.sum(x * fallback, axis=-1, keepdims=True) * x
        b = np.where(np.linalg.norm(b, axis=-1, keepdims=True) > 1e-8, b, fallback)
        y = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
        return np.stack([x, y, np.cross(x, y)], axis=-1)
    if representation == "quaternion":
        q = value.reshape(-1, 4).copy()
        q[np.linalg.norm(q, axis=-1) < 1e-8] = [0, 0, 0, 1]
        return Rotation.from_quat(q).as_matrix().reshape(*shape, 3, 3)
    return Rotation.from_rotvec(value.reshape(-1, 3)).as_matrix().reshape(*shape, 3, 3)


def rotation_vector(matrix, representation="rot6d"):
    matrix = np.asarray(matrix)
    shape = matrix.shape[:-2]
    if representation == "rot6d":
        return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)
    r = Rotation.from_matrix(matrix.reshape(-1, 3, 3))
    result = r.as_quat(canonical=True) if representation == "quaternion" else r.as_rotvec()
    return result.reshape(*shape, result.shape[-1])


def pose_matrix(pose, representation="rot6d"):
    pose = np.asarray(pose)
    result = np.broadcast_to(np.eye(4), (*pose.shape[:-1], 4, 4)).copy()
    result[..., :3, :3] = rotation_matrix(pose[..., 3:-1], representation)
    result[..., :3, 3] = pose[..., :3]
    return result


def matrix_pose(matrix, gripper, representation="rot6d"):
    return np.concatenate(
        [
            matrix[..., :3, 3],
            rotation_vector(matrix[..., :3, :3], representation),
            np.asarray(gripper)[..., None],
        ],
        axis=-1,
    ).astype(np.float32)


def arm_slice(arm, dim):
    half = dim // 2
    return slice(None) if arm == "both" else slice(0, half) if arm == "left" else slice(half, dim)


def convert_pose(pose, settings, *, inverse=False, arm="both"):
    """Canonical base/flange rot6d ↔ configured parent/EE/rotation, per selected arm."""
    pose = np.asarray(pose)
    if (
        settings["rotation"] == "rot6d"
        and settings.get("base_to_parent") is None
        and settings.get("flange_to_tcp") is None
    ):
        return pose.copy()
    width = 10 if not inverse else 4 + {"rot6d": 6, "quaternion": 4, "axis_angle": 3}[settings["rotation"]]
    sides = ("left", "right") if arm == "both" else (arm,)
    parent = np.asarray(settings["base_to_parent"] if settings.get("base_to_parent") is not None else np.eye(4))
    if inverse:
        parent = np.linalg.inv(parent)
    pieces = []
    for index, side in enumerate(sides):
        value = pose[..., index * width : (index + 1) * width]
        tcp = settings.get("flange_to_tcp")
        tcp = np.eye(4) if tcp is None else np.asarray(tcp[side])
        if inverse:
            transform = parent @ pose_matrix(value, settings["rotation"]) @ np.linalg.inv(tcp)
            pieces.append(matrix_pose(transform, value[..., -1]))
        else:
            transform = parent @ pose_matrix(value) @ tcp
            pieces.append(matrix_pose(transform, value[..., -1], settings["rotation"]))
    return np.concatenate(pieces, axis=-1)


def pose_settings(coordinates, obs):
    settings = dict(coordinates)
    if coordinates["parent_frame"] == "arm_base_link":
        if "observation/base_to_parent" not in obs:
            raise ValueError("arm_base_link requires observation/base_to_parent")
        settings["base_to_parent"] = obs["observation/base_to_parent"]
    return settings


def action_dimensions(policy, arm="both"):
    kind, spec = action_branch(policy)
    quantity = QUANTITIES[kind].get(spec["quantity"])
    if quantity == "pose_target":
        width = 4 + {"rot6d": 6, "quaternion": 4, "axis_angle": 3}[spec["coordinates"]["rotation"]]
        return width * (2 if arm == "both" else 1), 20
    if quantity in ("joint_position", "joint_velocity", "joint_torque"):
        return 8 * (2 if arm == "both" else 1), 16
    raise ValueError(f"Unsupported G2 control quantity: {spec['quantity']!r}")


def source_pose(obs, source):
    key = "observation/pose" if source == "state" else "observation/command_pose"
    if key not in obs:
        raise ValueError(f"Reference {source} requires {key}")
    result = np.asarray(obs[key], dtype=np.float32)
    if result.shape != (20,) or not np.isfinite(result).all():
        raise ValueError(f"{key} must be a finite full (20,) EEF pose")
    return result


def source_joint(obs, source):
    key = "observation/joint" if source == "state" else "observation/command_joint"
    if key not in obs:
        raise ValueError(f"Reference {source} requires {key}")
    result = np.asarray(obs[key], dtype=np.float32)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError(f"{key} must be a finite full (16,) joint vector")
    return result


def encode_poses(values, obs, spec, arm="both"):
    settings = pose_settings(spec["coordinates"], obs)
    values = convert_pose(values, settings, arm=arm)
    if not spec["delta"]:
        return values
    reference = source_pose(obs, spec["reference"]["source"])[arm_slice(arm, 20)]
    ref = convert_pose(reference, settings, arm=arm)
    return pose_delta(values, ref, spec, settings["rotation"], inverse=False)


def pose_delta(values, ref, spec, rotation, *, inverse):
    width = 4 + {"rot6d": 6, "quaternion": 4, "axis_angle": 3}[rotation]
    step = spec["reference"]["scope"] == "step"
    if spec["method"] == "component":
        mask = np.ones(len(ref), dtype=bool)
        mask[width - 1 :: width] = False
        out = values.copy()
        if inverse:
            out[..., mask] = (np.cumsum(values[..., mask], axis=0) if step else values[..., mask]) + ref[mask]
        else:
            refs = np.concatenate([ref[None], values[:-1]], axis=0) if step else ref
            out[..., mask] -= refs[..., mask]
        return out
    out = []
    current_reference = ref.copy()
    for value in values:
        parts = []
        for offset in range(0, len(ref), width):
            pose, reference_pose = value[offset : offset + width], current_reference[offset : offset + width]
            pose_transform, reference_transform = pose_matrix(pose, rotation), pose_matrix(reference_pose, rotation)
            transform = (
                reference_transform @ pose_transform if inverse else np.linalg.inv(reference_transform) @ pose_transform
            )
            parts.append(matrix_pose(transform, pose[-1], rotation))
        result = np.concatenate(parts)
        out.append(result)
        if step:
            current_reference = result if inverse else value
    return np.asarray(out, dtype=np.float32)


def joint_delta(values, ref, spec, *, inverse):
    mask = np.ones(len(ref), dtype=bool)
    mask[7::8] = False
    step = spec["reference"]["scope"] == "step"
    out = values.copy()
    if inverse:
        out[..., mask] = (np.cumsum(values[..., mask], axis=0) if step else values[..., mask]) + ref[mask]
    else:
        refs = np.concatenate([ref[None], values[:-1]], axis=0) if step else ref
        out[..., mask] -= refs[..., mask]
    return out


@dataclasses.dataclass(frozen=True)
class ActionCodec:
    """The same codec encodes labels/RTC prefixes and decodes policy outputs."""

    policy: dict
    arm: str = "both"

    @property
    def dimensions(self):
        return action_dimensions(self.policy, self.arm)

    def encode(self, actions, obs):
        kind, spec = action_branch(self.policy)
        quantity = QUANTITIES[kind].get(spec["quantity"])
        _, wire_dim = self.dimensions
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[-1] != wire_dim or not np.isfinite(values).all():
            raise ValueError(f"Actions must be finite (T, {wire_dim}) wire values")
        selection = arm_slice(self.arm, wire_dim)
        values = values[..., selection]
        if quantity == "pose_target":
            return encode_poses(values, obs, spec, self.arm)
        if spec["delta"]:
            reference = source_joint(obs, spec["reference"]["source"])[selection]
            return joint_delta(values, reference, spec, inverse=False)
        return values.copy()

    def decode(self, actions, obs):
        kind, spec = action_branch(self.policy)
        quantity = QUANTITIES[kind].get(spec["quantity"])
        encoded_dim, wire_dim = self.dimensions
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or not len(values) or values.shape[-1] < encoded_dim or not np.isfinite(values).all():
            raise ValueError(f"Model actions must be finite with at least {encoded_dim} columns")
        values = values[..., :encoded_dim].copy()
        source = source_pose if kind == "pose" else source_joint
        selection = arm_slice(self.arm, wire_dim)
        if self.arm != "both":
            measured = (
                source(obs, "state")
                if quantity in ("pose_target", "joint_position")
                else np.zeros(wire_dim, np.float32)
            )
        if spec["delta"]:
            reference = (
                measured
                if self.arm != "both" and spec["reference"]["source"] == "state"
                else source(obs, spec["reference"]["source"])
            )[selection]
        if quantity == "pose_target":
            settings = pose_settings(spec["coordinates"], obs)
            if spec["delta"]:
                reference = convert_pose(reference, settings, arm=self.arm)
                values = pose_delta(values, reference, spec, settings["rotation"], inverse=True)
            values = convert_pose(values, settings, inverse=True, arm=self.arm)
        elif spec["delta"]:
            values = joint_delta(values, reference, spec, inverse=True)
        if self.arm != "both":
            full = np.broadcast_to(measured, (len(values), wire_dim)).copy()
            full[..., selection] = values
            return full
        return values


def encode_state(policy, observation, arm="both", *, flatten=True):
    """Encode canonical histories; references are fixed at observation time.

    ``flatten=False`` preserves (history, dimensions) for temporal policies.
    """
    history_steps = policy["timing"]["history_steps"]
    states = []
    for kind, width in (("pose", 20), ("joint", 16)):
        spec = policy["state"][kind]
        if not spec["enabled"]:
            continue
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
            values = encode_poses(values, observation, spec, arm)
        elif spec["delta"]:
            values = joint_delta(
                values, source_joint(observation, spec["reference"]["source"])[selection], spec, inverse=False
            )
        states.append(values)
    result = np.concatenate(states, axis=-1)
    return result.reshape(-1) if flatten else result
