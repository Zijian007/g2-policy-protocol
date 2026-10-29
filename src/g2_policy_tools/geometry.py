"""G2 pose/frame/action conversions shared by policy implementations."""

import dataclasses

import numpy as np

from .schema import QUANTITIES, action_branch


def rotation_matrix(value, representation="rot6d"):
    """Project rot6d onto SO(3); quaternions use xyzw, axis angles use radians."""
    from scipy.spatial.transform import Rotation

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
    from scipy.spatial.transform import Rotation

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


def convert_pose(pose, settings, *, inverse=False):
    """Canonical base/flange rot6d ↔ configured parent/EE/rotation, per selected arm."""
    pose = np.asarray(pose)
    if not np.isfinite(pose).all():
        raise ValueError("Pose contains non-finite values")
    if (
        settings["rotation"] == "rot6d"
        and settings.get("base_to_parent") is None
        and settings.get("flange_to_tcp") is None
    ):
        return pose.copy()
    width = 10 if not inverse else 4 + {"rot6d": 6, "quaternion": 4, "axis_angle": 3}[settings["rotation"]]
    arms = pose.shape[-1] // width
    pieces = []
    for i in range(arms):
        p = pose[..., i * width : (i + 1) * width]
        parent = np.asarray(settings["base_to_parent"] if settings.get("base_to_parent") is not None else np.eye(4))
        tcp = settings.get("flange_to_tcp")
        tcp = np.eye(4) if tcp is None else np.asarray(tcp["left" if i == 0 else "right"])
        if inverse:
            t = np.linalg.inv(parent) @ pose_matrix(p, settings["rotation"]) @ np.linalg.inv(tcp)
            pieces.append(matrix_pose(t, p[..., -1]))
        else:
            t = parent @ pose_matrix(p) @ tcp
            pieces.append(matrix_pose(t, p[..., -1], settings["rotation"]))
    return np.concatenate(pieces, axis=-1)


def pose_settings(coordinates, obs):
    settings = dict(coordinates)
    if isinstance(settings.get("base_to_parent"), str) and settings["base_to_parent"] == "observation":
        if "observation/base_to_parent" not in obs:
            raise ValueError("Dynamic parent frame requires observation/base_to_parent")
        settings["base_to_parent"] = np.asarray(obs["observation/base_to_parent"]).tolist()
        from .validation import validate_rigid_transform

        validate_rigid_transform(settings["base_to_parent"], "observation/base_to_parent")
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


def measured_pose(obs):
    return source_pose(obs, "state")


def measured_joint(obs):
    return source_joint(obs, "state")


def encode_poses(values, obs, spec):
    settings = pose_settings(spec["coordinates"], obs)
    values = convert_pose(np.asarray(values, dtype=np.float32), settings)
    if not spec["delta"]:
        return values
    ref = convert_pose(source_pose(obs, spec["reference"]["source"]), settings)
    return pose_delta(values, ref, spec, settings["rotation"], inverse=False)


def decode_poses(values, obs, spec):
    settings = pose_settings(spec["coordinates"], obs)
    if spec["delta"]:
        ref = convert_pose(source_pose(obs, spec["reference"]["source"]), settings)
        values = pose_delta(values, ref, spec, settings["rotation"], inverse=True)
    return convert_pose(values, settings, inverse=True)


def pose_delta(values, ref, spec, rotation, *, inverse):
    width = len(ref) // 2
    step = spec["reference"]["scope"] == "step"
    if spec["method"] == "component":
        mask = np.tile([True] * (width - 1) + [False], 2)
        out = values.copy()
        if inverse:
            out[..., mask] = (np.cumsum(values[..., mask], axis=0) if step else values[..., mask]) + ref[mask]
        else:
            refs = np.concatenate([ref[None], values[:-1]], axis=0) if step else ref
            out[..., mask] -= refs[..., mask]
        return out
    out = []
    current = ref.copy()
    for value in values:
        parts = []
        for offset in (0, width):
            p, r = value[offset : offset + width], current[offset : offset + width]
            tp, tr = pose_matrix(p, rotation), pose_matrix(r, rotation)
            t = tr @ tp if inverse else np.linalg.inv(tr) @ tp
            parts.append(matrix_pose(t, p[-1], rotation))
        result = np.concatenate(parts)
        out.append(result)
        if step:
            current = result if inverse else value
    return np.asarray(out, dtype=np.float32)


def joint_delta(values, ref, spec, *, inverse):
    mask = np.tile([True] * 7 + [False], 2)
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

    def _selection(self, dim):
        half = dim // 2
        return slice(None) if self.arm == "both" else slice(0, half) if self.arm == "left" else slice(half, dim)

    def encode(self, actions, obs):
        kind, spec = action_branch(self.policy)
        quantity = QUANTITIES[kind].get(spec["quantity"])
        values = np.asarray(actions, dtype=np.float32).copy()
        if values.ndim != 2 or values.shape[-1] != self.dimensions[1] or not np.isfinite(values).all():
            raise ValueError(f"Actions must be finite (T, {self.dimensions[1]}) wire values")
        if quantity == "pose_target":
            values = encode_poses(values, obs, spec)
        elif spec["delta"]:
            values = joint_delta(values, source_joint(obs, spec["reference"]["source"]), spec, inverse=False)
        return values[..., self._selection(values.shape[-1])].copy()

    def decode(self, actions, obs):
        kind, spec = action_branch(self.policy)
        quantity = QUANTITIES[kind].get(spec["quantity"])
        encoded_dim, wire_dim = self.dimensions
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or not len(values) or values.shape[-1] < encoded_dim or not np.isfinite(values).all():
            raise ValueError(f"Model actions must be finite with at least {encoded_dim} columns")
        values = values[..., :encoded_dim].copy()
        if self.arm != "both":
            if quantity == "pose_target":
                full = convert_pose(measured_pose(obs), pose_settings(spec["coordinates"], obs))
            elif quantity == "joint_position":
                full = measured_joint(obs)
            else:
                full = np.zeros(wire_dim, dtype=np.float32)
            values_full = np.broadcast_to(full, (len(values), len(full))).copy()
            values_full[..., self._selection(len(full))] = values
            values = values_full
        if quantity == "pose_target":
            values = decode_poses(values, obs, spec)
            if self.arm != "both":
                other = slice(10, 20) if self.arm == "left" else slice(0, 10)
                values[..., other] = measured_pose(obs)[other]
        elif spec["delta"]:
            values = joint_delta(values, source_joint(obs, spec["reference"]["source"]), spec, inverse=True)
            if self.arm != "both":
                other = slice(8, 16) if self.arm == "left" else slice(0, 8)
                values[..., other] = measured_joint(obs)[other]
        return values
