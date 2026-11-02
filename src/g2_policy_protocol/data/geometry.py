"""Rotation and pose conversions shared by datasets, encoders and model adapters.

A per-arm pose vector is ``[x, y, z, rotation..., gripper]``. ``rotation`` is
``rot6d`` (first two rotation-matrix columns), ``quaternion`` (xyzw) or
``axis_angle`` (radians). The canonical wire pose is base_link / flange / rot6d.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .._constants import POSE_ARM_DIM, SIDES

ROTATION_WIDTHS = {"rot6d": 6, "quaternion": 4, "axis_angle": 3}


def pose_width(rotation="rot6d"):
    """Per-arm pose width: xyz + rotation + gripper."""
    return 3 + ROTATION_WIDTHS[rotation] + 1


def arm_slice(arm, dim):
    """Columns of one arm (or both) in a dual-arm vector of width ``dim``."""
    half = dim // 2
    return slice(None) if arm == "both" else slice(0, half) if arm == "left" else slice(half, dim)


def rotation_to_matrix(value, representation="rot6d"):
    """Rotation vector(s) -> 3×3 matrices; rot6d is projected onto SO(3)."""
    value = np.asarray(value, dtype=np.float64)
    shape = value.shape[:-1]
    if representation == "rot6d":
        a, b = value[..., :3], value[..., 3:6]
        norm = np.linalg.norm(a, axis=-1, keepdims=True)
        x = np.where(norm > 1e-8, a / np.maximum(norm, 1e-8), np.array([1.0, 0.0, 0.0]))
        b = b - np.sum(x * b, axis=-1, keepdims=True) * x
        # A degenerate second column falls back to the axis least aligned with x.
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


def matrix_to_rotation(matrix, representation="rot6d"):
    """3×3 matrices -> rotation vector(s) in the given representation."""
    matrix = np.asarray(matrix)
    shape = matrix.shape[:-2]
    if representation == "rot6d":
        return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)
    rotation = Rotation.from_matrix(matrix.reshape(-1, 3, 3))
    result = rotation.as_quat(canonical=True) if representation == "quaternion" else rotation.as_rotvec()
    return result.reshape(*shape, result.shape[-1])


def xyz_rot6d_to_matrix(value):
    """xyz + rot6d (no gripper) -> 4×4 transform, e.g. the recorded arm base pose."""
    value = np.asarray(value)
    result = np.broadcast_to(np.eye(4), (*value.shape[:-1], 4, 4)).copy()
    result[..., :3, :3] = rotation_to_matrix(value[..., 3:9])
    result[..., :3, 3] = value[..., :3]
    return result


def pose_to_matrix(pose, representation="rot6d"):
    """Per-arm pose vector(s) -> 4×4 transforms; the gripper is dropped."""
    pose = np.asarray(pose)
    result = np.broadcast_to(np.eye(4), (*pose.shape[:-1], 4, 4)).copy()
    result[..., :3, :3] = rotation_to_matrix(pose[..., 3:-1], representation)
    result[..., :3, 3] = pose[..., :3]
    return result


def matrix_to_pose(matrix, gripper, representation="rot6d"):
    """4×4 transform(s) plus gripper -> per-arm pose vector(s)."""
    return np.concatenate(
        [matrix[..., :3, 3], matrix_to_rotation(matrix[..., :3, :3], representation), np.asarray(gripper)[..., None]],
        axis=-1,
    ).astype(np.float32)


def to_policy_frame(pose, frame, arm="both"):
    """Canonical base_link / flange / rot6d poses -> the configured parent, end effector and rotation.

    ``frame`` holds ``rotation`` and optionally ``base_to_parent`` and ``flange_to_tcp``
    ({side: 4×4}). ``base_to_parent`` is T_parent←base: it maps base_link poses into
    the parent frame, i.e. the inverse of the parent's pose in base_link.
    """
    return _transform_arms(pose, frame, arm, inverse=False)


def from_policy_frame(pose, frame, arm="both"):
    """Inverse of :func:`to_policy_frame`."""
    return _transform_arms(pose, frame, arm, inverse=True)


def _transform_arms(pose, frame, arm, *, inverse):
    pose = np.asarray(pose)
    rotation = frame["rotation"]
    base_to_parent, flange_to_tcp = frame.get("base_to_parent"), frame.get("flange_to_tcp")
    if rotation == "rot6d" and base_to_parent is None and flange_to_tcp is None:
        return pose.copy()
    width = pose_width(rotation) if inverse else POSE_ARM_DIM
    parent = np.eye(4) if base_to_parent is None else np.asarray(base_to_parent)
    if inverse:
        parent = np.linalg.inv(parent)
    pieces = []
    for index, side in enumerate(SIDES if arm == "both" else (arm,)):
        value = pose[..., index * width : (index + 1) * width]
        tcp = np.eye(4) if flange_to_tcp is None else np.asarray(flange_to_tcp[side])
        if inverse:
            transform = parent @ pose_to_matrix(value, rotation) @ np.linalg.inv(tcp)
            pieces.append(matrix_to_pose(transform, value[..., -1]))
        else:
            transform = parent @ pose_to_matrix(value) @ tcp
            pieces.append(matrix_to_pose(transform, value[..., -1], rotation))
    return np.concatenate(pieces, axis=-1)
