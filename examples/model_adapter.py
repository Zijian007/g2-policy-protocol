"""Explicit conversion examples; select conventions that match your trained model."""

import numpy as np

from g2_policy_protocol import protocol as p


def quaternion_wxyz_to_rot6d(quaternion):
    """Unit quaternion, active rotation, wxyz order; returns first two columns."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("expected a finite nonzero wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    return rotation[:, :2].reshape(-1, order="F").astype(np.float32)


def expand_single_arm(actions10, measured_pose20, *, arm):
    """Fill the unused arm with its measured pose, including gripper."""
    actions = np.asarray(actions10, dtype=np.float32)
    measured = np.asarray(measured_pose20, dtype=np.float32)
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    if actions.ndim != 2 or actions.shape[1] != 10 or len(actions) == 0:
        raise ValueError("single-arm actions must have nonempty shape (T, 10)")
    if measured.shape != (20,) or not np.isfinite(measured).all() or not np.isfinite(actions).all():
        raise ValueError("expected finite actions and measured pose20")
    expanded = np.repeat(measured[None, :], len(actions), axis=0)
    offset = 0 if arm == "left" else 10
    expanded[:, offset:offset + 10] = actions
    return expanded


def additive_pose_delta_to_absolute(deltas20, measured_pose20):
    """For component-wise xyz/rot6d deltas relative to the SAME measured pose.

    Grippers are already absolute. NOT for axis-angle, quaternion, local-frame
    translations or incremental deltas relative to each preceding action.
    """
    delta = np.asarray(deltas20, dtype=np.float32)
    measured = np.asarray(measured_pose20, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[1] != 20 or len(delta) == 0:
        raise ValueError("expected nonempty (T, 20) deltas")
    if measured.shape != (20,) or not np.isfinite(measured).all() or not np.isfinite(delta).all():
        raise ValueError("expected finite deltas and measured pose20")
    result = delta.copy()
    for start in (0, 10):
        result[:, start:start + 9] += measured[start:start + 9]
    return result


if __name__ == "__main__":
    measured = np.zeros(20, dtype=np.float32)
    measured[3:9] = measured[13:19] = quaternion_wxyz_to_rot6d([1, 0, 0, 0])
    delta = np.zeros((2, 20), dtype=np.float32)
    delta[:, 10] = [0.01, 0.02]  # Both relative to the original measured right x.
    delta[:, 19] = 0.5  # Absolute gripper command.
    absolute = additive_pose_delta_to_absolute(delta, measured)
    expanded = expand_single_arm(absolute[:, 10:20], measured, arm="right")
    response = p.build_response(request_id=1, actions=expanded)
    print("Converted right-arm actions:", response["actions"].shape)
