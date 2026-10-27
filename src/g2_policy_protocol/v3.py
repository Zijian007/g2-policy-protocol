"""Model-neutral v3 capabilities and dynamic arrays; v2 stays backward compatible."""

import numpy as np

from . import _generated_v3 as fields
from . import _validation as check

PROTOCOL = "g2.policy.v3"


def action_spec(control_quantity="pose_target"):
    """Canonical dual-arm wire layouts, independent of model's latent action space."""
    if control_quantity == "pose_target":
        names = ["x", "y", "z"] + [f"rot6d_{i}" for i in range(6)] + ["gripper"]
        units = ["m"] * 3 + ["dimensionless"] * 6 + ["close_ratio"]
    elif control_quantity == "eef_twist":
        names = ["vx", "vy", "vz", "wx", "wy", "wz", "gripper"]
        units = ["m/s"] * 3 + ["rad/s"] * 3 + ["close_ratio"]
    elif control_quantity in ("joint_position", "joint_velocity", "joint_torque"):
        names = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]
        units = [{"joint_position": "rad", "joint_velocity": "rad/s", "joint_torque": "N*m"}[control_quantity]] * 7 + [
            "close_ratio"
        ]
    else:
        raise ValueError(f"unsupported control_quantity {control_quantity!r}")
    names = [f"{side}_{name}" for side in ("left", "right") for name in names]
    return {
        "control_quantity": control_quantity,
        "dim": len(names),
        "fields": names,
        "units": units * 2,
        "frame": "base_link",
        "ee_frame": "flange",
        "representation": "absolute",
    }


def validate_metadata(meta):
    fields.validate_metadata_fields(meta)
    if meta["action_spec"] != action_spec(meta["action_spec"]["control_quantity"]):
        raise ValueError("action_spec must use the canonical layout, units and frames")
    if meta["requires_joint_state"] != meta["observation_spec"]["joint"]:
        raise ValueError("requires_joint_state disagrees with observation_spec.joint")
    if not np.isclose(meta["timing"]["action_dt"], 1 / meta["policy_hz"]):
        raise ValueError("action_dt disagrees with policy_hz")
    obs = meta["observation_spec"]
    if obs["history_steps"] > 1:
        needed = {"image_history"} if meta["required_cameras"] else set()
        needed |= {"pose_history"} if obs["pose"] else set()
        needed |= {"joint_history"} if obs["joint"] else set()
        if not needed <= set(obs["context_keys"]):
            raise ValueError("history observation_spec must require all state/image histories")


def actions(value, metadata):
    dim = metadata["action_spec"]["dim"]
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or not len(arr) or arr.shape[1] != dim or not np.isfinite(arr).all():
        raise ValueError(f"actions must be finite nonempty (T, {dim}), got {arr.shape}")
    if len(arr) > metadata["action_horizon"]:
        raise ValueError("actions exceeds action_horizon")
    return np.ascontiguousarray(arr)


def context(payload, metadata):
    from .protocol import _vector

    spec = metadata["observation_spec"]
    value = payload.get("context", {})
    allowed = (
        "command_pose",
        "previous_command_pose",
        "object_pose",
        "object_joint",
        "start_pose",
        "start_joint",
        "base_to_parent",
        "pose_history",
        "joint_history",
        "image_history",
    )
    check.object_fields(value, "context", fields=allowed, required=spec["context_keys"])
    result = {}
    for name, item in value.items():
        if name == "image_history":
            check.object_fields(
                item,
                name,
                fields=metadata["required_cameras"],
                required=metadata["required_cameras"],
            )
            frames = {}
            for camera, frames_value in item.items():
                sequence = np.asarray(frames_value)
                hw = metadata["image_sizes"].get(camera)
                if sequence.dtype != np.uint8 or sequence.ndim != 4 or len(sequence) != spec["history_steps"]:
                    raise ValueError("image_history must contain raw uint8 (L,H,W,3) frames")
                if sequence.shape[-1] != 3 or (hw and tuple(sequence.shape[1:3]) != tuple(hw)):
                    raise ValueError("image_history size disagrees with metadata")
                frames[camera] = sequence
            result[name] = frames
        elif name == "base_to_parent":
            from g2_policy_tools.validation import validate_rigid_transform

            arr = np.asarray(item, dtype=np.float32)
            validate_rigid_transform(arr.tolist(), "context.base_to_parent")
            result[name] = arr
        elif name.endswith("_history"):
            width = 16 if name == "joint_history" else 20
            arr = np.asarray(item, dtype=np.float32)
            if arr.shape != (spec["history_steps"], width) or not np.isfinite(arr).all():
                raise ValueError(f"{name} must be finite (L, {width})")
            result[name] = arr
        else:
            result[name] = _vector(item, 16 if name in ("object_joint", "start_joint") else 20, name)
    return result


def parse_request(payload, metadata):
    from .protocol import _vector
    from .protocol import validate_images

    fields.validate_obs_fields(payload)
    spec = metadata["observation_spec"]
    result = dict(payload)
    prompt = payload.get("prompt", "")
    if spec["prompt"] and not prompt.strip():
        raise ValueError("server requires a nonempty prompt")
    result["prompt"] = prompt
    for key, required, width in [
        ("state", spec["pose"], 20),
        ("joint_state", spec["joint"], 16),
    ]:
        if required or key in payload:
            result[key] = _vector(payload.get(key), width, key)
    result["images"] = validate_images(payload["images"], metadata=metadata)
    result["context"] = context(payload, metadata)
    if spec["history_steps"] > 1 and "observation_time_ns" not in payload:
        raise ValueError("history requires observation_time_ns for the latest sample")
    rtc = payload.get("rtc_enabled", False)
    delay = payload.get("inference_delay", 0)
    prefix = payload.get("prev_chunk_left_over")
    if rtc and not metadata["rtc_supported"]:
        raise ValueError("server does not support RTC")
    if not rtc and (delay or prefix is not None):
        raise ValueError("RTC fields require rtc_enabled=true")
    if delay >= metadata["action_horizon"]:
        raise ValueError("inference_delay exceeds action horizon")
    prefix = None if prefix is None else actions(prefix, metadata)
    if prefix is not None and delay > len(prefix):
        raise ValueError("inference_delay exceeds RTC prefix")
    result.update(rtc_enabled=rtc, inference_delay=delay, prev_chunk_left_over=prefix)
    return result


def reset_message(request_id, *, ack=False):
    result = {"type": "reset_ack" if ack else "reset", "request_id": request_id}
    (fields.validate_reset_ack_fields if ack else fields.validate_reset_fields)(result)
    return result
