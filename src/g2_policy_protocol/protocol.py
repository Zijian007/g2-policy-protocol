"""G2 policy v3 messages and negotiated observation/action capabilities.

Decode with codec before parsing. Validate the handshake once, then use its
metadata for request, action, RTC and reset processing.
"""

from collections.abc import Mapping

import numpy as np

from . import _generated as fields
from . import _validation as check
from ._validation import finite_action_rows, finite_vector
from .codec import validate_image
from .validation import validate_rigid_transform

PROTOCOL = fields.PROTOCOL
CAMERA_IDS = fields.CAMERA_IDS
SIDES = fields.SIDES
POSE_DIM = fields.POSE_DIM
POSE_FIELDS = fields.POSE_FIELDS
JOINT_DIM = fields.JOINT_DIM
JOINT_FIELDS = fields.JOINT_FIELDS
ENCODING_VALUES = fields.ENCODING_VALUES
ACTION_TYPE = fields.ACTION_TYPE
ERROR_TYPE = fields.ERROR_TYPE


def cameras_for_arm(arm):
    if arm == "both":
        return CAMERA_IDS
    if arm in SIDES:
        return (CAMERA_IDS[0], CAMERA_IDS[SIDES.index(arm) + 1])
    raise ValueError(f"unsupported arm: {arm!r}")


def build_action_spec(control_quantity="pose_target"):
    """Canonical dual-arm wire layouts, independent of model's latent action space."""
    if control_quantity == "pose_target":
        names = ["x", "y", "z"] + [f"rot6d_{i}" for i in range(6)] + ["gripper"]
        units = ["m"] * 3 + ["dimensionless"] * 6 + ["close_ratio"]
    elif control_quantity in ("joint_position", "joint_velocity", "joint_torque"):
        names = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]
        units = [
            {"joint_position": "rad", "joint_velocity": "rad/s", "joint_torque": "N*m"}[
                control_quantity
            ]
        ] * 7 + ["close_ratio"]
    else:
        raise ValueError(f"unsupported control_quantity {control_quantity!r}")
    names = [f"{side}_{name}" for side in ("left", "right") for name in names]
    return {
        "control_quantity": control_quantity,
        "dim": len(names),
        "fields": names,
        "units": units * 2,
        "frame": fields.FRAME,
        "ee_frame": fields.EE_FRAME,
        "representation": fields.ACTION_SEMANTICS,
    }


def parse_actions(value, metadata):
    """Finite nonempty (T, D) actions bounded by the negotiated horizon."""
    actions = finite_action_rows(value, metadata["action_spec"]["dim"])
    if len(actions) > metadata["action_horizon"]:
        raise ValueError("actions exceeds action_horizon")
    return actions


def parse_rtc(payload, metadata):
    enabled = payload.get("rtc_enabled", False)
    delay = int(payload.get("inference_delay", 0))
    raw_prefix = payload.get("prev_chunk_left_over")
    if enabled and not metadata["rtc_supported"]:
        raise ValueError("server does not support RTC")
    if not enabled and (delay or raw_prefix is not None):
        raise ValueError("RTC fields require rtc_enabled=true")
    prefix = None if raw_prefix is None else parse_actions(raw_prefix, metadata)
    if delay >= metadata["action_horizon"]:
        raise ValueError("inference_delay exceeds action horizon")
    if prefix is not None and delay > len(prefix):
        raise ValueError("inference_delay exceeds RTC prefix")
    return enabled, delay, prefix


def build_metadata(
    *,
    policy_hz,
    action_horizon=50,
    arm="both",
    image_encoding="raw",
    image_resize="resize",
    image_sizes=None,
    required_cameras=None,
    rtc_supported=False,
    requires_joint_state=False,
    extensions=None,
    action_spec=None,
    observation_spec=None,
    timing=None,
    reset_supported=False,
):
    meta = dict(
        type=fields.METADATA_TYPE,
        protocol=PROTOCOL,
        policy_hz=policy_hz,
        action_horizon=action_horizon,
        arm=arm,
        image_encoding=image_encoding,
        image_resize=image_resize,
        image_sizes={} if image_sizes is None else dict(image_sizes),
        required_cameras=list(
            cameras_for_arm(arm) if required_cameras is None else required_cameras
        ),
        rtc_supported=rtc_supported,
        requires_joint_state=requires_joint_state,
        action_spec=build_action_spec() if action_spec is None else dict(action_spec),
        observation_spec=dict(
            pose=True,
            joint=requires_joint_state,
            prompt=True,
            history_steps=1,
            history_hz=policy_hz,
            context_keys=[],
        )
        if observation_spec is None
        else dict(observation_spec),
        timing=dict(action_dt=1 / policy_hz, first_target_offset_s=0)
        if timing is None
        else dict(timing),
        reset_supported=reset_supported,
    )
    if extensions is not None:
        meta["extensions"] = dict(extensions)
    validate_metadata(meta)
    meta["policy_hz"] = float(meta["policy_hz"])
    meta["action_horizon"] = int(meta["action_horizon"])
    meta["required_cameras"] = [str(key) for key in meta["required_cameras"]]
    meta["image_sizes"] = {
        str(key): [int(value) for value in size]
        for key, size in meta["image_sizes"].items()
    }
    meta["observation_spec"]["history_steps"] = int(
        meta["observation_spec"]["history_steps"]
    )
    meta["observation_spec"]["history_hz"] = float(
        meta["observation_spec"]["history_hz"]
    )
    meta["timing"] = {key: float(value) for key, value in meta["timing"].items()}
    return meta


def validate_metadata(meta, *, policy_hz=None, require_rtc=False):
    fields.validate_metadata_fields(meta)
    if meta["action_spec"] != build_action_spec(
        meta["action_spec"]["control_quantity"]
    ):
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
            raise ValueError(
                "history observation_spec must require all state/image histories"
            )
    if policy_hz is not None:
        fields.validate_policy_hz(policy_hz)
        if not np.isclose(meta["policy_hz"], policy_hz):
            raise ValueError("server policy_hz disagrees with configured policy_hz")
    if require_rtc and not meta["rtc_supported"]:
        raise ValueError("server does not support RTC")


def parse_context(payload, metadata):
    spec = metadata["observation_spec"]
    value = payload.get("context", {})
    allowed = (
        "command_pose",
        "command_joint",
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
                if (
                    sequence.dtype != np.uint8
                    or sequence.ndim != 4
                    or len(sequence) != spec["history_steps"]
                ):
                    raise ValueError(
                        "image_history must contain raw uint8 (L,H,W,3) frames"
                    )
                if sequence.shape[-1] != 3 or (
                    hw and tuple(sequence.shape[1:3]) != tuple(hw)
                ):
                    raise ValueError("image_history size disagrees with metadata")
                frames[camera] = sequence
            result[name] = frames
        elif name == "base_to_parent":
            arr = np.asarray(item, dtype=np.float32)
            validate_rigid_transform(arr.tolist(), "context.base_to_parent")
            result[name] = arr
        elif name.endswith("_history"):
            width = JOINT_DIM if name == "joint_history" else POSE_DIM
            arr = np.asarray(item, dtype=np.float32)
            if (
                arr.shape != (spec["history_steps"], width)
                or not np.isfinite(arr).all()
            ):
                raise ValueError(f"{name} must be finite (L, {width})")
            result[name] = arr
        else:
            result[name] = finite_vector(
                item,
                JOINT_DIM if name in ("object_joint", "start_joint", "command_joint") else POSE_DIM,
                name,
            )
    return result


def build_request(
    *,
    request_id,
    prompt,
    state,
    images,
    metadata,
    joint_state=None,
    rtc_enabled=False,
    inference_delay=0,
    prev_chunk_left_over=None,
    observation_time_ns=None,
    context=None,
):
    payload = dict(
        type=fields.OBS_TYPE,
        request_id=request_id,
        prompt=prompt,
        state=state,
        images=images,
        rtc_enabled=rtc_enabled,
        inference_delay=inference_delay,
        prev_chunk_left_over=prev_chunk_left_over,
    )
    if state is None:
        payload.pop("state")
    if context is not None:
        payload["context"] = context
    if joint_state is not None:
        payload["joint_state"] = joint_state
    if observation_time_ns is not None:
        payload["observation_time_ns"] = observation_time_ns
    return parse_request(payload, metadata=metadata)


def parse_request(payload, *, metadata):
    fields.validate_obs_fields(payload)
    spec = metadata["observation_spec"]
    result = dict(payload)
    prompt = payload.get("prompt", "")
    if spec["prompt"] and not prompt.strip():
        raise ValueError("server requires a nonempty prompt")
    result["prompt"] = prompt
    for key, required, width in [
        ("state", spec["pose"], POSE_DIM),
        ("joint_state", spec["joint"], JOINT_DIM),
    ]:
        if required or key in payload:
            result[key] = finite_vector(payload.get(key), width, key)
    result["images"] = validate_images(payload["images"], metadata=metadata)
    result["context"] = parse_context(payload, metadata)
    if spec["history_steps"] > 1 and "observation_time_ns" not in payload:
        raise ValueError("history requires observation_time_ns for the latest sample")
    rtc_enabled, inference_delay, prefix = parse_rtc(payload, metadata)
    result.update(
        rtc_enabled=rtc_enabled,
        inference_delay=inference_delay,
        prev_chunk_left_over=prefix,
    )
    return result


def validate_images(images, *, metadata, decoded=False):
    """Check negotiated camera requirements before/after adapter JPEG decoding."""
    fields.validate_images_fields(images)
    if any(images.get(key) is None for key in metadata["required_cameras"]):
        raise ValueError("missing required cameras")
    result = {}
    for key, image in images.items():
        if image is None:
            continue
        image = validate_image(
            image, encoding="raw" if decoded else metadata["image_encoding"]
        )
        expected = metadata["image_sizes"].get(key)
        if (
            isinstance(image, np.ndarray)
            and expected is not None
            and tuple(image.shape[:2]) != tuple(expected)
        ):
            raise ValueError(f"{key} size disagrees with metadata image_sizes")
        result[key] = image
    return result


def build_response(
    *,
    request_id,
    actions,
    server_timing=None,
    policy_timing=None,
    rtc_info=None,
    metadata,
):
    result = dict(type=ACTION_TYPE, request_id=request_id, actions=actions)
    for key, block in (
        ("server_timing", server_timing),
        ("policy_timing", policy_timing),
        ("rtc_info", rtc_info),
    ):
        if block is not None:
            result[key] = dict(block)
    result["actions"] = parse_response(result, request_id=request_id, metadata=metadata)
    result["request_id"] = int(request_id)
    for key in ("server_timing", "policy_timing"):
        if key in result:
            result[key] = {name: float(value) for name, value in result[key].items()}
    return result


def build_error(*, request_id, message, code="inference_error"):
    result = dict(type=ERROR_TYPE, request_id=request_id, code=code, message=message)
    fields.validate_error_fields(result)
    result["request_id"] = int(request_id)
    return result


def parse_response(payload, *, request_id, metadata):
    """Validate an action/error against its request ID and negotiated metadata."""
    if not isinstance(payload, Mapping):
        raise ValueError("response must be an object")
    kind = payload.get("type")
    if kind == ACTION_TYPE:
        fields.validate_action_fields(payload)
    elif kind == ERROR_TYPE:
        fields.validate_error_fields(payload)
    else:
        raise ValueError("expected action or error response")
    fields.validate_request_id(request_id)
    if payload["request_id"] != request_id:
        raise ValueError("response request_id does not match observation")
    if kind == ERROR_TYPE:
        raise RuntimeError(f"policy server {payload['code']}: {payload['message']}")
    actions = parse_actions(payload["actions"], metadata)
    if metadata.get("extensions", {}).get("response_mode") == "full_chunk":
        if len(actions) != metadata["action_horizon"]:
            raise ValueError("full_chunk response must contain action_horizon actions")
    return actions


def build_reset(*, request_id, metadata):
    if not metadata["reset_supported"]:
        raise ValueError("server does not support episode reset")
    result = dict(type=fields.RESET_TYPE, request_id=request_id)
    fields.validate_reset_fields(result)
    return result


def build_reset_ack(*, request_id):
    result = dict(type=fields.RESET_ACK_TYPE, request_id=request_id)
    fields.validate_reset_ack_fields(result)
    return result


def parse_reset_ack(payload, *, request_id):
    fields.validate_reset_ack_fields(payload)
    if payload["request_id"] != request_id:
        raise ValueError("reset acknowledgement request_id mismatch")
    return dict(payload)
