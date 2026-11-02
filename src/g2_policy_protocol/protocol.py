"""Build and parse g2.policy.v3 messages against the negotiated handshake metadata.

Decode frames with codec first. Validate the handshake once, then pass the same
metadata to every build_* / parse_* call. Invalid messages raise ValueError.
"""

from collections.abc import Mapping

import numpy as np

from . import _messages as messages
from . import validation as check
from ._constants import (
    ACTION_SEMANTICS,
    ACTION_TYPE,
    CAMERA_IDS,
    CONTEXT_KEYS,
    EE_FRAME,
    ENCODING_VALUES,
    ERROR_TYPE,
    FRAME,
    JOINT_DIM,
    JOINT_FIELDS,
    JOINT_SIDE_UNITS,
    METADATA_TYPE,
    OBS_TYPE,
    POSE_DIM,
    POSE_FIELDS,
    POSE_SIDE_UNITS,
    PROTOCOL,
    RESET_ACK_TYPE,
    RESET_TYPE,
    SIDES,
)
from .codec import validate_image
from .validation import finite_action_rows, finite_vector, validate_rigid_transform

__all__ = [
    "ACTION_TYPE", "CAMERA_IDS", "ENCODING_VALUES", "ERROR_TYPE", "JOINT_DIM", "JOINT_FIELDS",
    "POSE_DIM", "POSE_FIELDS", "PROTOCOL", "SIDES",
    "build_action_spec", "build_error", "build_metadata", "build_request", "build_reset",
    "build_reset_ack", "build_response", "cameras_for_arm", "parse_actions", "parse_context",
    "parse_request", "parse_reset_ack", "parse_response", "parse_rtc", "validate_images",
    "validate_metadata",
]

# Context entries by kind; together with base_to_parent and image_history they cover CONTEXT_KEYS.
_CONTEXT_VECTORS = {
    "command_pose": POSE_DIM,
    "previous_command_pose": POSE_DIM,
    "start_pose": POSE_DIM,
    "object_pose": POSE_DIM,
    "command_joint": JOINT_DIM,
    "start_joint": JOINT_DIM,
    "object_joint": JOINT_DIM,
}
_CONTEXT_HISTORIES = {"pose_history": POSE_DIM, "joint_history": JOINT_DIM}


# --- Handshake ---------------------------------------------------------------


def cameras_for_arm(arm):
    """Default cameras: all three for both arms, otherwise head plus the same-side wrist."""
    if arm == "both":
        return CAMERA_IDS
    if arm in SIDES:
        return (CAMERA_IDS[0], CAMERA_IDS[SIDES.index(arm) + 1])
    raise ValueError(f"unsupported arm: {arm!r}")


def build_action_spec(control_quantity="pose_target"):
    """Canonical dual-arm wire layout, independent of the model's internal action space."""
    if control_quantity == "pose_target":
        names, units = POSE_FIELDS, POSE_SIDE_UNITS
    elif control_quantity == "joint_position":
        names, units = JOINT_FIELDS, JOINT_SIDE_UNITS
    else:
        raise ValueError(f"unsupported control_quantity {control_quantity!r}")
    return {
        "control_quantity": control_quantity,
        "dim": len(names),
        "fields": list(names),
        "units": list(units) * len(SIDES),
        "frame": FRAME,
        "ee_frame": EE_FRAME,
        "representation": ACTION_SEMANTICS,
    }


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
    """Server capabilities; omitted fields get values consistent with the given ones."""
    if required_cameras is None:
        required_cameras = cameras_for_arm(arm)
    if action_spec is None:
        action_spec = build_action_spec()
    if observation_spec is None:
        observation_spec = dict(pose=True, joint=requires_joint_state, prompt=True,
                                history_steps=1, history_hz=policy_hz, context_keys=[])
    if timing is None:
        timing = dict(action_dt=1 / policy_hz, first_target_offset_s=0)
    metadata = dict(
        type=METADATA_TYPE,
        protocol=PROTOCOL,
        policy_hz=policy_hz,
        action_horizon=action_horizon,
        arm=arm,
        image_encoding=image_encoding,
        image_resize=image_resize,
        image_sizes=dict(image_sizes or {}),
        required_cameras=list(required_cameras),
        rtc_supported=rtc_supported,
        requires_joint_state=requires_joint_state,
        action_spec=dict(action_spec),
        observation_spec=dict(observation_spec),
        timing=dict(timing),
        reset_supported=reset_supported,
    )
    if extensions is not None:
        metadata["extensions"] = dict(extensions)
    validate_metadata(metadata)
    return _plain_numbers(metadata)


def _plain_numbers(metadata):
    """Replace caller NumPy scalars with Python numbers so JSON stays readable."""
    observation = metadata["observation_spec"]
    metadata["policy_hz"] = float(metadata["policy_hz"])
    metadata["action_horizon"] = int(metadata["action_horizon"])
    metadata["required_cameras"] = [str(camera) for camera in metadata["required_cameras"]]
    metadata["image_sizes"] = {
        str(camera): [int(value) for value in size] for camera, size in metadata["image_sizes"].items()
    }
    observation["history_steps"] = int(observation["history_steps"])
    observation["history_hz"] = float(observation["history_hz"])
    metadata["timing"] = {key: float(value) for key, value in metadata["timing"].items()}
    return metadata


def validate_metadata(metadata, *, policy_hz=None, require_rtc=False):
    """Check the handshake; optionally compare it with the client's own expectations."""
    messages.check_metadata(metadata)
    action_spec = metadata["action_spec"]
    if action_spec != build_action_spec(action_spec["control_quantity"]):
        raise ValueError(f"action_spec must use the canonical layout, units and frames{messages.VERSION_HINT}")
    observation = metadata["observation_spec"]
    if metadata["requires_joint_state"] != observation["joint"]:
        raise ValueError("requires_joint_state disagrees with observation_spec.joint")
    if not np.isclose(metadata["timing"]["action_dt"], 1 / metadata["policy_hz"]):
        raise ValueError("action_dt disagrees with policy_hz")
    if observation["history_steps"] > 1:
        needed = {"image_history"} if metadata["required_cameras"] else set()
        needed |= {"pose_history"} if observation["pose"] else set()
        needed |= {"joint_history"} if observation["joint"] else set()
        if not needed <= set(observation["context_keys"]):
            raise ValueError("history observation_spec must require all state/image histories")
    if policy_hz is not None:
        check.number(policy_hz, "policy_hz", exclusive_minimum=0)
        if not np.isclose(metadata["policy_hz"], policy_hz):
            raise ValueError("server policy_hz disagrees with configured policy_hz")
    if require_rtc and not metadata["rtc_supported"]:
        raise ValueError("server does not support RTC")


# --- Observation -------------------------------------------------------------


def build_request(
    *,
    request_id,
    prompt,
    state,
    images=None,
    metadata,
    joint_state=None,
    rtc_enabled=False,
    inference_delay=0,
    prev_chunk_left_over=None,
    observation_time_ns=None,
    context=None,
):
    """Client side: build and validate an obs message. Optional arguments left as None are omitted."""
    payload = dict(
        type=OBS_TYPE,
        request_id=request_id,
        prompt=prompt,
        rtc_enabled=rtc_enabled,
        inference_delay=inference_delay,
        prev_chunk_left_over=prev_chunk_left_over,
    )
    optional = dict(state=state, joint_state=joint_state, images=images,
                    context=context, observation_time_ns=observation_time_ns)
    payload.update({key: value for key, value in optional.items() if value is not None})
    return parse_request(payload, metadata=metadata)


def parse_request(payload, *, metadata):
    """Server side: validate an obs message and normalize its arrays."""
    messages.check_obs(payload)
    observation = metadata["observation_spec"]
    result = dict(payload)
    prompt = payload.get("prompt", "")
    if observation["prompt"] and not prompt.strip():
        raise ValueError("server requires a nonempty prompt")
    result["prompt"] = prompt
    for key, required, width in (("state", observation["pose"], POSE_DIM),
                                 ("joint_state", observation["joint"], JOINT_DIM)):
        if required or key in payload:
            result[key] = finite_vector(payload.get(key), width, key)
    result["context"] = parse_context(payload, metadata)
    if "images" in payload:
        result["images"] = validate_images(payload["images"], metadata=metadata)
    elif "image_history" not in result["context"]:
        raise ValueError(f"obs requires images or context.image_history{messages.VERSION_HINT}")
    if observation["history_steps"] > 1 and "observation_time_ns" not in payload:
        raise ValueError("history requires observation_time_ns for the latest sample")
    rtc_enabled, inference_delay, prefix = parse_rtc(payload, metadata)
    result.update(rtc_enabled=rtc_enabled, inference_delay=inference_delay, prev_chunk_left_over=prefix)
    return result


def validate_images(images, *, metadata, decoded=False):
    """Check required cameras and sizes, before or (decoded=True) after JPEG decoding."""
    check.object_fields(images, "images", fields=CAMERA_IDS)
    if any(images.get(camera) is None for camera in metadata["required_cameras"]):
        raise ValueError("missing required cameras")
    encoding = "raw" if decoded else metadata["image_encoding"]
    result = {}
    for camera, image in images.items():
        if image is None:
            continue
        image = validate_image(image, encoding=encoding)
        expected = metadata["image_sizes"].get(camera)
        if isinstance(image, np.ndarray) and expected is not None and tuple(image.shape[:2]) != tuple(expected):
            raise ValueError(f"{camera} size disagrees with metadata image_sizes")
        result[camera] = image
    return result


def parse_context(payload, metadata):
    """Validate the context keys declared by observation_spec.context_keys."""
    observation = metadata["observation_spec"]
    context = payload.get("context", {})
    check.object_fields(context, "context", fields=CONTEXT_KEYS, required=observation["context_keys"])
    steps = observation["history_steps"]
    result = {}
    for name, value in context.items():
        if name in _CONTEXT_VECTORS:
            result[name] = finite_vector(value, _CONTEXT_VECTORS[name], name)
        elif name in _CONTEXT_HISTORIES:
            result[name] = _parse_history(value, steps, _CONTEXT_HISTORIES[name], name)
        elif name == "base_to_parent":
            matrix = np.asarray(value, dtype=np.float32)
            validate_rigid_transform(matrix.tolist(), "context.base_to_parent")
            result[name] = matrix
        else:
            result[name] = _parse_image_history(value, steps, metadata)
    return result


def _parse_history(value, steps, width, name):
    history = np.asarray(value, dtype=np.float32)
    if history.shape != (steps, width) or not np.isfinite(history).all():
        raise ValueError(f"{name} must be finite (L, {width})")
    return history


def _parse_image_history(value, steps, metadata):
    cameras = metadata["required_cameras"]
    check.object_fields(value, "image_history", fields=cameras, required=cameras)
    result = {}
    for camera, frames in value.items():
        if metadata["image_encoding"] == "jpeg":
            if (not isinstance(frames, (list, tuple)) or len(frames) != steps
                    or any(not isinstance(frame, bytes) or not frame for frame in frames)):
                raise ValueError(
                    f"image_history.{camera} must be {steps} nonempty JPEG bytes{messages.VERSION_HINT}"
                )
            result[camera] = list(frames)
            continue
        frames = np.asarray(frames)
        if frames.dtype != np.uint8 or frames.ndim != 4 or len(frames) != steps:
            raise ValueError(f"image_history must contain raw uint8 (L,H,W,3) frames{messages.VERSION_HINT}")
        size = metadata["image_sizes"].get(camera)
        if frames.shape[-1] != 3 or (size and tuple(frames.shape[1:3]) != tuple(size)):
            raise ValueError("image_history size disagrees with metadata")
        result[camera] = frames
    return result


def parse_rtc(payload, metadata):
    """Return (enabled, inference_delay, prefix); RTC fields require rtc_enabled."""
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


# --- Action / error ----------------------------------------------------------


def parse_actions(value, metadata):
    """Finite nonempty (T, D) actions, T bounded by the negotiated horizon."""
    actions = finite_action_rows(value, metadata["action_spec"]["dim"])
    if len(actions) > metadata["action_horizon"]:
        raise ValueError("actions exceeds action_horizon")
    return actions


def build_response(*, request_id, actions, metadata, server_timing=None, policy_timing=None, rtc_info=None):
    """Server side: build and validate an action message."""
    result = dict(type=ACTION_TYPE, request_id=request_id, actions=actions)
    for key, block in (("server_timing", server_timing), ("policy_timing", policy_timing), ("rtc_info", rtc_info)):
        if block is not None:
            result[key] = dict(block)
    result["actions"] = parse_response(result, request_id=request_id, metadata=metadata)
    result["request_id"] = int(request_id)
    for key in ("server_timing", "policy_timing"):
        if key in result:
            result[key] = {name: float(value) for name, value in result[key].items()}
    return result


def parse_response(payload, *, request_id, metadata):
    """Client side: return the actions, or raise RuntimeError for an error message."""
    if not isinstance(payload, Mapping):
        raise ValueError("response must be an object")
    kind = payload.get("type")
    if kind == ACTION_TYPE:
        messages.check_action(payload)
    elif kind == ERROR_TYPE:
        messages.check_error(payload)
    else:
        raise ValueError("expected action or error response")
    check.integer(request_id, "request_id", minimum=0)
    if payload["request_id"] != request_id:
        raise ValueError("response request_id does not match observation")
    if kind == ERROR_TYPE:
        raise RuntimeError(f"policy server {payload['code']}: {payload['message']}")
    actions = parse_actions(payload["actions"], metadata)
    full_chunk = metadata.get("extensions", {}).get("response_mode") == "full_chunk"
    if full_chunk and len(actions) != metadata["action_horizon"]:
        raise ValueError("full_chunk response must contain action_horizon actions")
    return actions


def build_error(*, request_id, message, code="inference_error"):
    result = dict(type=ERROR_TYPE, request_id=request_id, code=code, message=message)
    messages.check_error(result)
    result["request_id"] = int(request_id)
    return result


# --- Episode reset -----------------------------------------------------------


def build_reset(*, request_id, metadata):
    if not metadata["reset_supported"]:
        raise ValueError("server does not support episode reset")
    result = dict(type=RESET_TYPE, request_id=request_id)
    messages.check_reset(result)
    return result


def build_reset_ack(*, request_id):
    result = dict(type=RESET_ACK_TYPE, request_id=request_id)
    messages.check_reset_ack(result)
    return result


def parse_reset_ack(payload, *, request_id):
    messages.check_reset_ack(payload)
    if payload["request_id"] != request_id:
        raise ValueError("reset acknowledgement request_id mismatch")
    return dict(payload)
