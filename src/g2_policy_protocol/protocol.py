"""Build and parse g2.policy.v3 messages against the negotiated handshake metadata.

Decode frames with ``runtime.codec`` first. Validate the handshake once, then pass the same
metadata to every build_* / parse_* call. Invalid messages raise ValueError.

Client path: validate_metadata -> build_request -> parse_response.
Server path: build_metadata -> parse_request -> runtime image decoding ->
backend.infer -> build_response.
Array layouts and the backend's input/output records are documented in types.
This module normalizes numeric values but never decodes JPEG or loads a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from . import validation as check
from .constraints import validate_rigid_transform
from .types import (
    ACTION_SEMANTICS,
    ACTION_TYPE,
    ARM_VALUES,
    CAMERA_IDS,
    CONTEXT_KEYS,
    CONTROL_QUANTITIES,
    EE_FRAME,
    ENCODING_VALUES,
    ERROR_TYPE,
    FRAME,
    IMAGE_ENCODING_VALUES,
    IMAGE_RESIZE_VALUES,
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
    RESPONSE_MODES,
    SIDES,
    ActionResponse,
    ActionSpec,
    ErrorResponse,
    Metadata,
    ObservationContext,
    ObservationMessage,
    ObservationRequest,
    ObservationSpec,
    ResetAcknowledgement,
    ResetRequest,
    TimingSpec,
    WireImage,
    WireImageHistory,
)
from .validation import finite_action_rows, finite_vector

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ACTION_TYPE",
    "CAMERA_IDS",
    "ENCODING_VALUES",
    "ERROR_TYPE",
    "JOINT_DIM",
    "JOINT_FIELDS",
    "POSE_DIM",
    "POSE_FIELDS",
    "PROTOCOL",
    "SIDES",
    "build_action_spec",
    "build_error",
    "build_metadata",
    "build_request",
    "build_reset",
    "build_reset_ack",
    "build_response",
    "cameras_for_arm",
    "parse_request",
    "parse_reset",
    "parse_reset_ack",
    "parse_response",
    "validate_images",
    "validate_image",
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

_VERSION_HINT = "; the client and server may run different g2-policy-protocol versions"


# --- Handshake ---------------------------------------------------------------


def cameras_for_arm(arm: str) -> tuple[str, ...]:
    """Default cameras: all three for both arms, otherwise head plus the same-side wrist."""
    if arm == "both":
        return CAMERA_IDS
    if arm in SIDES:
        return (CAMERA_IDS[0], CAMERA_IDS[SIDES.index(arm) + 1])
    raise ValueError(f"unsupported arm: {arm!r}")


def build_action_spec(control_quantity: str = "pose_target") -> ActionSpec:
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
    policy_hz: float,
    action_horizon: int = 50,
    arm: str = "both",
    image_encoding: str = "raw",
    image_resize: str = "resize",
    image_sizes: Mapping[str, Sequence[int]] | None = None,
    required_cameras: Sequence[str] | None = None,
    rtc_supported: bool = False,
    requires_joint_state: bool | None = None,
    extensions: Mapping[str, Any] | None = None,
    reset_supported: bool = False,
    action_spec: ActionSpec | None = None,
    observation_spec: ObservationSpec | None = None,
    timing: TimingSpec | None = None,
) -> Metadata:
    """Build validated wire capabilities from backend settings.

    ``observation_spec.joint`` determines the wire ``requires_joint_state``.
    The optional argument supports constructing a default observation spec;
    an explicit duplicate must agree with an explicitly supplied spec.
    """
    if required_cameras is None:
        required_cameras = cameras_for_arm(arm)
    if action_spec is None:
        action_spec = build_action_spec()
    if observation_spec is None:
        observation_spec = dict(
            pose=True,
            joint=bool(requires_joint_state),
            prompt=True,
            history_steps=1,
            history_hz=policy_hz,
            context_keys=[],
        )
    if requires_joint_state is not None:
        check.boolean(requires_joint_state, "metadata.requires_joint_state")
        if requires_joint_state != observation_spec["joint"]:
            raise ValueError("requires_joint_state disagrees with observation_spec.joint")
    if timing is None:
        check.number(policy_hz, "metadata.policy_hz", exclusive_minimum=0)
        timing = dict(action_dt=1 / policy_hz, first_target_offset_s=0)
    metadata: Metadata = dict(
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
        requires_joint_state=observation_spec["joint"],
        action_spec=dict(action_spec),
        observation_spec=dict(observation_spec),
        timing=dict(timing),
        reset_supported=reset_supported,
    )
    if extensions is not None:
        metadata["extensions"] = dict(extensions)
    validate_metadata(metadata)
    return _plain_numbers(metadata)


def _plain_numbers(metadata: Metadata) -> Metadata:
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


def validate_metadata(
    metadata: Mapping[str, Any],
    *,
    policy_hz: float | None = None,
    require_rtc: bool = False,
) -> None:
    """Check the handshake; optionally compare it with the client's own expectations."""
    _fields(metadata, Metadata, "metadata")
    check.one_of(metadata["type"], (METADATA_TYPE,), "metadata.type")
    if metadata["protocol"] != PROTOCOL:
        raise ValueError(f"server speaks {metadata['protocol']!r}, client expects {PROTOCOL!r}{_VERSION_HINT}")
    check.number(metadata["policy_hz"], "metadata.policy_hz", exclusive_minimum=0)
    check.integer(metadata["action_horizon"], "metadata.action_horizon", minimum=1)
    check.one_of(metadata["arm"], ARM_VALUES, "metadata.arm")
    check.array(metadata["required_cameras"], "metadata.required_cameras", maximum=len(CAMERA_IDS), unique=True)
    for camera in metadata["required_cameras"]:
        check.one_of(camera, CAMERA_IDS, "metadata.required_cameras")
    check.one_of(metadata["image_encoding"], IMAGE_ENCODING_VALUES, "metadata.image_encoding")
    check.one_of(metadata["image_resize"], IMAGE_RESIZE_VALUES, "metadata.image_resize")
    check.object_fields(metadata["image_sizes"], "metadata.image_sizes", fields=CAMERA_IDS)
    for camera, size in metadata["image_sizes"].items():
        check.array(size, f"metadata.image_sizes.{camera}", minimum=2, maximum=2)
        for value in size:
            check.integer(value, f"metadata.image_sizes.{camera}", minimum=1)
    for key in ("rtc_supported", "requires_joint_state", "reset_supported"):
        check.boolean(metadata[key], f"metadata.{key}")

    extensions = metadata.get("extensions", {})
    check.object_fields(extensions, "metadata.extensions")
    if "response_mode" in extensions:
        check.one_of(extensions["response_mode"], RESPONSE_MODES, "metadata.extensions.response_mode")

    _fields(metadata["action_spec"], ActionSpec, "metadata.action_spec")
    check.one_of(
        metadata["action_spec"]["control_quantity"], CONTROL_QUANTITIES, "metadata.action_spec.control_quantity"
    )
    observation = metadata["observation_spec"]
    _fields(observation, ObservationSpec, "metadata.observation_spec")
    for key in ("pose", "joint", "prompt"):
        check.boolean(observation[key], f"metadata.observation_spec.{key}")
    check.integer(observation["history_steps"], "metadata.observation_spec.history_steps", minimum=1)
    check.number(observation["history_hz"], "metadata.observation_spec.history_hz", exclusive_minimum=0)
    check.array(observation["context_keys"], "metadata.observation_spec.context_keys", unique=True)
    for key in observation["context_keys"]:
        check.one_of(key, CONTEXT_KEYS, "metadata.observation_spec.context_keys")

    timing = metadata["timing"]
    _fields(timing, TimingSpec, "metadata.timing")
    check.number(timing["action_dt"], "metadata.timing.action_dt", exclusive_minimum=0)
    check.number(timing["first_target_offset_s"], "metadata.timing.first_target_offset_s", minimum=0)
    action_spec = metadata["action_spec"]
    if action_spec != build_action_spec(action_spec["control_quantity"]):
        raise ValueError(f"action_spec must use the canonical layout, units and frames{_VERSION_HINT}")
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
    request_id: int,
    prompt: str,
    state: ArrayLike | None,
    images: Mapping[str, WireImage | None] | None = None,
    metadata: Metadata,
    joint_state: ArrayLike | None = None,
    rtc_enabled: bool = False,
    inference_delay: int = 0,
    prev_chunk_left_over: ArrayLike | None = None,
    observation_time_ns: int | None = None,
    context: ObservationContext | None = None,
) -> ObservationRequest:
    """Client: construct an obs and apply the same checks as the receiving server.

    state is measured pose20; pass None when it is not required or supplied.
    Images must already use the negotiated size/encoding. Full image history
    includes the current frame and permits images=None. Other None-valued
    optional observations are omitted. See types.ObservationRequest for units.
    """
    payload = dict(
        type=OBS_TYPE,
        request_id=request_id,
        prompt=prompt,
        rtc_enabled=rtc_enabled,
        inference_delay=inference_delay,
        prev_chunk_left_over=prev_chunk_left_over,
    )
    optional = dict(
        state=state, joint_state=joint_state, images=images, context=context, observation_time_ns=observation_time_ns
    )
    payload.update({key: value for key, value in optional.items() if value is not None})
    return parse_request(payload, metadata=metadata)


def parse_request(payload: object, *, metadata: Metadata) -> ObservationRequest:
    """Server: check structure, then requirements, arrays, context, images and RTC.

    Returns a new dictionary with normalized numeric arrays and explicit
    prompt/context/RTC defaults. JPEG stays encoded until the runtime worker
    converts this wire request to Observation.
    Metadata-dependent optional state and image fields remain absent.
    """
    payload = _envelope(payload, ObservationMessage, OBS_TYPE)
    if "prompt" in payload:
        check.text(payload["prompt"], "obs.prompt", min_length=0)
    if "rtc_enabled" in payload:
        check.boolean(payload["rtc_enabled"], "obs.rtc_enabled")
    for key in ("inference_delay", "observation_time_ns"):
        if key in payload:
            check.integer(payload[key], f"obs.{key}", minimum=0)
    observation = metadata["observation_spec"]
    result = dict(payload)
    prompt = payload.get("prompt", "")
    if observation["prompt"] and not prompt.strip():
        raise ValueError("server requires a nonempty prompt")
    result["prompt"] = prompt
    for key, required, width in (
        ("state", observation["pose"], POSE_DIM),
        ("joint_state", observation["joint"], JOINT_DIM),
    ):
        if required or key in payload:
            result[key] = finite_vector(payload.get(key), width, key)
    result["context"] = _parse_context(payload.get("context", {}), metadata)
    if "images" in payload:
        result["images"] = validate_images(payload["images"], metadata=metadata)
    elif "image_history" not in result["context"]:
        raise ValueError(f"obs requires images or context.image_history{_VERSION_HINT}")
    if observation["history_steps"] > 1 and "observation_time_ns" not in payload:
        raise ValueError("history requires observation_time_ns for the latest sample")
    rtc_enabled, inference_delay, prefix = _parse_rtc(payload, metadata)
    result.update(rtc_enabled=rtc_enabled, inference_delay=inference_delay, prev_chunk_left_over=prefix)
    return cast(ObservationRequest, result)


def validate_images(
    images: object,
    *,
    metadata: Metadata,
    decoded: bool = False,
) -> dict[str, WireImage]:
    """Check required cameras and sizes, before or (decoded=True) after JPEG decoding."""
    check.object_fields(images, "images", fields=CAMERA_IDS)
    images = cast(Mapping[str, Any], images)
    if any(images.get(camera) is None for camera in metadata["required_cameras"]):
        raise ValueError("missing required cameras")
    encoding = "raw" if decoded else metadata["image_encoding"]
    result: dict[str, WireImage] = {}
    for camera, image in images.items():
        if image is None:
            continue
        image = validate_image(image, encoding=encoding)
        expected = metadata["image_sizes"].get(camera)
        if isinstance(image, np.ndarray) and expected is not None and tuple(image.shape[:2]) != tuple(expected):
            raise ValueError(f"{camera} size disagrees with metadata image_sizes")
        result[camera] = image
    return result


def validate_image(value: object, *, encoding: str) -> WireImage:
    """Validate an image's wire representation without decoding JPEG bytes."""
    if encoding == "jpeg":
        if not isinstance(value, bytes) or not value:
            raise ValueError("jpeg image must be nonempty bytes")
        return value
    if encoding != "raw":
        raise ValueError(f"unsupported image encoding: {encoding!r}")
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[-1] != 3
        or value.shape[0] < 1
        or value.shape[1] < 1
    ):
        raise ValueError("raw image must be nonempty RGB uint8 HxWx3")
    return value


def _parse_context(context: object, metadata: Metadata) -> ObservationContext:
    """Validate the context keys declared by observation_spec.context_keys."""
    observation = metadata["observation_spec"]
    check.object_fields(context, "context", fields=CONTEXT_KEYS, required=observation["context_keys"])
    context = cast(Mapping[str, Any], context)
    steps = observation["history_steps"]
    result: dict[str, Any] = {}
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
    return cast(ObservationContext, result)


def _parse_history(value: object, steps: int, width: int, name: str) -> NDArray[np.float32]:
    history = np.asarray(value, dtype=np.float32)
    if history.shape != (steps, width) or not np.isfinite(history).all():
        raise ValueError(f"{name} must be finite (L, {width})")
    return history


def _parse_image_history(value: object, steps: int, metadata: Metadata) -> dict[str, WireImageHistory]:
    cameras = metadata["required_cameras"]
    check.object_fields(value, "image_history", fields=cameras, required=cameras)
    value = cast(Mapping[str, Any], value)
    result: dict[str, WireImageHistory] = {}
    for camera, frames in value.items():
        if metadata["image_encoding"] == "jpeg":
            if (
                not isinstance(frames, (list, tuple))
                or len(frames) != steps
                or any(not isinstance(frame, bytes) or not frame for frame in frames)
            ):
                raise ValueError(f"image_history.{camera} must be {steps} nonempty JPEG bytes{_VERSION_HINT}")
            result[camera] = list(frames)
            continue
        frames = np.asarray(frames)
        if frames.dtype != np.uint8 or frames.ndim != 4 or len(frames) != steps:
            raise ValueError(f"image_history must contain raw uint8 (L,H,W,3) frames{_VERSION_HINT}")
        size = metadata["image_sizes"].get(camera)
        if frames.shape[-1] != 3 or min(frames.shape[1:3]) < 1 or (size and tuple(frames.shape[1:3]) != tuple(size)):
            raise ValueError("image_history size disagrees with metadata")
        result[camera] = frames
    return result


def _parse_rtc(
    payload: Mapping[str, Any],
    metadata: Metadata,
) -> tuple[bool, int, NDArray[np.float32] | None]:
    """Return (enabled, inference_delay, prefix); RTC fields require rtc_enabled."""
    enabled = payload.get("rtc_enabled", False)
    delay = int(payload.get("inference_delay", 0))
    raw_prefix = payload.get("prev_chunk_left_over")
    if enabled and not metadata["rtc_supported"]:
        raise ValueError("server does not support RTC")
    if not enabled and (delay or raw_prefix is not None):
        raise ValueError("RTC fields require rtc_enabled=true")
    prefix = None if raw_prefix is None else _parse_actions(raw_prefix, metadata)
    if delay >= metadata["action_horizon"]:
        raise ValueError("inference_delay exceeds action horizon")
    if prefix is not None and delay > len(prefix):
        raise ValueError("inference_delay exceeds RTC prefix")
    return enabled, delay, prefix


# --- Action / error ----------------------------------------------------------


def _parse_actions(value: object, metadata: Metadata) -> NDArray[np.float32]:
    """Finite nonempty (T, D) actions, T bounded by the negotiated horizon."""
    actions = finite_action_rows(value, metadata["action_spec"]["dim"])
    if len(actions) > metadata["action_horizon"]:
        raise ValueError("actions exceeds action_horizon")
    return actions


def build_response(
    *,
    request_id: int,
    actions: ArrayLike,
    metadata: Metadata,
    server_timing: Mapping[str, float] | None = None,
    policy_timing: Mapping[str, float] | None = None,
    rtc_info: Mapping[str, Any] | None = None,
) -> ActionResponse:
    """Server: validate absolute targets and diagnostics, attach the request ID."""
    check.integer(request_id, "action.request_id", minimum=0)
    result = dict(type=ACTION_TYPE, request_id=int(request_id), actions=actions)
    for key, block in (("server_timing", server_timing), ("policy_timing", policy_timing), ("rtc_info", rtc_info)):
        if block is not None:
            result[key] = dict(block)
    result["actions"] = _action_values(result, metadata)
    for key in ("server_timing", "policy_timing"):
        if key in result:
            result[key] = {name: float(value) for name, value in result[key].items()}
    return cast(ActionResponse, result)


def parse_response(payload: object, *, request_id: int, metadata: Metadata) -> NDArray[np.float32]:
    """Client: match the request ID, return validated actions or raise on remote error."""
    response = _response(payload, ActionResponse, ACTION_TYPE, request_id)
    return _action_values(response, metadata)


def _action_values(payload: Mapping[str, Any], metadata: Metadata) -> NDArray[np.float32]:
    for key in ("server_timing", "policy_timing"):
        if key in payload:
            check.object_fields(payload[key], f"action.{key}")
            for value in payload[key].values():
                check.number(value, f"action.{key}", minimum=0)
    if "rtc_info" in payload:
        check.object_fields(payload["rtc_info"], "action.rtc_info")
    actions = _parse_actions(payload["actions"], metadata)
    full_chunk = metadata.get("extensions", {}).get("response_mode") == "full_chunk"
    if full_chunk and len(actions) != metadata["action_horizon"]:
        raise ValueError("full_chunk response must contain action_horizon actions")
    return actions


def build_error(*, request_id: int, message: str, code: str = "inference_error") -> ErrorResponse:
    """Server: reject a request without returning executable actions."""
    result = dict(type=ERROR_TYPE, request_id=request_id, code=code, message=message)
    _error_fields(result)
    result["request_id"] = int(request_id)
    return cast(ErrorResponse, result)


# --- Episode reset -----------------------------------------------------------


def build_reset(*, request_id: int, metadata: Metadata) -> ResetRequest:
    """Client: request episode reset on this connection."""
    return parse_reset(dict(type=RESET_TYPE, request_id=request_id), metadata=metadata)


def parse_reset(payload: object, *, metadata: Metadata) -> ResetRequest:
    """Server: validate a reset request and require the advertised capability."""
    payload = _envelope(payload, ResetRequest, RESET_TYPE)
    if not metadata["reset_supported"]:
        raise ValueError("server does not support episode reset")
    return cast(ResetRequest, dict(payload))


def build_reset_ack(*, request_id: int) -> ResetAcknowledgement:
    result = dict(type=RESET_ACK_TYPE, request_id=request_id)
    _envelope(result, ResetAcknowledgement, RESET_ACK_TYPE)
    return cast(ResetAcknowledgement, result)


def parse_reset_ack(payload: object, *, request_id: int) -> ResetAcknowledgement:
    """Client: match a reset acknowledgement or raise on remote error."""
    response = _response(payload, ResetAcknowledgement, RESET_ACK_TYPE, request_id)
    return cast(ResetAcknowledgement, dict(response))


# --- Shared envelope checks (field sets come from types.py) --------------------


def _fields(value: object, record: type, name: str) -> None:
    try:
        check.object_fields(value, name, fields=record.__annotations__, required=record.__required_keys__)
    except ValueError as exc:
        if isinstance(value, Mapping):
            raise ValueError(f"{exc}{_VERSION_HINT}") from exc
        raise


def _envelope(payload: object, record: type, kind: str) -> Mapping[str, Any]:
    _fields(payload, record, kind)
    payload = cast(Mapping[str, Any], payload)
    check.one_of(payload["type"], (kind,), f"{kind}.type")
    check.integer(payload["request_id"], f"{kind}.request_id", minimum=0)
    return payload


def _error_fields(payload: object) -> Mapping[str, Any]:
    error = _envelope(payload, ErrorResponse, ERROR_TYPE)
    check.text(error["code"], "error.code", min_length=1)
    check.text(error["message"], "error.message", min_length=1)
    return error


def _response(payload: object, record: type, kind: str, request_id: int) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("response must be an object")
    is_error = payload.get("type") == ERROR_TYPE
    response = _error_fields(payload) if is_error else _envelope(payload, record, kind)
    check.integer(request_id, "request_id", minimum=0)
    if response["request_id"] != request_id:
        raise ValueError("response request_id does not match observation or reset")
    if is_error:
        raise RuntimeError(f"policy server {response['code']}: {response['message']}")
    return response
