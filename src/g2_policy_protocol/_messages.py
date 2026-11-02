"""Static structure of each g2.policy.v3 message: field names, types, enums and ranges.

Array shapes, finite values and checks against the negotiated metadata live in
protocol.py.
"""

from collections.abc import Mapping

from . import validation as check
from ._constants import (
    ACTION_TYPE,
    ARM_VALUES,
    CAMERA_IDS,
    CONTEXT_KEYS,
    CONTROL_QUANTITIES,
    ERROR_TYPE,
    IMAGE_ENCODING_VALUES,
    IMAGE_RESIZE_VALUES,
    METADATA_TYPE,
    OBS_TYPE,
    PROTOCOL,
    RESET_ACK_TYPE,
    RESET_TYPE,
    RESPONSE_MODES,
)

METADATA_REQUIRED = (
    "type", "protocol", "policy_hz", "action_horizon", "arm", "required_cameras",
    "image_encoding", "image_resize", "image_sizes", "rtc_supported", "requires_joint_state",
    "action_spec", "observation_spec", "timing", "reset_supported",
)
ACTION_SPEC_FIELDS = ("control_quantity", "dim", "fields", "units", "frame", "ee_frame", "representation")
OBSERVATION_SPEC_FIELDS = ("pose", "joint", "prompt", "history_steps", "history_hz", "context_keys")
TIMING_FIELDS = ("action_dt", "first_target_offset_s")
OBS_FIELDS = (
    "type", "request_id", "prompt", "state", "joint_state", "images", "rtc_enabled",
    "inference_delay", "prev_chunk_left_over", "observation_time_ns", "context",
)
ACTION_FIELDS = ("type", "request_id", "actions", "server_timing", "policy_timing", "rtc_info")
ERROR_FIELDS = ("type", "request_id", "code", "message")
RESET_FIELDS = ("type", "request_id")

# Unknown/missing wire fields and protocol/layout mismatches usually mean the
# two ends installed different versions of this package.
VERSION_HINT = "; the client and server may run different g2-policy-protocol versions"


def wire_fields(value, name, fields, required):
    """Exact field set of a wire object; a mismatch points at a version difference."""
    try:
        check.object_fields(value, name, fields=fields, required=required)
    except ValueError as exc:
        if not isinstance(value, Mapping):
            raise
        raise ValueError(f"{exc}{VERSION_HINT}") from exc


def check_envelope(payload, kind, fields, required):
    wire_fields(payload, kind, fields, required)
    check.one_of(payload["type"], (kind,), f"{kind}.type")
    check.integer(payload["request_id"], f"{kind}.request_id", minimum=0)


def check_metadata(metadata):
    wire_fields(metadata, "metadata", (*METADATA_REQUIRED, "extensions"), METADATA_REQUIRED)
    check.one_of(metadata["type"], (METADATA_TYPE,), "metadata.type")
    if metadata["protocol"] != PROTOCOL:
        raise ValueError(f"server speaks {metadata['protocol']!r}, client expects {PROTOCOL!r}{VERSION_HINT}")
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

    # The remaining action_spec values must equal the canonical layout (protocol.validate_metadata).
    action_spec = metadata["action_spec"]
    wire_fields(action_spec, "metadata.action_spec", ACTION_SPEC_FIELDS, ACTION_SPEC_FIELDS)
    check.one_of(action_spec["control_quantity"], CONTROL_QUANTITIES, "metadata.action_spec.control_quantity")

    observation = metadata["observation_spec"]
    wire_fields(observation, "metadata.observation_spec", OBSERVATION_SPEC_FIELDS, OBSERVATION_SPEC_FIELDS)
    for key in ("pose", "joint", "prompt"):
        check.boolean(observation[key], f"metadata.observation_spec.{key}")
    check.integer(observation["history_steps"], "metadata.observation_spec.history_steps", minimum=1)
    check.number(observation["history_hz"], "metadata.observation_spec.history_hz", exclusive_minimum=0)
    check.array(observation["context_keys"], "metadata.observation_spec.context_keys", unique=True)
    for key in observation["context_keys"]:
        check.one_of(key, CONTEXT_KEYS, "metadata.observation_spec.context_keys")

    timing = metadata["timing"]
    wire_fields(timing, "metadata.timing", TIMING_FIELDS, TIMING_FIELDS)
    check.number(timing["action_dt"], "metadata.timing.action_dt", exclusive_minimum=0)
    check.number(timing["first_target_offset_s"], "metadata.timing.first_target_offset_s", minimum=0)


def check_obs(payload):
    check_envelope(payload, OBS_TYPE, OBS_FIELDS, ("type", "request_id"))
    if "prompt" in payload:
        check.text(payload["prompt"], "obs.prompt", min_length=0)
    if "rtc_enabled" in payload:
        check.boolean(payload["rtc_enabled"], "obs.rtc_enabled")
    for key in ("inference_delay", "observation_time_ns"):
        if key in payload:
            check.integer(payload[key], f"obs.{key}", minimum=0)


def check_action(payload):
    check_envelope(payload, ACTION_TYPE, ACTION_FIELDS, ("type", "request_id", "actions"))
    for key in ("server_timing", "policy_timing"):
        if key in payload:
            check.object_fields(payload[key], f"action.{key}")
            for value in payload[key].values():
                check.number(value, f"action.{key}", minimum=0)
    if "rtc_info" in payload:
        check.object_fields(payload["rtc_info"], "action.rtc_info")


def check_error(payload):
    check_envelope(payload, ERROR_TYPE, ERROR_FIELDS, ERROR_FIELDS)
    check.text(payload["code"], "error.code", min_length=1)
    check.text(payload["message"], "error.message", min_length=1)


def check_reset(payload):
    check_envelope(payload, RESET_TYPE, RESET_FIELDS, RESET_FIELDS)


def check_reset_ack(payload):
    check_envelope(payload, RESET_ACK_TYPE, RESET_FIELDS, RESET_FIELDS)
