"""Public data definitions for ``g2.policy.v3`` and the backend boundary.

Read ``Metadata`` for service requirements, ``Observation`` for backend
inputs, and ``InferenceResult`` for backend outputs. Messages are ordinary
dictionaries; ``protocol`` validates their values and normalizes numeric arrays.

Connection exchange (the server sends the first application message):
    server -> client: Metadata
    client validates it; there is no separate handshake acknowledgement.
    client -> server: ObservationMessage  -> ActionResponse or ErrorResponse
    client -> server: ResetRequest        -> ResetAcknowledgement or ErrorResponse
Request IDs belong to the client and are echoed by every response.

Array annotations describe normalized data, not arbitrary untrusted input.
NumPy is imported only by type checkers, so these definitions need only the
standard library at runtime. Shapes and units are documented on each record.
"""

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray


# Fixed wire vocabulary and vector layouts. Left arm always precedes right.
PROTOCOL = "g2.policy.v3"
SIDES = ("left", "right")
CAMERA_IDS = ("head_color", "hand_left_color", "hand_right_color")

# Per-side components, concatenated left -> right. rot6d is the first two
# rotation-matrix columns (Fortran order); gripper is 0 open .. 1 closed.
POSE_SIDE_FIELDS = ("x", "y", "z", *(f"rot6d_{i}" for i in range(6)), "gripper")
POSE_SIDE_UNITS = ("m",) * 3 + ("dimensionless",) * 6 + ("close_ratio",)
JOINT_SIDE_FIELDS = (*(f"joint_{i}" for i in range(1, 8)), "gripper")
JOINT_SIDE_UNITS = ("rad",) * 7 + ("close_ratio",)
POSE_FIELDS = tuple(f"{side}_{name}" for side in SIDES for name in POSE_SIDE_FIELDS)
JOINT_FIELDS = tuple(f"{side}_{name}" for side in SIDES for name in JOINT_SIDE_FIELDS)
POSE_ARM_DIM = len(POSE_SIDE_FIELDS)
JOINT_ARM_DIM = len(JOINT_SIDE_FIELDS)
POSE_DIM = len(POSE_FIELDS)
JOINT_DIM = len(JOINT_FIELDS)

FRAME = "base_link"
EE_FRAME = "flange"
ACTION_SEMANTICS = "absolute"
CONTROL_QUANTITIES = ("pose_target", "joint_position")

ENCODING_VALUES = ("msgpack", "json")
ARM_VALUES = ("both", "left", "right")
IMAGE_ENCODING_VALUES = ("raw", "jpeg")
IMAGE_RESIZE_VALUES = ("resize", "letterbox")
RESPONSE_MODES = ("full_chunk", "variable_chunk")
CONTEXT_KEYS = (
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

METADATA_TYPE = "metadata"
OBS_TYPE = "obs"
ACTION_TYPE = "action"
ERROR_TYPE = "error"
RESET_TYPE = "reset"
RESET_ACK_TYPE = "reset_ack"


FloatArray: TypeAlias = "NDArray[np.float32]"
RGBImage: TypeAlias = "NDArray[np.uint8]"
WireImage: TypeAlias = "RGBImage | bytes"
WireImageHistory: TypeAlias = "NDArray[np.uint8] | list[bytes]"


class ActionSpec(TypedDict):
    """Absolute dual-arm targets: pose20 or joint16, always left then right.

    Pose: each arm has xyz (m), rotation's first two columns (rot6d), gripper.
    The pose is the flange in base_link. Joint: seven angles (rad), gripper.
    Grippers are absolute close ratios, 0 open and 1 closed. A single-arm model
    must fill the inactive arm from measured state before returning actions.
    """

    control_quantity: str
    dim: int
    fields: list[str]
    units: list[str]
    frame: str
    ee_frame: str
    representation: str


class ObservationSpec(TypedDict):
    """Required measured quantities, prompt, history length and extra context.

    Histories run oldest to newest and include the current sample as the last
    entry, separated by 1/history_hz seconds. Requirements are checked against
    each request; optional state fields are not synthesized from other fields.
    """

    pose: bool
    joint: bool
    prompt: bool
    history_steps: int
    history_hz: float
    context_keys: list[str]


class TimingSpec(TypedDict):
    """Action spacing and first target offset from the observation (seconds).

    Target i has offset first_target_offset_s + i * action_dt. This is the
    action timeline, not a promise about GPU inference frequency or latency.
    """

    action_dt: float
    first_target_offset_s: float


class _MetadataFields(TypedDict):
    type: Literal["metadata"]
    protocol: str
    policy_hz: float
    action_horizon: int
    arm: str
    image_encoding: str
    image_resize: str
    image_sizes: dict[str, list[int]]
    required_cameras: list[str]
    rtc_supported: bool
    requires_joint_state: bool
    action_spec: ActionSpec
    observation_spec: ObservationSpec
    timing: TimingSpec
    reset_supported: bool


class Metadata(_MetadataFields, total=False):
    """Server's first message; validate it before exchanging requests."""

    extensions: dict[str, Any]


class PolicyCapabilities(TypedDict):
    """Handshake fields derived from resolved policy semantics."""

    policy_hz: float
    reset_supported: bool
    action_spec: ActionSpec
    observation_spec: ObservationSpec
    timing: TimingSpec


class MetadataOptions(PolicyCapabilities, total=False):
    """Backend settings accepted by ``protocol.build_metadata``."""

    action_horizon: int
    arm: str
    image_encoding: str
    image_resize: str
    image_sizes: dict[str, list[int]]
    required_cameras: list[str]
    rtc_supported: bool
    extensions: dict[str, Any]


class ObservationReferences(TypedDict, total=False):
    """Non-image references and histories requested by observation_spec.context_keys.

    Pose vectors are pose20; joint vectors are joint16. Command references are
    commanded targets, not measurements. start_* are episode-start references;
    previous_command_pose is the command at the previous history sample.
    object_* are additional application-provided references in the same layout.

    base_to_parent is T_armbase<-base, a (4,4) rigid transform mapping base_link
    coordinates to arm_base_link. State histories have shape (L,20)/(L,16).
    All histories run oldest to newest and include the current sample last.
    """

    command_pose: FloatArray
    command_joint: FloatArray
    previous_command_pose: FloatArray
    object_pose: FloatArray
    object_joint: FloatArray
    start_pose: FloatArray
    start_joint: FloatArray
    base_to_parent: FloatArray
    pose_history: FloatArray
    joint_history: FloatArray


class ObservationContext(ObservationReferences, total=False):
    """Wire context: numeric references plus optional encoded image histories.

    Image histories have shape (L,H,W,3) for raw uint8, or L JPEG byte strings.
    The runtime moves decoded image histories into Observation.images.
    """

    image_history: dict[str, WireImageHistory]


class _ObservationIdentity(TypedDict):
    type: Literal["obs"]
    request_id: int


class _ObservationValues(TypedDict, total=False):
    state: FloatArray
    joint_state: FloatArray
    observation_time_ns: int


class _WireImages(TypedDict, total=False):
    images: dict[str, WireImage]


class ObservationMessage(_ObservationIdentity, _ObservationValues, _WireImages, total=False):
    """Wire obs message, before metadata-dependent defaults are applied.

    Only type and request_id are unconditionally required. Metadata determines
    which state, image, prompt, context and timestamp fields must be present.
    Use build_request/parse_request instead of relying on TypedDict at runtime.
    """

    prompt: str
    context: ObservationContext
    rtc_enabled: bool
    inference_delay: int
    prev_chunk_left_over: "FloatArray | None"


class ObservationRequest(_ObservationIdentity, _ObservationValues, _WireImages):
    """Validated wire input returned by protocol.parse_request.

    state is measured pose20, joint_state is measured joint16. They remain
    absent when neither supplied nor required. Images are RGB uint8 (H,W,3)
    for raw transport and bytes for JPEG. The runtime decodes JPEG in its model
    thread before creating an Observation. With image_history, images may be
    absent altogether at this wire boundary.

    The fields below always exist after parsing. Defaults are empty prompt and
    context, RTC disabled, zero delay and no leftover chunk. An enabled RTC
    prefix contains absolute actions in the negotiated action layout; delay
    counts action steps, not seconds. observation_time_ns uses the client's
    monotonic clock and names the newest sample; server clocks are independent.
    """

    prompt: str
    context: ObservationContext
    rtc_enabled: bool
    inference_delay: int
    prev_chunk_left_over: "FloatArray | None"


class Observation(_ObservationIdentity, _ObservationValues):
    """Backend input: validated robot data with decoded RGB uint8 images.

    images always exists. Each value is (H,W,3) for current images, or
    (L,H,W,3) when an image history was sent, oldest to newest including now.
    Image history is consumed from wire context and appears only in images;
    other references/history arrays remain in context. No JPEG decoding or
    image-history selection is required in the model adapter.

    State layouts, timestamp and RTC defaults follow ObservationRequest.
    Arrays may be shared with the parsed request; adapters must not mutate them.
    """

    prompt: str
    images: dict[str, "NDArray[np.uint8]"]
    context: ObservationReferences
    rtc_enabled: bool
    inference_delay: int
    prev_chunk_left_over: "FloatArray | None"


class _InferenceActions(TypedDict):
    actions: "ArrayLike"


class InferenceResult(_InferenceActions, total=False):
    """Backend output, without a wire envelope or request ID.

    actions must be finite absolute dual-arm targets of shape (T,D):
    1 <= T <= action_horizon, D=20 for pose_target or 16 for joint_position.
    full_chunk requires T=action_horizon. Model padding, delta representations
    and model coordinate frames must be resolved by the model adapter first.
    The server normalizes actions to float32 and echoes the request ID.

    Timing maps contain finite nonnegative milliseconds; rtc_info is optional
    diagnostic data. The transport does not use diagnostics to schedule actions.
    """

    server_timing: dict[str, float]
    policy_timing: dict[str, float]
    rtc_info: dict[str, Any]


class _ActionFields(TypedDict):
    type: Literal["action"]
    request_id: int
    actions: FloatArray


class ActionResponse(_ActionFields, total=False):
    """Validated action message; its arrays follow InferenceResult's semantics."""

    server_timing: dict[str, float]
    policy_timing: dict[str, float]
    rtc_info: dict[str, Any]


class ErrorResponse(TypedDict):
    """Request rejection or inference failure; never contains executable actions."""

    type: Literal["error"]
    request_id: int
    code: str
    message: str


class ResetRequest(TypedDict):
    """Clear backend episode state on this connection; does not move the robot."""

    type: Literal["reset"]
    request_id: int


class ResetAcknowledgement(TypedDict):
    """Acknowledges that the connection's episode state has been reset."""

    type: Literal["reset_ack"]
    request_id: int
