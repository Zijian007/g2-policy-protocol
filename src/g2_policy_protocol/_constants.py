"""g2.policy.v3 fixed layouts and enums: the single source for wire and dataset names.

Standard library only, so data.contract can share the vector layouts.
"""

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
