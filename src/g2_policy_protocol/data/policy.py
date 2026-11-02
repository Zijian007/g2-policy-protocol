"""The ``policy`` section of a training config: defaults, validation and derived wire capabilities.

A policy has ``state`` and ``action`` groups, each with a ``pose`` and a ``joint``
branch, plus ``timing``. A branch says whether it is enabled, whether it is a
delta, relative to which reference, and (for poses) in which coordinates.
"""

import copy
import math

from g2_policy_protocol import protocol
from g2_policy_protocol.validation import validate_rigid_transform

SOURCES = ("state", "action")
METHODS = ("component", "se3")
KINDS = ("pose", "joint")
# Branch kind -> config quantity -> wire control_quantity.
QUANTITIES = {
    "joint": {"position": "joint_position"},
    "pose": {"pose": "pose_target"},
}
COORDINATES = {
    "parent_frame": "base_link",
    "ee_frame": "flange",
    "rotation": "rot6d",
    "flange_to_tcp": None,
}


# --- Queries shared by training, datasets and deployment ----------------------


def active_action(policy):
    """Return (kind, branch) of the single enabled action branch."""
    kind = next(kind for kind, branch in policy["action"].items() if branch["enabled"])
    return kind, policy["action"][kind]


def control_quantity(policy):
    kind, branch = active_action(policy)
    return QUANTITIES[kind][branch["quantity"]]


def uses_kind(policy, kind):
    """True when pose/joint values are needed as state or action."""
    return policy["state"][kind]["enabled"] or policy["action"][kind]["enabled"]


def command_reference_kinds(policy):
    """Kinds with an enabled branch relative to the recorded command (needs command_<kind>)."""
    return tuple(
        kind for kind in KINDS
        if any(_relative_to_command(branch) for branch in (policy["state"][kind], policy["action"][kind]))
    )


def uses_arm_base(policy):
    """True when an enabled pose branch is expressed in arm_base_link (needs base_to_parent)."""
    return any(
        branch["enabled"] and branch["coordinates"]["parent_frame"] == "arm_base_link"
        for branch in (policy["state"]["pose"], policy["action"]["pose"])
    )


def _relative_to_command(branch):
    return branch["enabled"] and branch["delta"] and branch["reference"]["source"] == "action"


def policy_capabilities(policy, cameras, *, prompt, reset):
    """Derive build_metadata arguments (observation needs, timing, action layout) from a policy."""
    timing = policy["timing"]
    pose, joint = uses_kind(policy, "pose"), uses_kind(policy, "joint")
    context = [f"command_{kind}" for kind in command_reference_kinds(policy)]
    if uses_arm_base(policy):
        context.append("base_to_parent")
    if timing["history_steps"] > 1:
        context.extend((["pose_history"] if pose else []) + (["joint_history"] if joint else []))
        if cameras:
            context.append("image_history")
    dt = timing["stride_steps"] / timing["sample_hz"]
    return {
        "policy_hz": 1 / dt,
        "requires_joint_state": joint,
        "reset_supported": reset,
        "action_spec": protocol.build_action_spec(control_quantity(policy)),
        "observation_spec": {
            "pose": pose,
            "joint": joint,
            "prompt": prompt,
            "history_steps": timing["history_steps"],
            "history_hz": timing["sample_hz"],
            "context_keys": context,
        },
        "timing": {
            "action_dt": dt,
            "first_target_offset_s": timing["target_offset_steps"] / timing["sample_hz"],
        },
    }


# --- Resolution and validation ------------------------------------------------


def resolve_policy(raw, *, horizon, sample_hz=30.0):
    """Fill defaults into a partial policy and validate it once, at the configuration boundary.

    The model owns its prediction horizon; a copied ``timing.action_horizon`` must match it.
    """
    if "action_horizon" in raw.get("timing", {}):
        declared = raw["timing"]["action_horizon"]
        if type(declared) is not int or declared != horizon:
            raise ValueError(
                "policy.timing.action_horizon must match the model prediction horizon; adapters cannot override it"
            )
    policy = {group: {kind: _branch_defaults(group, kind) for kind in ("joint", "pose")} for group in SOURCES}
    policy["state"]["pose"]["enabled"] = True
    policy["action"]["pose"]["enabled"] = True
    policy["timing"] = {
        "sample_hz": sample_hz,
        "history_steps": 1,
        "action_horizon": horizon,
        "target_offset_steps": 0,
        "stride_steps": 1,
    }
    merge(policy, raw, "policy")
    validate_policy(policy)
    return policy


def merge(base, override, path=""):
    """Merge explicit branch switches and partial leaves; never migrate another schema."""
    if not isinstance(override, dict):
        raise ValueError(f"{path or 'settings'} must be a YAML mapping")
    for key, value in override.items():
        name = f"{path}.{key}" if path else key
        if key not in base:
            raise ValueError(f"Unknown YAML field: {name}")
        if isinstance(base[key], dict):
            merge(base[key], value, name)
        else:
            base[key] = copy.deepcopy(value)
    # Turning delta off also clears its reference and method unless they were given.
    if path in ("policy.state.joint", "policy.state.pose", "policy.action.joint", "policy.action.pose"):
        if override.get("delta") is False:
            if "reference" not in override:
                base["reference"] = dict.fromkeys(base["reference"])
            if "method" not in override:
                base["method"] = None


def validate_policy(policy):
    if not isinstance(policy, dict) or policy.keys() != {"state", "action", "timing"}:
        raise ValueError("policy fields must be state, action and timing")
    for group in SOURCES:
        branches = policy[group]
        if not isinstance(branches, dict) or branches.keys() != {"joint", "pose"}:
            raise ValueError(f"policy.{group} fields must be joint and pose")
        for kind, branch in branches.items():
            _validate_branch_fields(branch, group, kind)
            _validate_branch_delta(branch, group, kind)
        active = [kind for kind, branch in branches.items() if branch["enabled"]]
        if not active or (group == "action" and len(active) != 1):
            raise ValueError(
                f"policy.{group} requires {'exactly one' if group == 'action' else 'at least one'} active branch"
            )
        pose = branches["pose"]
        if pose["enabled"]:
            _validate_coordinates(pose["coordinates"], f"policy.{group}.pose.coordinates")
    timing = policy["timing"]
    for key in ("history_steps", "action_horizon", "target_offset_steps", "stride_steps"):
        value = timing[key]
        if type(value) is not int or value < (0 if key == "target_offset_steps" else 1):
            raise ValueError(f"policy.timing.{key} must be a valid integer step count")
    hz = timing["sample_hz"]
    if type(hz) not in (int, float) or not math.isfinite(hz) or hz <= 0:
        raise ValueError("policy.timing.sample_hz must be a positive finite number")
    if timing["history_steps"] == 1:
        for kind, branch in policy["state"].items():
            if branch["enabled"] and branch["delta"] and branch["reference"]["source"] == "state":
                raise ValueError(f"policy.state.{kind} relative to source: state requires history_steps > 1")
    kind, branch = active_action(policy)
    _choice(branch["quantity"], QUANTITIES[kind], f"policy.action.{kind}.quantity")


def _branch_defaults(group, kind):
    result = {
        "enabled": False,
        "delta": False,
        "method": None,
        "reference": {"source": None, "scope": None},
    }
    if kind == "pose":
        result["coordinates"] = copy.deepcopy(COORDINATES)
    if group == "action":
        result.update(quantity="pose" if kind == "pose" else "position", source="action")
    return result


def _choice(value, choices, path):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{path} must be one of {tuple(choices)}")
    return choices[value] if isinstance(choices, dict) else value


def _validate_branch_fields(branch, group, kind):
    path = f"policy.{group}.{kind}"
    expected = _branch_defaults(group, kind)
    if not isinstance(branch, dict) or branch.keys() != expected.keys():
        raise ValueError(f"{path} fields must be {sorted(expected)}")
    if type(branch["enabled"]) is not bool:
        raise ValueError(f"{path}.enabled must be a YAML boolean")
    if type(branch["delta"]) is not bool:
        raise ValueError(f"{path}.delta must be a YAML boolean")
    if not isinstance(branch["reference"], dict) or branch["reference"].keys() != expected["reference"].keys():
        raise ValueError(f"{path}.reference fields must be {sorted(expected['reference'])}")
    if kind == "pose" and (
        not isinstance(branch["coordinates"], dict) or branch["coordinates"].keys() != COORDINATES.keys()
    ):
        raise ValueError(f"{path}.coordinates fields must be {sorted(COORDINATES)}")


def _validate_branch_delta(branch, group, kind):
    path = f"policy.{group}.{kind}"
    if group == "action":
        _choice(branch["source"], SOURCES, f"{path}.source")
    reference = branch["reference"]
    if not branch["delta"]:
        if branch["method"] is not None:
            raise ValueError(f"{path}: method requires delta: true")
        if any(value is not None for value in reference.values()):
            raise ValueError(f"{path}: reference requires delta: true")
        return
    if kind == "joint":
        if branch["method"] != "component":
            raise ValueError(f"{path}.method must be component for joint deltas")
    elif branch["method"] not in METHODS:
        raise ValueError(f"{path}.method must be component or se3")
    _choice(reference["source"], SOURCES, f"{path}.reference.source")
    _choice(reference["scope"], ("chunk", "step"), f"{path}.reference.scope")


def _validate_coordinates(coordinates, path):
    _choice(coordinates["parent_frame"], ("base_link", "arm_base_link"), f"{path}.parent_frame")
    for key in ("ee_frame", "rotation"):
        if not isinstance(coordinates[key], str) or not coordinates[key].strip():
            raise ValueError(f"{path}.{key} must be a nonempty string")
    if coordinates["rotation"] not in ("rot6d", "quaternion", "axis_angle"):
        raise ValueError(f"{path}.rotation must be rot6d, quaternion or axis_angle")
    tcp = coordinates["flange_to_tcp"]
    if tcp is not None:
        if not isinstance(tcp, dict) or tcp.keys() != {"left", "right"}:
            raise ValueError(f"{path}.flange_to_tcp must contain left and right transforms")
        for side, transform in tcp.items():
            validate_rigid_transform(transform, f"{path}.flange_to_tcp.{side}")
    if (coordinates["ee_frame"] == "flange") != (tcp is None):
        raise ValueError(f"{path}: A named TCP requires flange_to_tcp; flange requires null flange_to_tcp")
