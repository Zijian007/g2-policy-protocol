"""Shared G2 state/action schema: filled public YAML is the runtime contract."""

import copy
import math

from .validation import validate_rigid_transform

SOURCES = ("state", "action")
METHODS = ("component", "se3")
QUANTITIES = {
    "joint": {"position": "joint_position", "velocity": "joint_velocity", "torque": "joint_torque"},
    "pose": {"pose": "pose_target"},
}
COORDINATES = {
    "ee_frame": "flange",
    "rotation": "rot6d",
    "base_to_parent": None,
    "flange_to_tcp": None,
}


def branch_defaults(group, kind):
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


def action_branch(policy):
    kinds = [kind for kind, spec in policy["action"].items() if spec["enabled"]]
    return kinds[0], policy["action"][kinds[0]]


def control_quantity(policy):
    kind, spec = action_branch(policy)
    return QUANTITIES[kind][spec["quantity"]]


def resolve_policy(raw, *, horizon, sample_hz=30.0):
    """Resolve a partial robot policy once at the configuration boundary.

    The model owns its prediction horizon. A copied public action_horizon must match it.
    """
    if "action_horizon" in raw.get("timing", {}):
        declared = raw["timing"]["action_horizon"]
        if type(declared) is not int or declared != horizon:
            raise ValueError(
                "policy.timing.action_horizon must match the model prediction horizon; adapters cannot override it"
            )
    policy = {
        group: {kind: branch_defaults(group, kind) for kind in ("joint", "pose")} for group in ("state", "action")
    }
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
    return compile_policy(policy)


def validate_coordinates(pose, path="coordinates"):
    for key in ("ee_frame", "rotation"):
        if not isinstance(pose[key], str) or not pose[key].strip():
            raise ValueError(f"{path}.{key} must be a nonempty string")
    if pose["rotation"] not in ("rot6d", "quaternion", "axis_angle"):
        raise ValueError(f"{path}.rotation must be rot6d, quaternion or axis_angle")
    parent = pose["base_to_parent"]
    if parent is not None and not (isinstance(parent, str) and parent == "observation"):
        validate_rigid_transform(parent, f"{path}.base_to_parent")
    tcp = pose["flange_to_tcp"]
    if tcp is not None:
        if not isinstance(tcp, dict) or tcp.keys() != {"left", "right"}:
            raise ValueError(f"{path}.flange_to_tcp must contain left and right transforms")
        for side, transform in tcp.items():
            validate_rigid_transform(transform, f"{path}.flange_to_tcp.{side}")
    if (pose["ee_frame"] == "flange") != (tcp is None):
        raise ValueError(f"{path}: A named TCP requires flange_to_tcp; flange requires null flange_to_tcp")


def validate_policy(policy):
    if not isinstance(policy, dict) or policy.keys() != {"state", "action", "timing"}:
        raise ValueError("policy fields must be state, action and timing")
    active = {}
    for group in ("state", "action"):
        branches = policy[group]
        if not isinstance(branches, dict) or branches.keys() != {"joint", "pose"}:
            raise ValueError(f"policy.{group} fields must be joint and pose")
        for kind, spec in branches.items():
            _validate_branch(spec, group, kind)
            _validate_encoding(spec, group, kind)
        active[group] = [kind for kind, spec in branches.items() if spec["enabled"]]
        if not active[group] or (group == "action" and len(active[group]) != 1):
            raise ValueError(
                f"policy.{group} requires {'exactly one' if group == 'action' else 'at least one'} active branch"
            )
    for group in ("state", "action"):
        spec = policy[group]["pose"]
        if spec["enabled"]:
            validate_coordinates(spec["coordinates"], f"policy.{group}.pose.coordinates")
    timing = policy["timing"]
    for key in ("history_steps", "action_horizon", "target_offset_steps", "stride_steps"):
        value = timing[key]
        if type(value) is not int or value < (0 if key == "target_offset_steps" else 1):
            raise ValueError(f"policy.timing.{key} must be a valid integer step count")
    hz = timing["sample_hz"]
    if type(hz) not in (int, float) or not math.isfinite(hz) or hz <= 0:
        raise ValueError("policy.timing.sample_hz must be a positive finite number")
    if timing["history_steps"] == 1:
        for kind, spec in policy["state"].items():
            if spec["enabled"] and spec["delta"] and spec["reference"]["source"] == "state":
                raise ValueError(f"policy.state.{kind} relative to source: state requires history_steps > 1")
    kind, spec = action_branch(policy)
    quantity = _choice(spec["quantity"], QUANTITIES[kind], f"policy.action.{kind}.quantity")
    if kind == "joint" and quantity in ("joint_velocity", "joint_torque") and spec["delta"]:
        raise ValueError(f"policy.action.{kind}: joint velocity and torque require delta: false")


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
    if path in ("policy.state.joint", "policy.state.pose", "policy.action.joint", "policy.action.pose"):
        if override.get("delta") is False:
            if "reference" not in override:
                base["reference"] = dict.fromkeys(base["reference"])
            if "method" not in override:
                base["method"] = None


def compile_policy(policy):
    if not isinstance(policy, dict) or policy.keys() != {"state", "action", "timing"}:
        raise ValueError("policy fields must be state, action and timing")
    for group in ("state", "action"):
        spec = policy.get(group, {}).get("pose")
        if isinstance(spec, dict) and isinstance(spec.get("coordinates"), dict):
            for key, default in COORDINATES.items():
                spec["coordinates"].setdefault(key, copy.deepcopy(default))
    validate_policy(policy)
    return policy


def public_settings(settings):
    return copy.deepcopy(settings)


def _choice(value, choices, path):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{path} must be one of {tuple(choices)}")
    return choices[value] if isinstance(choices, dict) else value


def _validate_branch(spec, group, kind):
    path = f"policy.{group}.{kind}"
    expected = branch_defaults(group, kind)
    if not isinstance(spec, dict) or spec.keys() != expected.keys():
        raise ValueError(f"{path} fields must be {sorted(expected)}")
    if type(spec["enabled"]) is not bool:
        raise ValueError(f"{path}.enabled must be a YAML boolean")
    if type(spec["delta"]) is not bool:
        raise ValueError(f"{path}.delta must be a YAML boolean")
    if not isinstance(spec["reference"], dict) or spec["reference"].keys() != expected["reference"].keys():
        raise ValueError(f"{path}.reference fields must be {sorted(expected['reference'])}")
    if kind == "pose" and (
        not isinstance(spec["coordinates"], dict) or spec["coordinates"].keys() != COORDINATES.keys()
    ):
        raise ValueError(f"{path}.coordinates fields must be {sorted(COORDINATES)}")


def _validate_encoding(spec, group, kind):
    path = f"policy.{group}.{kind}"
    if group == "action":
        _choice(spec["source"], SOURCES, f"{path}.source")
    reference = spec["reference"]
    if not spec["delta"]:
        if spec["method"] is not None:
            raise ValueError(f"{path}: method requires delta: true")
        if any(value is not None for value in reference.values()):
            raise ValueError(f"{path}: reference requires delta: true")
        return
    if kind == "joint":
        if spec["method"] != "component":
            raise ValueError(f"{path}.method must be component for joint deltas")
    elif spec["method"] not in METHODS:
        raise ValueError(f"{path}.method must be component or se3")
    _choice(reference["source"], SOURCES, f"{path}.reference.source")
    _choice(reference["scope"], ("chunk", "step"), f"{path}.reference.scope")
