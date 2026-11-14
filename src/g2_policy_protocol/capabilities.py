"""Translate resolved policy semantics into wire capability declarations."""

from collections.abc import Sequence

from . import protocol
from .contracts.policy import (
    PolicySemantics,
    command_reference_kinds,
    control_quantity,
    uses_arm_base,
    uses_kind,
)
from .types import PolicyCapabilities


def policy_capabilities(
    policy: PolicySemantics,
    cameras: Sequence[str],
    *,
    prompt: bool,
    reset: bool,
) -> PolicyCapabilities:
    """Derive build_metadata arguments (observation needs, timing, action layout) from a policy."""
    timing = policy["timing"]
    needs_pose, needs_joint = uses_kind(policy, "pose"), uses_kind(policy, "joint")
    context = [f"command_{kind}" for kind in command_reference_kinds(policy)]
    if uses_arm_base(policy):
        context.append("base_to_parent")
    if timing["history_steps"] > 1:
        context.extend((["pose_history"] if needs_pose else []) + (["joint_history"] if needs_joint else []))
        if cameras:
            context.append("image_history")
    action_interval_s = timing["stride_steps"] / timing["sample_hz"]
    return {
        "policy_hz": 1 / action_interval_s,
        "reset_supported": reset,
        "action_spec": protocol.build_action_spec(control_quantity(policy)),
        "observation_spec": {
            "pose": needs_pose,
            "joint": needs_joint,
            "prompt": prompt,
            "history_steps": timing["history_steps"],
            "history_hz": timing["sample_hz"],
            "context_keys": context,
        },
        "timing": {
            "action_dt": action_interval_s,
            "first_target_offset_s": timing["target_offset_steps"] / timing["sample_hz"],
        },
    }
