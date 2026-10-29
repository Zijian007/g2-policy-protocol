"""Derive wire requirements from a compiled robot policy, without model imports."""

from g2_policy_protocol import protocol

from .schema import control_quantity


def policy_capabilities(policy, cameras, *, prompt, reset):
    state, action, timing = policy["state"], policy["action"], policy["timing"]
    pose = state["pose"]["enabled"] or action["pose"]["enabled"]
    joint = state["joint"]["enabled"] or action["joint"]["enabled"]
    context = []
    for kind in ("pose", "joint"):
        if any(
            spec["enabled"] and spec["delta"] and spec["reference"]["source"] == "action"
            for spec in (state[kind], action[kind])
        ):
            context.append(f"command_{kind}")
    if any(
        spec["enabled"] and spec["coordinates"]["base_to_parent"] == "observation"
        for spec in (state["pose"], action["pose"])
    ):
        context.append("base_to_parent")
    if timing["history_steps"] > 1:
        context.extend((["pose_history"] if pose else []) + (["joint_history"] if joint else []))
        if cameras:
            context.append("image_history")
    dt = timing["stride_steps"] / timing["sample_hz"]
    return {
        "version": 3,
        "policy_hz": 1 / dt,
        "requires_joint_state": joint,
        "reset_supported": reset,
        "action_spec": protocol.v3.action_spec(control_quantity(policy)),
        "observation_spec": {
            "pose": pose,
            "joint": joint,
            "prompt": prompt,
            "history_steps": timing["history_steps"],
            "history_hz": timing["sample_hz"],
            "context_keys": list(dict.fromkeys(context)),
        },
        "timing": {
            "action_dt": dt,
            "first_target_offset_s": timing["target_offset_steps"] / timing["sample_hz"],
        },
    }
