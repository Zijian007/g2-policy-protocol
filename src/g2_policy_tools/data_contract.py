"""Validate disk semantics at framework entry points; never import a dataset library."""

import math

from .dataset import label_column
from .schema import action_branch, control_quantity


def validate_dataset_contract(features, info, policy, cameras):
    state, action = policy["state"], policy["action"]
    required = set(cameras)
    if state["pose"]["enabled"] or action["pose"]["enabled"]:
        required.add("state.pose")
    if state["joint"]["enabled"] or action["joint"]["enabled"]:
        required.add("state.joint")
    column = label_column(policy)
    kind, spec = action_branch(policy)
    if column not in features and control_quantity(policy) == "joint_velocity":
        required.add(f"{spec['source']}.joint")
    else:
        required.add(column)
    for group in ("state", "action"):
        for branch, disk in (("pose", "action.pose"), ("joint", "action.joint")):
            item = policy[group][branch]
            if item["enabled"] and item["delta"] and item["reference"]["source"] == "action":
                required.add(disk)
    if any(
        item["enabled"] and item["coordinates"]["base_to_parent"] == "observation"
        for item in (state["pose"], action["pose"])
    ):
        required.add("reference.base_to_parent")
    missing = required - features.keys()
    if missing:
        raise ValueError(f"G2 dataset is missing required columns: {sorted(missing)}")
    dimensions = {
        "state.pose": 20,
        "action.pose": 20,
        "state.joint": 16,
        "action.joint": 16,
        "state.joint_velocity": 16,
        "action.joint_velocity": 16,
        "state.joint_torque": 16,
        "action.joint_torque": 16,
    }
    for key in required & dimensions.keys():
        if "shape" in features[key] and tuple(features[key]["shape"]) != (
            dimensions[key],
        ):
            raise ValueError(f"G2 dataset {key} must have shape [{dimensions[key]}]")
    if "fps" in info:
        fps = info["fps"]
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Dataset fps must be a positive finite number")
    exported = info.get("wholebody_export", {})
    for field, expected in (("ee_frame", "flange"), ("rotation", "rot6d")):
        if field in exported and exported[field] != expected:
            raise ValueError(
                f"Dataset wholebody_export.{field}={exported[field]!r} must match {expected!r}"
            )
    pose_frame = exported.get("pose_frame", "base_link")
    action_parent = exported.get("action_parent_frame", pose_frame)
    state_needs_parent = state["pose"]["enabled"] and state["pose"]["coordinates"]["base_to_parent"] is None
    action_needs_parent = action["pose"]["enabled"] and action["pose"]["coordinates"]["base_to_parent"] is None
    if pose_frame != "base_link" and (state_needs_parent or action_needs_parent):
        raise ValueError(
            f"Dataset wholebody_export.pose_frame={pose_frame!r} must supply coordinates.base_to_parent"
        )
    if pose_frame != action_parent and action_needs_parent:
        raise ValueError(
            "Dataset pose_frame and action_parent_frame differ; the encoded parent must have coordinates.base_to_parent"
        )
    if spec["source"] == "action" and exported.get("action_supervision", "command") != "command":
        raise ValueError(
            "Dataset wholebody_export.action_supervision must match action labels"
        )
    if exported.get("segment_policy") == "concatenated":
        raise ValueError(
            "Dataset concatenates capture segments; export separate continuous episodes"
        )
