"""G2 dataset validation, temporal sampling and sample references."""

import math
from functools import cache, partial

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import encoding
from .policy import action_branch, control_quantity


def action_label_column(policy):
    kind, spec = action_branch(policy)
    suffix = {"pose": "pose", "position": "joint", "velocity": "joint_velocity", "torque": "joint_torque"}[
        spec["quantity"]
    ]
    return f"{spec['source']}.{suffix}"


def validate_dataset_contract(features, info, policy, cameras):
    state, action = policy["state"], policy["action"]
    required = set(cameras)
    if state["pose"]["enabled"] or action["pose"]["enabled"]:
        required.add("state.pose")
    if state["joint"]["enabled"] or action["joint"]["enabled"]:
        required.add("state.joint")
    column = action_label_column(policy)
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
        item["enabled"] and item["coordinates"]["parent_frame"] == "arm_base_link"
        for item in (state["pose"], action["pose"])
    ):
        required.add("state.arm_base")
    missing = required - features.keys()
    if missing:
        raise ValueError(f"G2 dataset is missing required columns: {sorted(missing)}")
    dimensions = {
        "state.arm_base": 9,
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
        if "shape" in features[key] and tuple(features[key]["shape"]) != (dimensions[key],):
            raise ValueError(f"G2 dataset {key} must have shape [{dimensions[key]}]")
    if "fps" in info:
        fps = info["fps"]
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Dataset fps must be a positive finite number")
    exported = info.get("wholebody_export", {})
    for field, expected in (("ee_frame", "flange"), ("rotation", "rot6d")):
        if field in exported and exported[field] != expected:
            raise ValueError(f"Dataset wholebody_export.{field}={exported[field]!r} must match {expected!r}")
    pose_frame = exported.get("pose_frame", "base_link")
    action_parent = exported.get("action_parent_frame", pose_frame)
    uses_pose = state["pose"]["enabled"] or action["pose"]["enabled"]
    if uses_pose and pose_frame != "base_link":
        raise ValueError(f"Dataset wholebody_export.pose_frame={pose_frame!r} must be 'base_link'")
    needs_command_pose = any(
        item["enabled"] and item["delta"] and item["reference"]["source"] == "action"
        for item in (state["pose"], action["pose"])
    ) or (action["pose"]["enabled"] and action["pose"]["source"] == "action")
    if needs_command_pose and action_parent != "base_link":
        raise ValueError(f"Dataset wholebody_export.action_parent_frame={action_parent!r} must be 'base_link'")
    if spec["source"] == "action" and exported.get("action_supervision", "command") != "command":
        raise ValueError("Dataset wholebody_export.action_supervision must match action labels")
    if exported.get("segment_policy") == "concatenated":
        raise ValueError("Dataset concatenates capture segments; export separate continuous episodes")


def references_from_sample(sample):
    """Map disk/sampler references to the shared geometry dictionary."""
    sources = {
        "state.pose": "pose",
        "state.joint": "joint",
        "g2.pose_history": "pose_history",
        "g2.joint_history": "joint_history",
        "g2.command_pose": "command_pose",
        "g2.command_joint": "command_joint",
        "reference.base_to_parent": "base_to_parent",
    }
    return {
        f"observation/{target}": np.asarray(sample[source]) for source, target in sources.items() if source in sample
    }


class TemporalDataset:
    """Sample in seconds within episodes; clamp boundary targets, repeat initial history.

    Low-dimensional signals interpolate linearly, poses interpolate on SO(3).
    Images use the nearest recorded frame. No sample crosses an episode boundary.
    Histories and derived reference fields are generated only when the policy needs them.
    """

    def __init__(self, dataset, policy, cameras, horizon, *, action_indices=None):
        self.dataset, self.policy, self.cameras = dataset, policy, cameras
        self.source_fps = float(dataset.meta.fps)
        self.sample_hz = float(policy["timing"]["sample_hz"])
        self.action_indices = np.arange(horizon) if action_indices is None else np.asarray(action_indices)
        state, action = policy["state"], policy["action"]
        self.state_modalities = tuple(kind for kind in ("pose", "joint") if state[kind]["enabled"])
        self.state_columns = []
        self.reference_columns = []
        for kind in ("pose", "joint"):
            state_enabled = state[kind]["enabled"]
            action_enabled = action[kind]["enabled"]
            if not (state_enabled or action_enabled):
                continue
            self.state_columns.append(f"state.{kind}")
            active_specs = ([state[kind]] if state_enabled else []) + ([action[kind]] if action_enabled else [])
            if any(spec["delta"] and spec["reference"]["source"] == "action" for spec in active_specs):
                self.reference_columns.append((f"g2.command_{kind}", f"action.{kind}"))
        if any(
            spec["enabled"] and spec["coordinates"]["parent_frame"] == "arm_base_link"
            for spec in (state["pose"], action["pose"])
        ):
            self.state_columns.append("state.arm_base")
        self.anchor_frames = []
        self.episode_bounds = []
        for first, last in zip(
            dataset.episode_data_index["from"],
            dataset.episode_data_index["to"],
            strict=True,
        ):
            start, stop = int(first), int(last)
            count = int(np.floor((stop - start - 1) * self.sample_hz / self.source_fps + 1e-7)) + 1
            for i in range(count):
                self.anchor_frames.append(start + i * self.source_fps / self.sample_hz)
                self.episode_bounds.append((start, stop))

    def __len__(self):
        return len(self.anchor_frames)

    @staticmethod
    def _sample_signal(key, frame_positions, episode_bounds, read_row):
        start, stop = episode_bounds
        samples = []
        for frame_position in frame_positions:
            clamped_frame = float(np.clip(frame_position, start, stop - 1))
            lower_frame, upper_frame = int(np.floor(clamped_frame)), int(np.ceil(clamped_frame))
            start_values = np.asarray(read_row(lower_frame)[key], dtype=np.float32)
            if lower_frame == upper_frame:
                samples.append(start_values.copy())
                continue
            end_values = np.asarray(read_row(upper_frame)[key], dtype=np.float32)
            fraction = clamped_frame - lower_frame
            value = start_values * (1 - fraction) + end_values * fraction
            if key in ("state.pose", "action.pose", "state.arm_base"):
                for offset in (0,) if key == "state.arm_base" else (0, 10):
                    rotations = encoding.rotation_matrix(
                        np.stack([start_values[offset + 3 : offset + 9], end_values[offset + 3 : offset + 9]])
                    )
                    value[offset + 3 : offset + 9] = encoding.rotation_vector(
                        Slerp([0, 1], Rotation.from_matrix(rotations))([fraction]).as_matrix()[0]
                    )
            samples.append(value)
        return np.asarray(samples)

    def __getitem__(self, index):
        return self.sample(index)

    def sample(self, index, *, images=True):
        """Sample vectors without decoding videos when computing normalization stats."""
        anchor_frame, episode_bounds = self.anchor_frames[index], self.episode_bounds[index]
        timing = self.policy["timing"]

        # A sample owns its caches: no decoded images or mutable rows survive it.
        @cache
        def read_row(frame_position):
            return self.dataset.hf_dataset[frame_position]

        @cache
        def read_frame(frame_position):
            return self.dataset[frame_position]

        signal = partial(self._sample_signal, episode_bounds=episode_bounds, read_row=read_row)
        sample = dict(read_frame(round(anchor_frame)) if images else read_row(round(anchor_frame)))
        features = self.dataset.meta.features
        for key in self.state_columns:
            if key in features:
                sample[key] = signal(key, [anchor_frame])[0]
        if "state.arm_base" in self.state_columns:
            arm_base = np.append(sample["state.arm_base"], 0)
            sample["reference.base_to_parent"] = np.linalg.inv(encoding.pose_matrix(arm_base))
        history = anchor_frame + np.arange(1 - timing["history_steps"], 1) * self.source_fps / self.sample_hz
        for kind in self.state_modalities:
            key = f"state.{kind}"
            if key in features:
                sample[f"g2.{kind}_history"] = signal(key, history)
        for reference_key, key in self.reference_columns:
            if key in features:
                sample[reference_key] = signal(key, [anchor_frame])[0]
        if images and timing["history_steps"] > 1:
            frames = [
                read_frame(int(np.clip(round(clamped_frame), episode_bounds[0], episode_bounds[1] - 1)))
                for clamped_frame in history
            ]
            for camera in self.cameras:
                key = f"observation.images.{camera}"
                sample[key] = np.stack([np.asarray(item[key]) for item in frames])
        target_frames = (
            anchor_frame
            + (timing["target_offset_steps"] + self.action_indices * timing["stride_steps"])
            * self.source_fps
            / self.sample_hz
        )
        column = action_label_column(self.policy)
        sample["g2.action_is_pad"] = (target_frames < episode_bounds[0]) | (target_frames > episode_bounds[1] - 1)
        if column in features:
            labels = signal(column, target_frames)
        else:
            # Joint velocity can be derived from positions; torque must be recorded.
            if action_branch(self.policy)[1]["quantity"] != "velocity":
                raise ValueError(f"Dataset requires recorded {column}")
            source = f"{action_branch(self.policy)[1]['source']}.joint"
            joints = signal(source, target_frames)
            next_frames = np.minimum(target_frames + self.source_fps / self.sample_hz, episode_bounds[1] - 1)
            previous_frames = np.maximum(
                np.minimum(target_frames, episode_bounds[1] - 1) - self.source_fps / self.sample_hz, episode_bounds[0]
            )
            # Forward differences, backward at the endpoint; a one-frame episode has zero velocity.
            start_frames = np.where(
                next_frames > np.minimum(target_frames, episode_bounds[1] - 1),
                np.minimum(target_frames, episode_bounds[1] - 1),
                previous_frames,
            )
            end_frames = np.where(
                next_frames > np.minimum(target_frames, episode_bounds[1] - 1),
                next_frames,
                np.minimum(target_frames, episode_bounds[1] - 1),
            )
            start_values, end_values = (
                signal(source, start_frames),
                signal(source, end_frames),
            )
            dt = np.maximum((end_frames - start_frames) / self.source_fps, 1e-8)
            labels = (end_values - start_values) / dt[:, None]
            labels[:, [7, 15]] = joints[:, [7, 15]]
        sample["g2.actions"] = labels.astype(np.float32)
        return sample
