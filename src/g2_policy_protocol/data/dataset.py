"""G2 dataset validation, temporal sampling and sample references."""

import math
from functools import cache, partial

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .._constants import JOINT_DIM, POSE_ARM_DIM, POSE_DIM
from .geometry import ROTATION_WIDTHS, matrix_to_rotation, rotation_to_matrix, xyz_rot6d_to_matrix
from .policy import KINDS, active_action, command_reference_kinds, uses_arm_base, uses_kind

_ROT6D_WIDTH = ROTATION_WIDTHS["rot6d"]
# state.arm_base: pose of arm_base_link in base_link, xyz + rot6d (no gripper).
_ARM_BASE_DIM = 3 + _ROT6D_WIDTH
_COLUMN_WIDTHS = {
    "state.arm_base": _ARM_BASE_DIM,
    "state.pose": POSE_DIM,
    "action.pose": POSE_DIM,
    "state.joint": JOINT_DIM,
    "action.joint": JOINT_DIM,
}


def action_label_column(policy):
    """Disk column that holds the training action labels."""
    kind, branch = active_action(policy)
    return f"{branch['source']}.{kind}"


def validate_dataset_contract(features, info, policy, cameras):
    """Check that a LeRobot dataset records everything the policy needs, in the expected frames."""
    action = policy["action"]
    _, label_branch = active_action(policy)
    required = {*cameras, action_label_column(policy)}
    required |= {f"state.{kind}" for kind in KINDS if uses_kind(policy, kind)}
    required |= {f"action.{kind}" for kind in command_reference_kinds(policy)}
    if uses_arm_base(policy):
        required.add("state.arm_base")
    missing = required - features.keys()
    if missing:
        raise ValueError(f"G2 dataset is missing required columns: {sorted(missing)}")
    for key in required & _COLUMN_WIDTHS.keys():
        if "shape" in features[key] and tuple(features[key]["shape"]) != (_COLUMN_WIDTHS[key],):
            raise ValueError(f"G2 dataset {key} must have shape [{_COLUMN_WIDTHS[key]}]")
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
    if uses_kind(policy, "pose") and pose_frame != "base_link":
        raise ValueError(f"Dataset wholebody_export.pose_frame={pose_frame!r} must be 'base_link'")
    reads_command_pose = "pose" in command_reference_kinds(policy) or (
        action["pose"]["enabled"] and action["pose"]["source"] == "action"
    )
    if reads_command_pose and action_parent != "base_link":
        raise ValueError(f"Dataset wholebody_export.action_parent_frame={action_parent!r} must be 'base_link'")
    if label_branch["source"] == "action" and exported.get("action_supervision", "command") != "command":
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
        "g2.base_to_parent": "base_to_parent",
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
        self.state_modalities = tuple(kind for kind in KINDS if policy["state"][kind]["enabled"])
        self.state_columns = [f"state.{kind}" for kind in KINDS if uses_kind(policy, kind)]
        self.reference_columns = [(f"g2.command_{kind}", f"action.{kind}") for kind in command_reference_kinds(policy)]
        if uses_arm_base(policy):
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
                # Rotations interpolate on SO(3), not component-wise.
                for offset in (0,) if key == "state.arm_base" else (0, POSE_ARM_DIM):
                    rot6d = slice(offset + 3, offset + 3 + _ROT6D_WIDTH)
                    rotations = rotation_to_matrix(np.stack([start_values[rot6d], end_values[rot6d]]))
                    value[rot6d] = matrix_to_rotation(
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
            # base_to_parent is T_armbase←base: the inverse of the recorded arm base pose.
            sample["g2.base_to_parent"] = np.linalg.inv(xyz_rot6d_to_matrix(sample["state.arm_base"]))
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
        if column not in features:
            raise ValueError(f"Dataset requires recorded {column}")
        labels = signal(column, target_frames)
        sample["g2.actions"] = labels.astype(np.float32)
        return sample
