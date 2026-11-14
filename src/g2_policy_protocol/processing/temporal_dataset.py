"""Episode-safe temporal sampling and bounded normalization reads."""

import copy
from collections import OrderedDict
from collections.abc import Sequence
from functools import cache, partial
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from g2_policy_protocol.contracts.dataset import action_label_column
from g2_policy_protocol.contracts.policy import (
    KINDS,
    PolicySemantics,
    command_reference_kinds,
    uses_arm_base,
    uses_kind,
)
from g2_policy_protocol.types import POSE_ARM_DIM

from .geometry import (
    ROTATION_WIDTHS,
    matrix_to_rotation,
    rotation_to_matrix,
    xyz_rot6d_to_matrix,
)

# Samples retain arbitrary source columns, camera keys and task text.
DatasetSample = dict[str, Any]

_ROT6D_WIDTH = ROTATION_WIDTHS["rot6d"]


class _NormalizationRows:
    """Two bounded blocks of numerical columns, local to each statistics worker."""

    def __init__(self, rows, columns, block_size=4096):
        self.rows = rows.select_columns(columns).with_format("numpy")
        self.block_size = block_size
        self._blocks = OrderedDict()

    def __getitem__(self, index):
        start = index // self.block_size * self.block_size
        if start not in self._blocks:
            self._blocks[start] = self.rows[start : start + self.block_size]
            if len(self._blocks) > 2:
                self._blocks.popitem(last=False)
        self._blocks.move_to_end(start)
        # Samples must own their arrays; callers cannot mutate the cached blocks.
        return {key: np.asarray(values[index - start]).copy() for key, values in self._blocks[start].items()}

    def __getstate__(self):
        # Spawned workers fill their own cache without serializing cached arrays.
        return {**self.__dict__, "_blocks": OrderedDict()}


class TemporalDataset:
    """Sample in seconds within episodes; clamp boundary targets, repeat initial history.

    Low-dimensional signals interpolate linearly, poses interpolate on SO(3).
    Images use the nearest recorded frame. No sample crosses an episode boundary.
    Histories and derived reference fields are generated only when the policy needs them.
    A separate normalization view can cache numerical columns without retaining images.
    """

    def __init__(
        self,
        dataset: Any,
        policy: PolicySemantics,
        cameras: Sequence[str],
        horizon: int,
        *,
        action_indices: Sequence[int] | None = None,
    ) -> None:
        self.dataset, self.policy, self.cameras = dataset, policy, cameras
        self.source_fps = float(dataset.meta.fps)
        self.sample_hz = float(policy["timing"]["sample_hz"])
        self.action_indices = np.arange(horizon) if action_indices is None else np.asarray(action_indices)
        self.state_modalities = tuple(kind for kind in KINDS if policy["state"][kind]["enabled"])
        self.state_columns = [f"state.{kind}" for kind in KINDS if uses_kind(policy, kind)]
        self.reference_columns = [(f"g2.command_{kind}", f"action.{kind}") for kind in command_reference_kinds(policy)]
        self._normalization_rows = None
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

    def __len__(self) -> int:
        return len(self.anchor_frames)

    def normalization_dataset(self) -> "TemporalDataset":
        """Share temporal indexing, but batch/cache low-dimensional HF reads for stats."""
        if not hasattr(self.dataset.hf_dataset, "select_columns"):
            return self
        columns = set(self.state_columns) | {action_label_column(self.policy)}
        columns.update(key for _, key in self.reference_columns)
        result = copy.copy(self)
        result._normalization_rows = _NormalizationRows(
            self.dataset.hf_dataset, sorted(columns & self.dataset.meta.features.keys())
        )
        return result

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

    def __getitem__(self, index: int) -> DatasetSample:
        return self.sample(index)

    def sample(self, index: int, *, images: bool = True) -> DatasetSample:
        """Sample vectors without decoding videos when computing normalization stats."""
        anchor_frame, episode_bounds = self.anchor_frames[index], self.episode_bounds[index]
        timing = self.policy["timing"]

        # Training retains no rows across samples. The normalization view shares
        # only bounded numerical blocks, never decoded images or task text.
        @cache
        def read_row(frame_position):
            if not images and self._normalization_rows is not None:
                return self._normalization_rows[frame_position]
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
