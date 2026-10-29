"""Episode-safe G2 temporal sampling over a LeRobot-shaped dataset interface."""

from functools import cache, partial

import numpy as np

from . import geometry as g2
from .schema import action_branch


def label_column(policy):
    kind, spec = action_branch(policy)
    suffix = {"pose": "pose", "position": "joint", "velocity": "joint_velocity", "torque": "joint_torque"}[spec["quantity"]]
    return f"{spec['source']}.{suffix}"


class G2Dataset:
    """Sample in seconds within episodes; clamp boundary targets, repeat initial history.

    Low-dimensional signals interpolate linearly, poses interpolate on SO(3).
    Images use the nearest recorded frame. No sample crosses an episode boundary.
    Histories and derived reference fields are generated only when the policy needs them.
    """

    def __init__(self, dataset, policy, cameras, horizon, *, action_indices=None):
        self.dataset, self.policy, self.cameras = dataset, policy, cameras
        self.fps = float(dataset.meta.fps)
        self.hz = float(policy["timing"]["sample_hz"])
        self.action_indices = np.arange(horizon) if action_indices is None else np.asarray(action_indices)
        state, action = policy["state"], policy["action"]
        self.state_kinds = tuple(kind for kind in ("pose", "joint") if state[kind]["enabled"])
        self.current_signals = []
        self.reference_signals = []
        for kind in ("pose", "joint"):
            active_state = state[kind]["enabled"]
            active_action = action[kind]["enabled"]
            if not (active_state or active_action):
                continue
            self.current_signals.append(f"state.{kind}")
            specs = ([state[kind]] if active_state else []) + ([action[kind]] if active_action else [])
            if any(spec["delta"] and spec["reference"]["source"] == "action" for spec in specs):
                self.reference_signals.append((f"g2.command_{kind}", f"action.{kind}"))
        if any(
            spec["enabled"] and spec["coordinates"]["base_to_parent"] == "observation"
            for spec in (state["pose"], action["pose"])
        ):
            self.current_signals.append("reference.base_to_parent")
        self.anchors = []
        self.bounds = []
        for first, last in zip(
            dataset.episode_data_index["from"],
            dataset.episode_data_index["to"],
            strict=True,
        ):
            start, stop = int(first), int(last)
            count = int(np.floor((stop - start - 1) * self.hz / self.fps + 1e-7)) + 1
            for i in range(count):
                self.anchors.append(start + i * self.fps / self.hz)
                self.bounds.append((start, stop))

    def __len__(self):
        return len(self.anchors)

    @staticmethod
    def _signal(key, positions, bounds, row):
        start, stop = bounds
        samples = []
        for position in positions:
            x = float(np.clip(position, start, stop - 1))
            low, high = int(np.floor(x)), int(np.ceil(x))
            a = np.asarray(row(low)[key], dtype=np.float32)
            if low == high:
                samples.append(a.copy())
                continue
            b = np.asarray(row(high)[key], dtype=np.float32)
            f = x - low
            value = a * (1 - f) + b * f
            if key in ("state.pose", "action.pose"):
                from scipy.spatial.transform import Rotation, Slerp

                for offset in (0, 10):
                    rotations = g2.rotation_matrix(np.stack([a[offset + 3 : offset + 9], b[offset + 3 : offset + 9]]))
                    value[offset + 3 : offset + 9] = g2.rotation_vector(
                        Slerp([0, 1], Rotation.from_matrix(rotations))([f]).as_matrix()[0]
                    )
            if key == "reference.base_to_parent":
                from scipy.spatial.transform import Rotation, Slerp

                value[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))([f]).as_matrix()[
                    0
                ]
            samples.append(value)
        return np.asarray(samples)

    def __getitem__(self, index):
        return self.sample(index)

    def sample(self, index, *, images=True):
        """Sample vectors without decoding videos when computing normalization stats."""
        anchor, bounds = self.anchors[index], self.bounds[index]
        timing, action = self.policy["timing"], self.policy["action"]
        # A sample owns its caches: no decoded images or mutable rows survive it.
        @cache
        def row(position):
            return self.dataset.hf_dataset[position]

        @cache
        def frame(position):
            return self.dataset[position]

        signal = partial(self._signal, bounds=bounds, row=row)
        current = dict(frame(round(anchor)) if images else row(round(anchor)))
        fields = self.dataset.meta.features
        for key in self.current_signals:
            if key in fields:
                current[key] = signal(key, [anchor])[0]
        history = anchor + np.arange(1 - timing["history_steps"], 1) * self.fps / self.hz
        for kind in self.state_kinds:
            key = f"state.{kind}"
            if key in fields:
                current[f"g2.{kind}_history"] = signal(key, history)
        for dest, key in self.reference_signals:
            if key in fields:
                current[dest] = signal(key, [anchor])[0]
        if images and timing["history_steps"] > 1:
            frames = [frame(int(np.clip(round(x), bounds[0], bounds[1] - 1))) for x in history]
            for camera in self.cameras:
                key = f"observation.images.{camera}"
                current[key] = np.stack([np.asarray(item[key]) for item in frames])
        targets = (
            anchor
            + (timing["target_offset_steps"] + self.action_indices * timing["stride_steps"]) * self.fps / self.hz
        )
        column = label_column(self.policy)
        current["g2.action_is_pad"] = (targets < bounds[0]) | (targets > bounds[1] - 1)
        if column in fields:
            labels = signal(column, targets)
        else:
            # Joint velocity can be derived from positions; torque must be recorded.
            if action_branch(self.policy)[1]["quantity"] != "velocity":
                raise ValueError(f"Dataset requires recorded {column}")
            source = f"{action_branch(self.policy)[1]['source']}.joint"
            joints = signal(source, targets)
            next_targets = np.minimum(targets + self.fps / self.hz, bounds[1] - 1)
            previous = np.maximum(np.minimum(targets, bounds[1] - 1) - self.fps / self.hz, bounds[0])
            # Forward differences, backward at the endpoint; a one-frame episode has zero velocity.
            begins = np.where(
                next_targets > np.minimum(targets, bounds[1] - 1),
                np.minimum(targets, bounds[1] - 1),
                previous,
            )
            ends = np.where(
                next_targets > np.minimum(targets, bounds[1] - 1),
                next_targets,
                np.minimum(targets, bounds[1] - 1),
            )
            a, b = (
                signal(source, begins),
                signal(source, ends),
            )
            dt = np.maximum((ends - begins) / self.fps, 1e-8)
            labels = (b - a) / dt[:, None]
            labels[:, [7, 15]] = joints[:, [7, 15]]
        current["g2.actions"] = labels.astype(np.float32)
        return current
