"""Episode-safe G2 temporal sampling over a LeRobot-shaped dataset interface."""

import numpy as np

from . import geometry as g2


def label_column(policy):
    action = policy["action"]
    prefix = "action" if action["label_source"] == "command" else "state"
    suffix = {
        "pose_target": "pose",
        "joint_position": "joint",
        "joint_velocity": "joint_velocity",
        "eef_twist": "eef_twist",
        "joint_torque": "joint_torque",
    }[action["control_quantity"]]
    return f"{prefix}.{suffix}"


class G2Dataset:
    """Sample in seconds within episodes; clamp boundary targets, repeat initial history.

    Low-dimensional signals interpolate linearly, poses interpolate on SO(3).
    Images use the nearest recorded frame. No sample crosses an episode boundary.
    """

    def __init__(self, dataset, policy, cameras, horizon):
        self.dataset, self.policy, self.cameras, self.horizon = (
            dataset,
            policy,
            cameras,
            horizon,
        )
        self.fps = float(dataset.meta.fps)
        self.hz = float(policy["timing"]["sample_hz"])
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

    def _row(self, index):
        return self.dataset.hf_dataset[int(index)]

    def _signal(self, key, positions, bounds):
        start, stop = bounds
        samples = []
        for position in positions:
            x = float(np.clip(position, start, stop - 1))
            low, high = int(np.floor(x)), int(np.ceil(x))
            a = np.asarray(self._row(low)[key], dtype=np.float32)
            b = np.asarray(self._row(high)[key], dtype=np.float32)
            f = x - low
            value = a * (1 - f) + b * f
            if key in ("state.pose", "action.pose", "reference.object_pose") and high != low:
                from scipy.spatial.transform import Rotation
                from scipy.spatial.transform import Slerp

                for offset in (0, 10):
                    rotations = g2.rotation_matrix(np.stack([a[offset + 3 : offset + 9], b[offset + 3 : offset + 9]]))
                    value[offset + 3 : offset + 9] = g2.rotation_vector(
                        Slerp([0, 1], Rotation.from_matrix(rotations))([f]).as_matrix()[0]
                    )
            if key == "reference.base_to_parent" and high != low:
                from scipy.spatial.transform import Rotation
                from scipy.spatial.transform import Slerp

                value[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))([f]).as_matrix()[
                    0
                ]
            samples.append(value)
        return np.asarray(samples)

    def __getitem__(self, index):
        anchor, bounds = self.anchors[index], self.bounds[index]
        timing, action = self.policy["timing"], self.policy["action"]
        current = dict(self.dataset[round(anchor)])
        fields = self.dataset.meta.features
        for key in (
            "state.pose",
            "state.joint",
            "reference.object_pose",
            "reference.object_joint",
            "reference.base_to_parent",
        ):
            if key in fields:
                current[key] = self._signal(key, [anchor], bounds)[0]
        for key, dest in (("state.pose", "pose"), ("state.joint", "joint")):
            if key in fields:
                current[f"g2.start_{dest}"] = self._signal(key, [bounds[0]], bounds)[0]
                history = anchor + np.arange(1 - timing["history_steps"], 1) * self.fps / self.hz
                current[f"g2.{dest}_history"] = self._signal(key, history, bounds)
        if "action.pose" in fields:
            current["g2.command_pose"] = self._signal("action.pose", [anchor], bounds)[0]
            current["g2.previous_command_pose"] = self._signal("action.pose", [anchor - self.fps / self.hz], bounds)[0]
        if timing["history_steps"] > 1:
            frames = [self.dataset[int(np.clip(round(x), bounds[0], bounds[1] - 1))] for x in history]
            for camera in self.cameras:
                key = f"observation.images.{camera}"
                current[key] = np.stack([np.asarray(frame[key]) for frame in frames])
        targets = (
            anchor
            + (timing["target_offset_steps"] + np.arange(self.horizon) * timing["stride_steps"]) * self.fps / self.hz
        )
        column = label_column(self.policy)
        if column in fields:
            labels = self._signal(column, targets, bounds)
        else:
            # Velocities can be derived from position/pose trajectories; torque must be recorded.
            suffix = "pose" if action["control_quantity"] == "eef_twist" else "joint"
            source = ("action" if action["label_source"] == "command" else "state") + "." + suffix
            if action["control_quantity"] not in ("joint_velocity", "eef_twist"):
                raise ValueError(f"Dataset requires recorded {column}")
            poses = self._signal(source, targets, bounds)
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
                self._signal(source, begins, bounds),
                self._signal(source, ends, bounds),
            )
            dt = np.maximum((ends - begins) / self.fps, 1e-8)
            if suffix == "joint":
                labels = (b - a) / dt[:, None]
                labels[:, [7, 15]] = poses[:, [7, 15]]
            else:
                from scipy.spatial.transform import Rotation

                parts = []
                for offset in (0, 10):
                    angular = (
                        Rotation.from_matrix(
                            g2.rotation_matrix(b[:, offset + 3 : offset + 9])
                            @ np.swapaxes(
                                g2.rotation_matrix(a[:, offset + 3 : offset + 9]),
                                -1,
                                -2,
                            )
                        ).as_rotvec()
                        / dt[:, None]
                    )
                    parts.append(
                        np.concatenate(
                            [
                                (b[:, offset : offset + 3] - a[:, offset : offset + 3]) / dt[:, None],
                                angular,
                                poses[:, offset + 9 : offset + 10],
                            ],
                            axis=-1,
                        )
                    )
                labels = np.concatenate(parts, axis=-1)
        current["g2.actions"] = labels.astype(np.float32)
        return current
