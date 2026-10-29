"""Sampling preserves geometry and boundaries while sharing reads within a sample."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g2_policy_tools.dataset import G2Dataset
from g2_policy_tools.geometry import ActionCodec, matrix_pose, rotation_matrix
from g2_policy_tools.schema import resolve_policy
from g2_policy_tools.state import encode_state, references_from_sample


class Rows:
    def __init__(self, rows):
        self.rows = rows
        self.reads = Counter()

    def __getitem__(self, index):
        self.reads[index] += 1
        return self.rows[index]


class Recording:
    def __init__(self):
        rows = []
        for index in range(8):
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_rotvec([0, 0, index * 0.2]).as_matrix()
            transform[0, 3] = index
            pose = np.tile(matrix_pose(transform, 0.4), 2).astype(np.float32)
            command = pose.copy()
            command[[0, 10]] += 10
            rows.append({
                "state.pose": pose,
                "action.pose": command,
                "state.joint": np.full(16, index, np.float32),
                "action.joint": np.full(16, index + 10, np.float32),
                "reference.object_pose": pose.copy(),
                "reference.object_joint": np.full(16, index - 1, np.float32),
                "reference.base_to_parent": transform,
                "observation.images.head_color": np.full((2, 2, 3), index, np.uint8),
            })
        self.hf_dataset = Rows(rows)
        self.meta = SimpleNamespace(fps=4, features=dict.fromkeys(rows[0]))
        self.episode_data_index = {"from": [0, 4], "to": [4, 8]}
        self.decodes = Counter()

    def __getitem__(self, index):
        self.decodes[index] += 1
        return dict(self.hf_dataset[index])


def test_sample_reads_each_required_row_once_and_does_not_retain_cache():
    source = Recording()
    policy = resolve_policy({"timing": {"sample_hz": 4}}, horizon=4)
    dataset = G2Dataset(source, policy, (), 4)
    sample = dataset.sample(0, images=False)
    assert source.hf_dataset.reads == Counter({0: 1, 1: 1, 2: 1, 3: 1})
    assert set(key for key in sample if key.startswith("g2.")) == {
        "g2.pose_history", "g2.actions", "g2.action_is_pad",
    }
    np.testing.assert_allclose(sample["g2.actions"][:, 0], [10, 11, 12, 13])
    source.hf_dataset.rows[0]["action.pose"][[0, 10]] = 42
    assert dataset.sample(0, images=False)["g2.actions"][0, 0] == 42
    # Samples own their arrays, including the integer-position fast path.
    sample["g2.pose_history"][0, 0] = -100
    assert source.hf_dataset.rows[0]["state.pose"][0] == 0


def test_boundary_image_history_decodes_each_frame_once():
    source = Recording()
    policy = resolve_policy({"timing": {"history_steps": 3, "sample_hz": 4}}, horizon=2)
    sample = G2Dataset(source, policy, ("head_color",), 2)[4]
    assert source.decodes == Counter({4: 1})
    np.testing.assert_array_equal(sample["g2.pose_history"][:, 0], [4, 4, 4])
    np.testing.assert_array_equal(sample["observation.images.head_color"][:, 0, 0, 0], [4, 4, 4])
    np.testing.assert_allclose(sample["g2.actions"][:, 0], [14, 15])


@pytest.mark.parametrize("reference", ["state", "action"])
def test_fractional_pose_references_restore_labels(reference):
    source = Recording()
    policy = resolve_policy({
        "state": {"pose": {
            "delta": True,
            "method": "component",
            "reference": {"source": reference, "scope": "chunk"},
        }},
        "action": {"pose": {
            "delta": True,
            "method": "component",
            "reference": {"source": reference, "scope": "step"},
        }},
        "timing": {"sample_hz": 8, "history_steps": 2},
    }, horizon=3)
    sample = G2Dataset(source, policy, (), 3).sample(3, images=False)
    observation = references_from_sample(sample)
    np.testing.assert_allclose(sample["state.pose"][0], 1.5)
    np.testing.assert_allclose(rotation_matrix(sample["state.pose"][3:9]),
                               Rotation.from_rotvec([0, 0, 0.3]).as_matrix(), atol=1e-6)
    expected_reference = {"state": 1.5, "action": 11.5}[reference]
    np.testing.assert_allclose(encode_state(policy, observation)[0], 1.0 - expected_reference)
    codec = ActionCodec(policy)
    np.testing.assert_allclose(codec.decode(codec.encode(sample["g2.actions"], observation), observation),
                               sample["g2.actions"], atol=1e-6)
    assert all(count == 1 for count in source.hf_dataset.reads.values())


@pytest.mark.parametrize("reference", ["state", "action"])
def test_joint_only_state_samples_its_reference_and_history(reference):
    policy = resolve_policy({
        "state": {"pose": {"enabled": False}, "joint": {
            "enabled": True, "delta": True, "method": "component",
            "reference": {"source": reference, "scope": "chunk"},
        }},
        "action": {"pose": {"enabled": False}, "joint": {
            "enabled": True, "delta": True, "method": "component",
            "reference": {"source": "state", "scope": "chunk"},
        }},
        "timing": {"sample_hz": 4, "history_steps": 2},
    }, horizon=2)
    sample = G2Dataset(Recording(), policy, (), 2).sample(5, images=False)
    assert "g2.pose_history" not in sample
    assert "g2.start_pose" not in sample
    state = encode_state(policy, references_from_sample(sample), flatten=False)
    current = 5 if reference == "state" else 15
    np.testing.assert_allclose(state[:, 0], [4 - current, 5 - current])
    np.testing.assert_allclose(state[:, 7], [4, 5])


def test_dynamic_parent_is_interpolated_even_with_a_separate_state_frame():
    policy = resolve_policy({
        "action": {"pose": {"coordinates": {"base_to_parent": "observation"}}},
        "timing": {"sample_hz": 8},
    }, horizon=2)
    sample = G2Dataset(Recording(), policy, (), 2).sample(1, images=False)
    transform = sample["reference.base_to_parent"]
    np.testing.assert_allclose(transform[0, 3], 0.5)
    np.testing.assert_allclose(transform[:3, :3], Rotation.from_rotvec([0, 0, 0.1]).as_matrix(), atol=1e-6)


def test_derived_velocity_clamps_to_episode_and_keeps_absolute_grippers():
    policy = resolve_policy({
        "action": {"pose": {"enabled": False}, "joint": {"enabled": True, "quantity": "velocity"}},
        "timing": {"sample_hz": 4},
    }, horizon=4)
    source = Recording()
    sample = G2Dataset(source, policy, (), 4).sample(2, images=False)
    np.testing.assert_allclose(sample["g2.actions"][:, 0], 4)
    np.testing.assert_allclose(sample["g2.actions"][:, 7], [12, 13, 13, 13])
    np.testing.assert_array_equal(sample["g2.action_is_pad"], [False, False, True, True])
    assert all(count == 1 for count in source.hf_dataset.reads.values())
