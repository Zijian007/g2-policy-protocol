import numpy as np
import pytest
from g2_policy_protocol.contracts.policy import resolve_policy
from g2_policy_protocol.processing.encoding import encode_state, state_width
from g2_policy_protocol.processing.statistics import RunningStatistics


def test_flat_history_width_includes_every_enabled_modality_and_frame():
    policy = resolve_policy({"state": {"joint": {"enabled": True}}, "timing": {"history_steps": 2}}, horizon=4)
    assert state_width(policy, "both") == 72
    assert state_width(policy, "right") == 36


def test_shared_statistics_keep_history_columns_separate_and_include_chunk_tails():
    states = np.array([[1.0, 10.0], [3.0, 10.0]], np.float32)
    actions = np.array([[[0.0], [1.0], [2.0]], [[1.0], [2.0], [2.0]]], np.float32)
    state_stats, action_stats = RunningStatistics(), RunningStatistics()
    state_stats.update(states)
    action_stats.update(actions)
    np.testing.assert_array_equal(state_stats.get_statistics()["mean"], [2, 10])
    np.testing.assert_allclose(action_stats.get_statistics()["mean"], [8 / 6])


@pytest.mark.parametrize("kind,width", [("pose", 20), ("joint", 16)])
def test_state_history_does_not_require_an_unused_current_vector(kind, width):
    policy = resolve_policy(
        {
            "state": {"pose": {"enabled": kind == "pose"}, "joint": {"enabled": kind == "joint"}},
            "timing": {"history_steps": 2},
        },
        horizon=4,
    )
    history = np.arange(2 * width, dtype=np.float32).reshape(2, width)
    encoded = encode_state(policy, {f"observation/{kind}_history": history}, flatten=False)
    np.testing.assert_array_equal(encoded, history)


def test_delta_state_still_requires_its_declared_measured_reference():
    policy = resolve_policy(
        {
            "state": {
                "pose": {"enabled": False},
                "joint": {
                    "enabled": True,
                    "delta": True,
                    "method": "component",
                    "reference": {"source": "state", "scope": "chunk"},
                },
            },
            "timing": {"history_steps": 2},
        },
        horizon=4,
    )
    with pytest.raises(ValueError, match="Reference state requires observation/joint"):
        encode_state(policy, {"observation/joint_history": np.zeros((2, 16), np.float32)})
