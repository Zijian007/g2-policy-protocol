"""Configuration choices have independently checked numerical meanings."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g2_policy_tools.capabilities import policy_capabilities
from g2_policy_tools.geometry import ActionCodec, matrix_pose
from g2_policy_tools.schema import public_settings, resolve_policy
from g2_policy_tools.state import encode_state


def transform(position, angles):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    result[:3, 3] = position
    return result


def pair(matrix, gripper):
    return np.tile(matrix_pose(matrix, gripper, "rot6d"), 2)


@pytest.mark.parametrize("method", [None, "component", "se3"])
@pytest.mark.parametrize("scope", ["chunk", "step"])
@pytest.mark.parametrize("rotation", ["rot6d", "quaternion", "axis_angle"])
@pytest.mark.parametrize("arm", ["both", "left", "right"])
def test_pose_method_and_scope_match_geometry_and_restore_wire_targets(method, scope, rotation, arm):
    spec = {"coordinates": {"rotation": rotation}, "delta": method is not None}
    if method is not None:
        spec["method"] = method
        spec["reference"] = {"source": "state", "scope": scope}
    policy = resolve_policy({"action": {"pose": spec}}, horizon=3)
    start = transform([0.4, -0.3, 0.6], [0.2, -0.1, np.pi / 2])
    matrices = [
        transform([0.5, -0.25, 0.62], [0.3, -0.2, 1.3]),
        transform([0.55, -0.23, 0.65], [0.35, -0.15, 1.2]),
        transform([0.6, -0.2, 0.7], [0.4, -0.1, 1.1]),
    ]
    targets = np.stack([pair(matrix, grip) for matrix, grip in zip(matrices, [0.1, 0.7, 0.4], strict=True)])
    observation = {"observation/pose": pair(start, 0.6)}
    codec = ActionCodec(policy, arm)
    encoded = codec.encode(targets, observation)
    expected = []
    reference = start
    for matrix, grip in zip(matrices, [0.1, 0.7, 0.4], strict=True):
        local = np.linalg.inv(reference) @ matrix if method == "se3" else matrix
        value = matrix_pose(local, grip, rotation)
        if method == "component":
            value[:-1] -= matrix_pose(reference, 0, rotation)[:-1]
        expected.append(np.tile(value, 2 if arm == "both" else 1))
        if scope == "step":
            reference = matrix
    np.testing.assert_allclose(encoded, expected, atol=5e-7)
    decoded = codec.decode(encoded, observation)
    active = slice(None) if arm == "both" else slice(0, 10) if arm == "left" else slice(10, 20)
    np.testing.assert_allclose(decoded[:, active], targets[:, active], atol=1e-6)
    if arm != "both":
        inactive = slice(10, 20) if arm == "left" else slice(0, 10)
        np.testing.assert_array_equal(decoded[:, inactive], np.tile(observation["observation/pose"][inactive], (3, 1)))
    public = public_settings({"schema_version": 7, "policy": policy})["policy"]
    assert resolve_policy(public, horizon=3) == policy


@pytest.mark.parametrize("method", ["component", "se3"])
@pytest.mark.parametrize("rotation", ["rot6d", "quaternion", "axis_angle"])
def test_state_history_uses_the_configured_method_and_chunk_reference(method, rotation):
    matrices = [transform([0.3, 0.2, 0.5], [0.1, 0.2, 0.3]), transform([0.4, 0.3, 0.6], [0.2, 0.3, 0.5])]
    history = np.stack([pair(matrix, 0.7) for matrix in matrices])
    policy = resolve_policy(
        {
            "state": {
                "pose": {
                    "coordinates": {"rotation": rotation},
                    "delta": True,
                    "method": method,
                    "reference": {"source": "state", "scope": "chunk"},
                }
            },
            "timing": {"history_steps": 2},
        },
        horizon=3,
    )
    observation = {"observation/pose": history[-1], "observation/pose_history": history}
    actual = encode_state(policy, observation, "right", flatten=False)
    expected = []
    for matrix in matrices:
        local = np.linalg.inv(matrices[-1]) @ matrix if method == "se3" else matrix
        value = matrix_pose(local, 0.7, rotation)
        if method == "component":
            value[:-1] -= matrix_pose(matrices[-1], 0, rotation)[:-1]
        expected.append(value)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize(
    "spec,message",
    [
        ({"delta": "true"}, "delta must be a YAML boolean"),
        ({"delta": 1}, "delta must be a YAML boolean"),
        ({"delta": True}, "method must be"),
        ({"coordinates": {"parent_frame": "reference"}}, "Unknown YAML field"),
        ({"delta": False, "reference": {"source": "state", "scope": "chunk"}}, "reference requires"),
        ({"delta": True, "method": "se3", "reference": {"source": "object", "scope": "chunk"}}, "must be one of"),
        (
            {"delta": True, "method": "component", "reference": {"source": "state", "scope": "sequence"}},
            "reference.scope",
        ),
        ({"quantity": "twist"}, "quantity must be one of"),
        ({"spatial": {"mode": "relative"}}, "Unknown YAML field"),
        ({"temporal": {"anchor": "object"}}, "Unknown YAML field"),
    ],
)
def test_invalid_combinations_fail_at_config_boundary(spec, message):
    with pytest.raises(ValueError, match=message):
        resolve_policy({"action": {"pose": spec}}, horizon=3)


def test_state_relative_to_itself_requires_history():
    with pytest.raises(ValueError, match="history_steps > 1"):
        resolve_policy(
            {
                "state": {
                    "pose": {
                        "delta": True,
                        "method": "component",
                        "reference": {"source": "state", "scope": "chunk"},
                    }
                }
            },
            horizon=3,
        )


def test_parent_frame_is_not_a_training_field():
    with pytest.raises(ValueError, match="Unknown YAML field"):
        resolve_policy({"action": {"pose": {"coordinates": {"parent_frame": "camera"}}}}, horizon=3)


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_model_horizon_is_validated_at_the_shared_boundary(horizon):
    with pytest.raises(ValueError, match="policy.timing.action_horizon"):
        resolve_policy({}, horizon=horizon)


@pytest.mark.parametrize("key,value", [
    ("history_steps", 0), ("stride_steps", True), ("target_offset_steps", -1),
    ("sample_hz", float("nan")), ("sample_hz", float("inf")), ("sample_hz", 0),
])
def test_invalid_timing_is_rejected_without_a_framework_validator(key, value):
    with pytest.raises(ValueError, match=f"policy.timing.{key}"):
        resolve_policy({"timing": {key: value}}, horizon=3)


@pytest.mark.parametrize(
    "source,context",
    [
        ("state", None),
        ("action", "command_pose"),
    ],
)
def test_se3_action_advertises_only_the_needed_command_context(source, context):
    policy = resolve_policy(
        {
            "action": {
                "pose": {
                    "delta": True,
                    "method": "se3",
                    "reference": {"source": source, "scope": "chunk"},
                }
            }
        },
        horizon=3,
    )
    caps = policy_capabilities(policy, [], prompt=False, reset=True)
    assert set(caps["observation_spec"]["context_keys"]) == ({context} if context else set())


def test_relative_labels_do_not_mutate_config_or_observations():
    raw = {
        "action": {
            "pose": {"delta": True, "method": "component", "reference": {"source": "action", "scope": "step"}}
        }
    }
    original = copy.deepcopy(raw)
    policy = resolve_policy(raw, horizon=3)
    observation = {"observation/pose": pair(np.eye(4), 0.6), "observation/command_pose": pair(np.eye(4), 0.0)}
    targets = np.stack([pair(np.eye(4), 0.2)] * 3)
    original_targets = targets.copy()
    ActionCodec(policy).encode(targets, observation)
    assert raw == original
    np.testing.assert_array_equal(targets, original_targets)


def test_joint_se3_is_rejected():
    with pytest.raises(ValueError, match="must be component"):
        resolve_policy(
            {
                "action": {
                    "pose": {"enabled": False},
                    "joint": {
                        "enabled": True,
                        "delta": True,
                        "method": "se3",
                        "reference": {"source": "state", "scope": "chunk"},
                    },
                }
            },
            horizon=3,
        )


def test_codec_rejects_removed_twist_in_saved_numerical_contract():
    policy = resolve_policy({}, horizon=3)
    policy["action"]["pose"]["quantity"] = "twist"
    with pytest.raises(ValueError, match='Unsupported G2 control quantity'):
        ActionCodec(policy).encode(np.zeros((3, 14)), {})
