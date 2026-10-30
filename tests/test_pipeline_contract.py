import copy

import numpy as np
import pytest
from g2_policy_protocol import protocol
from g2_policy_protocol.data.dataset import validate_dataset_contract
from g2_policy_protocol.data.policy import policy_capabilities, resolve_policy


def test_full_chunk_profile_rejects_short_response_but_accepts_short_rtc_prefix():
    metadata = protocol.build_metadata(

        policy_hz=30,
        action_horizon=4,
        required_cameras=[],
        rtc_supported=True,
        extensions={"response_mode": "full_chunk"},
    )
    with pytest.raises(ValueError, match="action_horizon"):
        protocol.build_response(
            request_id=1, actions=np.zeros((1, 20)), metadata=metadata
        )
    request = protocol.build_request(
        request_id=1,
        prompt="demo",
        state=np.zeros(20),
        images={},
        metadata=metadata,
        rtc_enabled=True,
        prev_chunk_left_over=np.zeros((2, 20)),
        inference_delay=1,
    )
    assert request["prev_chunk_left_over"].shape == (2, 20)
    assert protocol.build_response(
        request_id=1, actions=np.zeros((4, 20)), metadata=metadata
    )["actions"].shape == (4, 20)


def test_shared_capabilities_require_real_reference_histories():
    policy = resolve_policy(
        {
            "state": {
                "joint": {
                    "enabled": True,
                    "delta": True,
                    "method": "component",
                    "reference": {"source": "action", "scope": "chunk"},
                }
            },
            "action": {
                "pose": {
                    "delta": True,
                    "method": "component",
                    "reference": {"source": "action", "scope": "chunk"},
                }
            },
            "timing": {"history_steps": 3, "stride_steps": 2},
        },
        horizon=4,
    )
    caps = policy_capabilities(policy, ["head_color"], prompt=False, reset=True)
    assert caps["policy_hz"] == 15
    assert set(caps["observation_spec"]["context_keys"]) == {
        "command_pose",
        "command_joint",
        "pose_history",
        "joint_history",
        "image_history",
    }
    assert caps["observation_spec"]["joint"] and caps["observation_spec"]["pose"]


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("action_supervision", "state", "action labels"),
        ("pose_frame", "world", "pose_frame"),
        ("action_parent_frame", "camera", "action_parent_frame"),
        ("ee_frame", "tcp", "ee_frame"),
        ("rotation", "quaternion", "rotation"),
        ("segment_policy", "concatenated", "segments"),
    ],
)
def test_dataset_semantics_are_checked_before_sampling(field, value, message):
    policy = resolve_policy({}, horizon=4)
    features = {"state.pose": {"shape": [20]}, "action.pose": {"shape": [20]}}
    info = {"fps": 30, "wholebody_export": {field: value}}
    with pytest.raises(ValueError, match=message):
        validate_dataset_contract(features, info, policy, ())


def test_measured_supervision_selects_state_without_relabelling_disk_actions():
    policy = resolve_policy({"action": {"pose": {"source": "state"}}}, horizon=4)
    features = {"state.pose": {"shape": [20]}}
    original = copy.deepcopy(features)
    validate_dataset_contract(
        features,
        {"fps": 30, "wholebody_export": {"action_supervision": "state"}},
        policy,
        (),
    )
    assert features == original


@pytest.mark.parametrize("group", ["state", "action"])
def test_arm_base_requires_recorded_nine_dimensional_pose_and_wire_context(group):
    policy = resolve_policy({group: {"pose": {"coordinates": {"parent_frame": "arm_base_link"}}}}, horizon=4)
    features = {"state.pose": {"shape": [20]}, "action.pose": {"shape": [20]}}
    with pytest.raises(ValueError, match="state.arm_base"):
        validate_dataset_contract(features, {}, policy, ())
    features["state.arm_base"] = {"shape": [10]}
    with pytest.raises(ValueError, match="shape \\[9\\]"):
        validate_dataset_contract(features, {}, policy, ())
    features["state.arm_base"] = {"shape": [9]}
    validate_dataset_contract(features, {}, policy, ())
    metadata = policy_capabilities(policy, (), prompt=False, reset=True)
    assert metadata["observation_spec"]["context_keys"] == ["base_to_parent"]
