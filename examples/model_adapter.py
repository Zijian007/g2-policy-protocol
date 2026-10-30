"""Use the shared ActionCodec to turn model outputs into canonical robot actions."""

import numpy as np
from g2_policy_protocol import protocol
from g2_policy_protocol.data.encoding import (
    ActionCodec,
    rotation_matrix,
    rotation_vector,
)
from g2_policy_protocol.data.policy import resolve_policy

if __name__ == "__main__":
    # Reorder a robot's wxyz quaternion explicitly; shared geometry uses xyzw.
    quaternion_wxyz = np.array([2**-0.5, 0, 0, 2**-0.5])
    quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
    rot6d = rotation_vector(rotation_matrix(quaternion_xyzw, "quaternion"))
    measured_pose = np.zeros(20, dtype=np.float32)
    measured_pose[:3] = [0.35, 0.2, 0.5]
    measured_pose[10:13] = [0.35, -0.2, 0.5]
    measured_pose[3:9] = measured_pose[13:19] = rot6d

    # Resolve once when loading the model; reuse the codec for every inference.
    policy = resolve_policy(
        {
            "action": {
                "pose": {
                    "delta": True,
                    "method": "component",
                    "reference": {"source": "state", "scope": "chunk"},
                }
            },
        },
        horizon=2,
    )
    codec = ActionCodec(policy, arm="right")

    # Example model output: right-arm xyz/rot6d deltas, absolute gripper values.
    model_actions = np.zeros((2, 10), dtype=np.float32)
    model_actions[:, 0] = [0.01, 0.02]
    model_actions[:, 9] = 0.5
    actions = codec.decode(model_actions, {"observation/pose": measured_pose})
    # decode also fills the inactive left arm from measured_pose.
    metadata = protocol.build_metadata(

        policy_hz=30,
        action_horizon=2,
        arm="right",
        required_cameras=[],
    )
    response = protocol.build_response(request_id=1, actions=actions, metadata=metadata)
    print("Converted right-arm actions:", response["actions"].shape)
    print("Right-arm x targets:", response["actions"][:, 10])
