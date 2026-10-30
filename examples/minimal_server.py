"""Run a mock G2 v3 policy through the same server used by real model backends."""

import argparse
import logging

import numpy as np
from g2_policy_protocol import protocol
from g2_policy_protocol.server import PolicyServer


class MockPolicy:
    """Replace this stateless mock with your model and input/output adapter."""

    def __init__(self, *, action_horizon):
        self.action_horizon = action_horizon

    def infer(self, observation):
        actions = np.repeat(observation["state"][None, :], self.action_horizon, axis=0)
        return {"actions": actions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-hz", type=float, default=30)
    parser.add_argument("--action-horizon", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    metadata = protocol.build_metadata(

        policy_hz=args.policy_hz,
        action_horizon=args.action_horizon,
        required_cameras=["head_color"],
        image_sizes={"head_color": [224, 224]},
        extensions={"response_mode": "full_chunk"},
    )
    policy = MockPolicy(action_horizon=metadata["action_horizon"])
    try:
        PolicyServer(policy, metadata=metadata, host=args.host, port=args.port).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
