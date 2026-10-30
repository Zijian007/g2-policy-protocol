"""Run a mock G2 v3 policy through the same server used by real model backends."""

import argparse
import logging

import numpy as np
from g2_policy_protocol import protocol
from g2_policy_protocol.images import decode_images
from g2_policy_protocol.server import PolicyServer


class MockPolicy:
    """Replace this stateless mock with your model and input/output adapter."""

    def __init__(self, *, metadata):
        self.metadata = metadata

    def infer(self, observation):
        # A real model feeds these RGB uint8 arrays (JPEG decoded if negotiated) to its network.
        decode_images(observation["images"], self.metadata)
        actions = np.repeat(observation["state"][None, :], self.metadata["action_horizon"], axis=0)
        return {"actions": actions}


def build_metadata(args):
    if args.jpeg:
        # Right arm with head + right wrist cameras, JPEG letterboxed to 224x224.
        cameras = ["head_color", "hand_right_color"]
        options = dict(arm="right", image_encoding="jpeg", image_resize="letterbox")
    else:
        cameras = ["head_color"]
        options = {}
    return protocol.build_metadata(
        policy_hz=args.policy_hz,
        action_horizon=args.action_horizon,
        required_cameras=cameras,
        image_sizes={camera: [224, 224] for camera in cameras},
        extensions={"response_mode": "full_chunk"},
        **options,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-hz", type=float, default=30)
    parser.add_argument("--action-horizon", type=int, default=2)
    parser.add_argument("--jpeg", action="store_true", help="Right-arm JPEG/letterbox example")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    metadata = build_metadata(args)
    try:
        PolicyServer(MockPolicy(metadata=metadata), metadata=metadata, host=args.host, port=args.port).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
