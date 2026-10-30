"""G2 v3 JPEG input adaptation over the shared PolicyServer."""

import argparse
import logging

import minimal_server
from g2_policy_protocol import protocol
from g2_policy_protocol.images import decode_images
from g2_policy_protocol.server import PolicyServer


class JpegMockPolicy(minimal_server.MockPolicy):
    def __init__(self, *, metadata):
        super().__init__(action_horizon=metadata["action_horizon"])
        self.metadata = metadata

    def infer(self, observation):
        observation = dict(observation)
        # A real model consumes these RGB uint8 arrays alongside the state and task.
        observation["images"] = decode_images(observation["images"], self.metadata)
        return super().infer(observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    metadata = protocol.build_metadata(

        policy_hz=30,
        action_horizon=2,
        arm="right",
        required_cameras=["head_color", "hand_right_color"],
        image_encoding="jpeg",
        image_resize="letterbox",
        image_sizes={"head_color": [224, 224], "hand_right_color": [224, 224]},
        extensions={"response_mode": "full_chunk"},
    )
    try:
        PolicyServer(JpegMockPolicy(metadata=metadata), metadata=metadata, host="127.0.0.1", port=args.port).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
