"""JPEG variant of minimal_server.py using optional Pillow decoding."""

from functools import partial

from websockets.sync.server import serve

from g2_policy_protocol import protocol as p
from g2_policy_protocol.image_tools import decode_images
import minimal_server


class JpegMockPolicy(minimal_server.MockPolicy):
    def __init__(self, *, metadata):
        super().__init__(action_horizon=metadata["action_horizon"])
        self.metadata = metadata

    def infer(self, observation):
        observation = dict(observation)
        observation["images"] = decode_images(observation["images"], self.metadata)
        # Here every image is an RGB uint8 HxWx3 array, validated after decoding.
        return super().infer(observation)


metadata = p.build_metadata(
    policy_hz=30, action_horizon=2, arm="right",
    required_cameras=["head_color", "hand_right_color"],
    image_encoding="jpeg", image_resize="letterbox",
    image_sizes={"head_color": [224, 224], "hand_right_color": [224, 224]},
)


if __name__ == "__main__":
    policy = JpegMockPolicy(metadata=metadata)
    with serve(partial(minimal_server.handle_connection, metadata=metadata, policy=policy),
               "127.0.0.1", 8000, compression=None, max_size=None) as server:
        print("READY mock JPEG server ws://127.0.0.1:8000", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
