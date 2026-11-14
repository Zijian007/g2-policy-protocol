"""One complete handshake/obs/action exchange with examples/minimal_server.py.

Uses synthetic measurements and images, without robot hardware or a model.
Read exchange() for the client-side protocol lifecycle; no robot actions execute.
"""

import argparse

import numpy as np

from g2_policy_protocol.runtime.client import PolicyClient


def exchange(url: str, *, encoding: str = "msgpack", timeout: float = 10.0) -> np.ndarray:
    """Connect, validate the service declaration, send one obs, validate its reply."""
    with PolicyClient(url, encoding=encoding, connect_timeout_s=timeout) as client:
        # The client validates the handshake before exposing service requirements.
        metadata = client.metadata
        pose = np.tile([0.35, 0, 0.5, 1, 0, 0, 0, 1, 0, 0], 2).astype(np.float32)
        # Pass RGB pixels; the client handles negotiated resize and JPEG encoding.
        images = {camera: np.full((240, 320, 3), 128, np.uint8) for camera in metadata["required_cameras"]}
        return client.infer(
            prompt="hold the current pose",
            state=pose,
            joint_state=np.zeros(16, np.float32),
            images=images,
            recv_timeout_s=timeout,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000")
    parser.add_argument("--encoding", choices=("msgpack", "json"), default="msgpack")
    args = parser.parse_args()
    actions = exchange(args.url, encoding=args.encoding)
    print(f"Received absolute targets: shape={actions.shape}, dtype={actions.dtype}")


if __name__ == "__main__":
    main()
