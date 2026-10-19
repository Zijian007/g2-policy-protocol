"""Send synthetic observations to minimal_server.py; never execute actions."""

import argparse

import numpy as np
from websockets.sync.client import connect

from g2_policy_protocol import codec, protocol as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--encoding", choices=("msgpack", "json"), default="msgpack")
    parser.add_argument("--policy-hz", type=float, default=30)
    args = parser.parse_args()
    uri = f"ws://{args.host}:{args.port}/?encoding={args.encoding}"
    with connect(uri, compression=None, max_size=None, open_timeout=10) as ws:
        metadata = codec.loads(ws.recv(timeout=10), encoding=args.encoding)
        p.validate_metadata(metadata, policy_hz=args.policy_hz)
        # Synthetic pose with identity rotations. No robot hardware is used.
        state = np.zeros(p.POSE_DIM, dtype=np.float32)
        state[[3, 7, 13, 17]] = 1
        for request_id in (1, 2):
            request = p.build_request(
                request_id=request_id, prompt="保持当前位姿", state=state,
                images={"head_color": np.zeros((224, 224, 3), dtype=np.uint8)},
                metadata=metadata,
            )
            ws.send(codec.dumps(request, encoding=args.encoding))
            response = codec.loads(ws.recv(timeout=10), encoding=args.encoding)
            actions = p.parse_response(response, request_id=request_id)
            np.testing.assert_array_equal(
                actions, np.repeat(state[None, :], metadata["action_horizon"], axis=0),
            )
            print(f"{args.encoding}: request_id={request_id}, actions={actions.shape}, OK")


if __name__ == "__main__":
    main()
