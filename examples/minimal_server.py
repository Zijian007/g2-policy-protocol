"""Run a mock G2 WebSocket policy server without a GPU or model weights."""

import argparse
from functools import partial
import logging

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from g2_policy_protocol import codec, protocol as p

logger = logging.getLogger(__name__)


class MockPolicy:
    """Replace this stateless mock with your model and input/output adapter."""

    def __init__(self, *, action_horizon):
        self.action_horizon = action_horizon

    def infer(self, observation):
        # observation contains prompt, measured state, images and optional fields.
        # A real policy preprocesses inputs, runs its model and restores pose20.
        actions = np.repeat(observation["state"][None, :], self.action_horizon, axis=0)
        return {"actions": actions}


def handle_connection(websocket, *, metadata, policy):
    try:
        encoding = codec.encoding_from_path(websocket.request.path)
    except ValueError:
        websocket.close(code=1008, reason="encoding must be json or msgpack")
        return

    try:
        # Metadata must be the first application message on each connection.
        websocket.send(codec.dumps(metadata, encoding=encoding))
        for frame in websocket:
            request_id = 0
            error_code = "invalid_request"
            try:
                payload = codec.loads(frame, encoding=encoding)
                if isinstance(payload, dict):
                    candidate = payload.get("request_id")
                    if type(candidate) is int and candidate >= 0:
                        request_id = candidate
                observation = p.parse_request(payload, metadata=metadata)
                error_code = "inference_error"
                result = policy.infer(observation)
                response = p.build_response(request_id=request_id, actions=result["actions"])
            except Exception as exc:
                logger.exception("Request failed")
                response = p.build_error(
                    request_id=request_id,
                    code=error_code,
                    message=f"{type(exc).__name__}: {exc}",
                )
            websocket.send(codec.dumps(response, encoding=encoding))
    except ConnectionClosed:
        logger.info("Client disconnected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-hz", type=float, default=30)
    parser.add_argument("--action-horizon", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Validate capabilities before listening. This mock supports raw RGB only.
    metadata = p.build_metadata(
        policy_hz=args.policy_hz,
        action_horizon=args.action_horizon,
        arm="both",
        required_cameras=["head_color"],
        image_encoding="raw",
        image_sizes={"head_color": [224, 224]},
        image_resize="resize",
        rtc_supported=False,
        requires_joint_state=False,
    )
    # Load a real model here once, before listening. The mock has no mutable cache.
    policy = MockPolicy(action_horizon=metadata["action_horizon"])
    with serve(
        partial(handle_connection, metadata=metadata, policy=policy), args.host, args.port,
        compression=None, max_size=None,
    ) as server:
        print(f"READY mock server ws://{args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
