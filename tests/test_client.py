"""Shared client lifecycle: failed requests are closed, never replayed."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock, patch

import numpy as np
import pytest

from g2_policy_protocol import protocol
from g2_policy_protocol.runtime import codec
from g2_policy_protocol.runtime.client import PolicyClient
from g2_policy_protocol.runtime.images import decode_images, encode_jpeg


def connected_client(**metadata_options):
    client = PolicyClient()
    client._metadata = protocol.build_metadata(policy_hz=30, required_cameras=[], **metadata_options)
    client._ws = Mock()
    return client, client._ws


def test_invalid_handshake_fails_without_retry_and_snapshot_is_not_accepted():
    client = PolicyClient()
    websocket = Mock()
    metadata = protocol.build_metadata(policy_hz=30, required_cameras=[])
    websocket.recv.return_value = codec.dumps({**metadata, "protocol": "other"})
    with patch("g2_policy_protocol.runtime.client.connect", return_value=websocket) as connect:
        with pytest.raises(ValueError, match="expects"):
            client.connect()
        connect.assert_called_once()
    websocket.close_socket.assert_called_once()
    assert client._ws is None
    with pytest.raises(ConnectionError, match="connect before"):
        _ = client.metadata


def test_metadata_snapshot_cannot_change_negotiated_requirements():
    client, _ = connected_client()
    snapshot = client.metadata
    snapshot["action_spec"]["dim"] = 1
    snapshot["observation_spec"]["pose"] = False
    assert client.metadata["action_spec"]["dim"] == 20
    assert client.metadata["observation_spec"]["pose"] is True


@pytest.mark.parametrize("failure", ["timeout", "send", "decode", "id", "model", "shape"])
def test_failed_exchange_is_aborted_without_reconnect_or_replay(failure):
    client, websocket = connected_client()
    response = protocol.build_response(request_id=1, actions=np.zeros((1, 20)), metadata=client.metadata)
    expected = ValueError
    if failure == "timeout":
        websocket.recv.side_effect = TimeoutError("slow")
        expected = TimeoutError
    elif failure == "send":
        websocket.send.side_effect = OSError("disconnected")
        expected = OSError
    elif failure == "decode":
        websocket.recv.return_value = b"\xc1"
    else:
        if failure == "id":
            response["request_id"] = 99
        elif failure == "model":
            response = protocol.build_error(request_id=1, message="model failed")
            expected = RuntimeError
        else:
            response["actions"] = np.zeros((1, 10), np.float32)
        websocket.recv.return_value = codec.dumps(response)
    with patch("g2_policy_protocol.runtime.client.connect") as connect:
        with pytest.raises(expected):
            client.infer(prompt="test", state=np.zeros(20), images={})
        connect.assert_not_called()
    websocket.send.assert_called_once()
    websocket.close_socket.assert_called_once()
    assert client._ws is None


def test_reset_and_inference_share_a_monotonic_request_sequence():
    client, websocket = connected_client(reset_supported=True)
    websocket.recv.side_effect = [
        codec.dumps(protocol.build_response(request_id=1, actions=np.zeros((1, 20)), metadata=client.metadata)),
        codec.dumps(protocol.build_reset_ack(request_id=2)),
    ]
    client.infer(prompt="test", state=np.zeros(20), images={})
    client.reset()
    messages = [codec.loads(call.args[0]) for call in websocket.send.call_args_list]
    assert [(message["type"], message["request_id"]) for message in messages] == [("obs", 1), ("reset", 2)]


@pytest.mark.parametrize("image_encoding", ["raw", "jpeg"])
def test_client_prepares_current_rgb_frames(image_encoding):
    client, websocket = connected_client(image_encoding=image_encoding)
    client._metadata["required_cameras"] = ["head_color"]
    client._metadata["image_sizes"] = {"head_color": [8, 12]}
    websocket.recv.return_value = codec.dumps(
        protocol.build_response(
            request_id=1,
            actions=np.zeros((1, 20)),
            metadata=client.metadata,
        )
    )
    image = np.full((24, 36, 3), [255, 0, 0], np.uint8)
    client.infer(prompt="test", state=np.zeros(20), images={"head_color": image})
    request = codec.loads(websocket.send.call_args.args[0])
    rgb = decode_images(request["images"], client.metadata)["head_color"]
    assert rgb.shape == (8, 12, 3) and rgb.dtype == np.uint8
    assert rgb[..., 0].min() > 240 and rgb[..., 2].max() < 10


@pytest.mark.parametrize("image_encoding", ["raw", "jpeg"])
def test_history_uses_cached_wire_frames_and_skips_current_preparation(image_encoding):
    client, websocket = connected_client(
        image_encoding=image_encoding,
        observation_spec=dict(
            pose=False, joint=False, prompt=False, history_steps=2, history_hz=10, context_keys=["image_history"]
        ),
    )
    client._metadata["required_cameras"] = ["head_color"]
    client._prepare_images = Mock(side_effect=AssertionError("current image duplicated"))
    frame = np.zeros((8, 12, 3), np.uint8)
    frames = [encode_jpeg(frame)] * 2 if image_encoding == "jpeg" else np.stack([frame, frame])
    websocket.recv.return_value = codec.dumps(
        protocol.build_response(
            request_id=1,
            actions=np.zeros((1, 20)),
            metadata=client.metadata,
        )
    )
    client.infer(images={"head_color": frame}, context={"image_history": {"head_color": frames}}, observation_time_ns=1)
    request = codec.loads(websocket.send.call_args.args[0])
    assert "images" not in request and client.last_stats.letterbox_s == 0
    np.testing.assert_array_equal(request["context"]["image_history"]["head_color"], frames)
    client._prepare_images.assert_not_called()


def test_cancel_interrupts_blocking_receive_and_prevents_future_reconnects():
    client, websocket = connected_client()
    receiving, closed = Event(), Event()

    def receive(**kwargs):
        receiving.set()
        assert closed.wait(2)
        raise OSError("socket closed")

    websocket.recv.side_effect = receive
    websocket.close_socket.side_effect = closed.set
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(client.infer, prompt="test", state=np.zeros(20), images={})
        try:
            assert receiving.wait(2)
        finally:
            client.cancel_pending()
        with pytest.raises(ConnectionError, match="cancelled"):
            result.result(timeout=2)
    with patch("g2_policy_protocol.runtime.client.connect") as connect:
        with pytest.raises(ConnectionError, match="cancelled"):
            client.infer(prompt="test", state=np.zeros(20), images={})
        connect.assert_not_called()
    websocket.send.assert_called_once()


def test_cancel_while_connect_is_opening_discards_the_new_socket():
    client = PolicyClient()
    websocket = Mock()

    def opening(*args, **kwargs):
        client.cancel_pending()
        return websocket

    with patch("g2_policy_protocol.runtime.client.connect", side_effect=opening):
        with pytest.raises(ConnectionError, match="cancelled"):
            client.connect()
    websocket.close_socket.assert_called_once()
    websocket.recv.assert_not_called()
