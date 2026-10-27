"""Offline integration tooling must preserve pose and image conventions."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from g2_policy_protocol import protocol as p
from g2_policy_protocol.check_server import check_action_semantics, connection_uri
from g2_policy_protocol.samples import load_sample, prepare_images, save_sample, synthetic_sample


def test_sample_roundtrip(tmp_path):
    sample = synthetic_sample()
    path = tmp_path / "observation.npz"
    save_sample(path, sample)
    restored = load_sample(path)
    assert restored["provenance"].startswith("synthetic")
    np.testing.assert_array_equal(restored["state"], sample["state"])
    for camera in p.CAMERA_IDS:
        np.testing.assert_array_equal(restored["images"][camera], sample["images"][camera])


def test_object_sample_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, state=np.array([object()], dtype=object), prompt="task", provenance="test")
    with pytest.raises(ValueError, match="Object arrays"):
        load_sample(path)


@pytest.mark.parametrize("encoding", ["raw", "jpeg"])
@pytest.mark.parametrize("resize", ["resize", "letterbox"])
def test_image_preparation_and_decode(encoding, resize):
    pytest.importorskip("PIL.Image")
    from g2_policy_protocol.image_tools import decode_images

    sample = synthetic_sample()
    # Pure red lets us check RGB/BGR independently of JPEG loss.
    sample["images"]["head_color"][:] = [255, 0, 0]
    meta = p.build_metadata(policy_hz=30, required_cameras=["head_color"],
        image_sizes={"head_color": [100, 100]}, image_encoding=encoding, image_resize=resize)
    prepared = prepare_images(sample, meta)
    assert set(prepared) == {"head_color"}
    if encoding == "jpeg":
        assert isinstance(prepared["head_color"], bytes)
    image = decode_images(prepared, meta)["head_color"]
    assert image.shape == (100, 100, 3)
    assert image[50, 50, 0] > 240 and image[50, 50, 2] < 10
    if resize == "letterbox":
        assert image[0, 50].max() < 10


def test_missing_camera():
    pytest.importorskip("PIL.Image")
    sample = synthetic_sample()
    del sample["images"]["hand_right_color"]
    with pytest.raises(ValueError, match="missing required camera"):
        prepare_images(sample, p.build_metadata(policy_hz=30))


def test_url_preserves_path_and_query():
    assert connection_uri("wss://host/policy?token=abc&encoding=json", "msgpack") == (
        "wss://host/policy?token=abc&encoding=msgpack")


def test_additional_action_diagnostics():
    state = synthetic_sample()["state"]
    actions = np.repeat(state[None], 2, axis=0)
    check_action_semantics(actions)
    actions[0, 9] = 2
    with pytest.raises(ValueError, match="gripper"):
        check_action_semantics(actions)
    actions[0, 9] = 0
    actions[0, 6:9] = actions[0, 3:6]
    with pytest.raises(ValueError, match="degenerate"):
        check_action_semantics(actions)


def test_adapter_conventions():
    adapter = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/model_adapter.py"))
    quat = adapter["quaternion_wxyz_to_rot6d"]
    np.testing.assert_allclose(quat([1, 0, 0, 0]), [1, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(quat([2**-0.5, 0, 0, 2**-0.5]), [0, 1, 0, -1, 0, 0], atol=1e-6)
    measured = synthetic_sample()["state"]
    delta = np.zeros((2, 20), dtype=np.float32)
    delta[:, 10] = [0.01, 0.02]
    delta[:, 19] = 0.7
    absolute = adapter["additive_pose_delta_to_absolute"](delta, measured)
    np.testing.assert_allclose(absolute[:, 10], measured[10] + [0.01, 0.02])
    np.testing.assert_allclose(absolute[:, 19], 0.7)
    for arm, offset in (("left", 0), ("right", 10)):
        expanded = adapter["expand_single_arm"](absolute[:, offset:offset+10], measured, arm=arm)
        np.testing.assert_array_equal(expanded[:, offset:offset+10], absolute[:, offset:offset+10])
        other = 10 - offset
        np.testing.assert_array_equal(expanded[:, other:other+10], np.repeat(measured[None, other:other+10], 2, axis=0))


@pytest.mark.parametrize("encoding", ["msgpack", "json"])
@pytest.mark.parametrize("fails", [False, True])
def test_policy_instance_request_response(encoding, fails):
    pytest.importorskip("websockets.sync.server")
    from g2_policy_protocol import codec

    example = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/minimal_server.py"))
    meta = p.build_metadata(policy_hz=30, action_horizon=2, required_cameras=["head_color"])
    sample = synthetic_sample()
    seen = []

    class RecordingPolicy(example["MockPolicy"]):
        def infer(self, observation):
            seen.append(observation)
            if fails:
                raise RuntimeError("model failed")
            return super().infer(observation)

    policy = RecordingPolicy(action_horizon=2)
    request = p.build_request(request_id=42, prompt=sample["prompt"], state=sample["state"],
        images={"head_color": sample["images"]["head_color"]}, metadata=meta)

    class Connection:
        request = SimpleNamespace(path=f"/?encoding={encoding}")

        def __init__(self):
            self.sent = []

        def send(self, frame):
            self.sent.append(codec.loads(frame, encoding=encoding))

        def __iter__(self):
            return iter([codec.dumps(request, encoding=encoding)])

    connection = Connection()
    example["handle_connection"](connection, metadata=meta, policy=policy)
    assert connection.sent[0]["type"] == "metadata"
    assert seen[0]["prompt"] == sample["prompt"]
    np.testing.assert_array_equal(seen[0]["state"], sample["state"])
    np.testing.assert_array_equal(seen[0]["images"]["head_color"], sample["images"]["head_color"])
    response = connection.sent[1]
    assert response["request_id"] == 42
    if fails:
        assert response["type"] == "error" and response["code"] == "inference_error"
    else:
        actions = p.parse_response(response, request_id=42, metadata=meta)
        np.testing.assert_array_equal(actions, np.repeat(sample["state"][None], 2, axis=0))
