"""Offline integration tooling must preserve pose and image conventions."""

import asyncio
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from g2_policy_protocol import protocol as p
from g2_policy_protocol.check_server import check_action_semantics, connection_uri
from g2_policy_protocol.samples import (
    load_sample,
    prepare_images,
    save_sample,
    synthetic_context,
    synthetic_sample,
)


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
    from g2_policy_protocol.images import decode_images

    sample = synthetic_sample()
    # Pure red lets us check RGB/BGR independently of JPEG loss.
    sample["images"]["head_color"][:] = [255, 0, 0]
    meta = p.build_metadata(
        policy_hz=30,
        required_cameras=["head_color"],
        image_sizes={"head_color": [100, 100]},
        image_encoding=encoding,
        image_resize=resize,
    )
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
    sample = synthetic_sample()
    del sample["images"]["hand_right_color"]
    with pytest.raises(ValueError, match="missing required camera"):
        prepare_images(sample, p.build_metadata(policy_hz=30))


def test_empty_synthetic_context_does_not_read_or_prepare_sample(monkeypatch):
    from g2_policy_protocol import samples

    def unexpected(*args, **kwargs):
        pytest.fail("unrequested context must not prepare images")

    monkeypatch.setattr(samples, "prepare_images", unexpected)
    meta = p.build_metadata(policy_hz=30, required_cameras=[])
    assert synthetic_context({}, meta) == {}


def test_vector_synthetic_context_does_not_prepare_images(monkeypatch):
    from g2_policy_protocol import samples

    def unexpected(*args, **kwargs):
        pytest.fail("vector context must not prepare images")

    monkeypatch.setattr(samples, "prepare_images", unexpected)
    keys = ["start_pose", "command_joint", "pose_history", "joint_history", "base_to_parent"]
    meta = p.build_metadata(

        policy_hz=30,
        required_cameras=[],
        observation_spec=dict(pose=True, joint=True, prompt=True, history_steps=3, history_hz=30, context_keys=keys),
        requires_joint_state=True,
    )
    sample = synthetic_sample()
    context = synthetic_context(sample, meta)
    assert set(context) == set(keys)
    np.testing.assert_array_equal(context["start_pose"], sample["state"])
    np.testing.assert_array_equal(context["command_joint"], sample["joint_state"])
    np.testing.assert_array_equal(context["pose_history"], np.repeat(sample["state"][None], 3, axis=0))
    np.testing.assert_array_equal(context["joint_history"], np.repeat(sample["joint_state"][None], 3, axis=0))
    np.testing.assert_array_equal(context["base_to_parent"], np.eye(4))


def test_synthetic_image_history_uses_negotiated_cameras_and_size():
    meta = p.build_metadata(

        policy_hz=30,
        required_cameras=["head_color"],
        image_sizes={"head_color": [8, 10]},
        observation_spec=dict(
            pose=False, joint=False, prompt=False, history_steps=2, history_hz=30, context_keys=["image_history"]
        ),
    )
    sample = synthetic_sample()
    context = synthetic_context(sample, meta)
    assert set(context) == {"image_history"}
    assert set(context["image_history"]) == {"head_color"}
    history = context["image_history"]["head_color"]
    assert history.shape == (2, 8, 10, 3) and history.dtype == np.uint8
    np.testing.assert_array_equal(history[0], prepare_images(sample, meta)["head_color"])
    np.testing.assert_array_equal(history[0], history[1])


def test_url_preserves_path_and_query():
    assert connection_uri("wss://host/policy?token=abc&encoding=json", "msgpack") == (
        "wss://host/policy?token=abc&encoding=msgpack"
    )


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


def test_model_adapter_example_uses_shared_conventions():
    example = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/model_adapter.py"),
        run_name="__main__",
    )
    measured, actions = example["measured_pose"], example["actions"]
    np.testing.assert_allclose(example["rot6d"], [0, 1, 0, -1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(actions[:, 10], measured[10] + [0.01, 0.02])
    np.testing.assert_allclose(actions[:, 19], 0.5)
    np.testing.assert_array_equal(actions[:, :10], np.repeat(measured[None, :10], 2, axis=0))
    np.testing.assert_array_equal(actions[:, 13:19], np.repeat(measured[None, 13:19], 2, axis=0))
    parsed = p.parse_response(example["response"], request_id=1, metadata=example["metadata"])
    np.testing.assert_array_equal(parsed, actions)


@pytest.mark.parametrize("encoding", ["msgpack", "json"])
@pytest.mark.parametrize("outcome", ["success", "inference_error", "invalid_request"])
def test_policy_instance_request_response(encoding, outcome):
    from g2_policy_protocol import codec

    example = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/minimal_server.py"))
    meta = p.build_metadata(policy_hz=30, action_horizon=2, required_cameras=["head_color"])
    sample = synthetic_sample()
    seen = []

    class RecordingPolicy(example["MockPolicy"]):
        def infer(self, observation):
            seen.append(observation)
            if outcome == "inference_error":
                raise RuntimeError("model failed")
            return super().infer(observation)

    policy = RecordingPolicy(metadata=meta)
    request = p.build_request(
        request_id=42,
        prompt=sample["prompt"],
        state=sample["state"],
        images={"head_color": sample["images"]["head_color"]},
        metadata=meta,
    )
    if outcome == "invalid_request":
        del request["state"]

    class Connection:
        request = SimpleNamespace(path=f"/?encoding={encoding}")

        def __init__(self):
            self.sent = asyncio.Queue()
            self.incoming = asyncio.Queue()

        async def send(self, frame):
            await self.sent.put(codec.loads(frame, encoding=encoding))

        async def recv(self):
            return await self.incoming.get()

    async def run():
        from g2_policy_protocol.server import PolicyServer

        connection = Connection()
        server = PolicyServer(policy, metadata=meta)
        handler = asyncio.create_task(server._handler(connection))
        try:
            assert (await asyncio.wait_for(connection.sent.get(), 2))["type"] == "metadata"
            await connection.incoming.put(codec.dumps(request, encoding=encoding))
            response = await asyncio.wait_for(connection.sent.get(), 2)
            assert response["request_id"] == 42
            if outcome == "invalid_request":
                assert not seen
            else:
                assert seen[0]["prompt"] == sample["prompt"]
                np.testing.assert_array_equal(seen[0]["state"], sample["state"])
                np.testing.assert_array_equal(seen[0]["images"]["head_color"], sample["images"]["head_color"])
            if outcome == "success":
                actions = p.parse_response(response, request_id=42, metadata=meta)
                np.testing.assert_array_equal(actions, np.repeat(sample["state"][None], 2, axis=0))
            else:
                assert response["type"] == "error" and response["code"] == outcome
        finally:
            handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)
            server._worker.shutdown(wait=True)

    asyncio.run(run())
