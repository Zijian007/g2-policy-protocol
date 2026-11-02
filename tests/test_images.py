import asyncio
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from conftest import FakeConnection
from g2_policy_protocol import protocol
from g2_policy_protocol.images import ImageDecodeError, decode_image_history, decode_images, encode_jpeg
from g2_policy_protocol.server import PolicyServer


@pytest.mark.parametrize('image_encoding', ['raw', 'jpeg'])
def test_decode_history_preserves_order_and_rgb(image_encoding):
    frames = np.zeros((2, 8, 10, 3), np.uint8)
    frames[0, ..., 0] = 255
    frames[1, ..., 2] = 255
    history = {'head_color': [encode_jpeg(frame) for frame in frames] if image_encoding == 'jpeg' else frames}
    metadata = protocol.build_metadata(
        policy_hz=30, required_cameras=['head_color'], image_encoding=image_encoding,
        image_sizes={'head_color': [8, 10]})
    decoded = decode_image_history(history, metadata)
    result = decoded['head_color']
    assert result.shape == (2, 8, 10, 3) and result.dtype == np.uint8
    np.testing.assert_allclose(result, frames, atol=2)
    if image_encoding == 'raw':
        assert decoded is history


def test_history_without_declared_size_requires_consistent_frames():
    metadata = protocol.build_metadata(policy_hz=30, required_cameras=['head_color'], image_encoding='jpeg')
    frame = encode_jpeg(np.zeros((8, 10, 3), np.uint8))
    assert decode_image_history({'head_color': [frame, frame]}, metadata)['head_color'].shape == (2, 8, 10, 3)
    other = encode_jpeg(np.zeros((9, 10, 3), np.uint8))
    with pytest.raises(ImageDecodeError, match='size disagrees'):
        decode_image_history({'head_color': [frame, other]}, metadata)


@pytest.mark.parametrize('history', [False, True])
@pytest.mark.parametrize('failure', ['corrupt', 'size', 'format'])
def test_invalid_images_return_invalid_request_before_model(history, failure):
    metadata = protocol.build_metadata(
        policy_hz=30, required_cameras=['head_color'], image_encoding='jpeg', image_sizes={'head_color': [8, 10]},
        observation_spec=dict(pose=False, joint=False, prompt=False, history_steps=2 if history else 1,
                              history_hz=10, context_keys=['image_history'] if history else []))
    valid = encode_jpeg(np.zeros((8, 10, 3), np.uint8))
    if failure == 'corrupt':
        invalid = b'broken jpeg'
    elif failure == 'size':
        invalid = encode_jpeg(np.zeros((9, 10, 3), np.uint8))
    else:
        buffer = BytesIO()
        Image.new('RGB', (10, 8)).save(buffer, format='PNG')
        invalid = buffer.getvalue()
    request = protocol.build_request(
        request_id=7, prompt='', state=None, metadata=metadata, observation_time_ns=10**9,
        images=None if history else {'head_color': invalid},
        context={'image_history': {'head_color': [valid, invalid]}} if history else {})

    class Backend:
        def infer(self, request):
            if history:
                decode_image_history(request['context']['image_history'], metadata)
            else:
                decode_images(request['images'], metadata)
            pytest.fail('invalid images reached the model')

    async def run():
        connection = FakeConnection()
        server = PolicyServer(Backend(), metadata=metadata)
        handler = asyncio.create_task(server._handler(connection))
        try:
            await connection.get()
            await connection.put(request)
            response = await connection.get()
            assert response['request_id'] == 7
            assert response['code'] == 'invalid_request'
            assert 'head_color' in response['message']
        finally:
            handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)
            server._worker.shutdown(wait=True)

    asyncio.run(run())
