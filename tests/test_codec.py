
import msgpack
import numpy as np
import pytest
from g2_policy_protocol import codec


@pytest.mark.parametrize('array', [np.arange(12, dtype=np.float32).reshape(3, 4), np.arange(12, dtype='>i2')[::2], np.zeros((0, 20)), np.array(3, dtype=np.int64), np.ones((2, 3), dtype=np.bool_)])
def test_numpy_roundtrip_is_writable_and_exact(array):
    result = codec.unpackb(codec.packb({'a': array, 'jpeg': b'\xff\xd8payload\xff\xd9'}))
    np.testing.assert_array_equal(result['a'], array)
    assert result['a'].dtype == array.dtype
    assert result['a'].flags.writeable
    assert result['jpeg'] == b'\xff\xd8payload\xff\xd9'


@pytest.mark.parametrize('value', [np.float32(1.5), np.int64(7), np.bool_(True)])
def test_numpy_scalar_roundtrip(value):
    result = codec.unpackb(codec.packb(value))
    assert type(result) is type(value) and result == value


def test_streaming_codec_uses_identical_array_markers():
    unpacker = codec.Unpacker()
    array = np.arange(20, dtype=np.float32)
    packed = codec.Packer().pack({'a': array})
    unpacker.feed(packed[:3])
    assert list(unpacker) == []
    unpacker.feed(packed[3:])
    np.testing.assert_array_equal(next(unpacker)['a'], array)


@pytest.mark.parametrize('dtype', ['O', 'c8', 'V8'])
def test_unsafe_or_unsupported_array_dtypes_rejected_both_directions(dtype):
    with pytest.raises(ValueError):
        codec.packb(np.zeros(1, dtype=dtype))
    packet = {b'__ndarray__': True, b'dtype': dtype, b'shape': [1], b'data': bytes(8)}
    with pytest.raises(ValueError):
        codec.unpackb(msgpack.packb(packet))


@pytest.mark.parametrize('shape,data', [([-1], b''), ([True], b'1234'), ([2], b'1234'), ([1], b'12345678')])
def test_invalid_array_metadata_rejected(shape, data):
    packet = {b'__ndarray__': True, b'dtype': '<f4', b'shape': shape, b'data': data}
    with pytest.raises(ValueError):
        codec.unpackb(msgpack.packb(packet))


@pytest.mark.parametrize('image', [np.zeros((2, 3, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.uint8), np.zeros((0, 3, 3), dtype=np.uint8), b'jpeg'])
def test_raw_image_representation_is_explicit(image):
    with pytest.raises(ValueError):
        codec.validate_image(image, encoding='raw')


def test_jpeg_is_opaque_and_unknown_encoding_is_rejected():
    jpeg = b'compressed payload is decoded by the adapter'
    assert codec.validate_image(jpeg, encoding='jpeg') is jpeg
    with pytest.raises(ValueError):
        codec.validate_image(b'', encoding='jpeg')
    with pytest.raises(ValueError):
        codec.validate_image(jpeg, encoding='png')


@pytest.mark.parametrize('value', [np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(8, dtype='>i2')[::2], np.empty((2, 0, 3), dtype=np.uint8), np.array(3, dtype=np.int64), np.float32(1.5), np.float64(1.5), np.uint64(2**64-1), np.bool_(True), np.array([True, False]), np.array(['左', '右']), np.array([b'a', b'b'])])
def test_json_numpy_roundtrip_preserves_dtype_shape_and_values(value):
    result = codec.json_loads(codec.json_dumps({'value': value}))['value']
    assert type(result) is type(value)
    assert result.dtype == value.dtype
    np.testing.assert_array_equal(result, value)
    if isinstance(value, np.ndarray):
        assert result.shape == value.shape and result.flags.writeable


def test_json_uses_explicit_base64_and_array_tags():
    import json
    packet = {'image': b'\xff\xd8jpeg\xff\xd9', 'state': np.array([1., 2.], dtype=np.float32), 'prompt': '拿饮料'}
    text = codec.json_dumps(packet)
    raw = json.loads(text)
    assert raw['image'] == {'$g2': 'bytes', 'encoding': 'base64', 'data': '/9hqcGVn/9k='}
    assert raw['state'] == {'$g2': 'ndarray', 'dtype': '<f4', 'shape': [2], 'data': [1., 2.]}
    assert raw['prompt'] == '拿饮料'
    restored = codec.json_loads(text.encode())
    assert restored['image'] == packet['image']
    assert codec.json_loads('{"state":[1,2],"image":null}') == {'state': [1, 2], 'image': None}


@pytest.mark.parametrize('value', [float('nan'), float('inf'), np.float32(float('inf')), {'a': np.array([float('nan')])}])
def test_json_encoder_rejects_nonfinite_values(value):
    with pytest.raises(ValueError):
        codec.json_dumps(value)


@pytest.mark.parametrize('text', ['NaN', 'Infinity', '-Infinity', '1e400', '{"state":[NaN]}', '{"$g2":"ndarray","dtype":"<f4","shape":[1],"data":[1e100]}'])
def test_json_decoder_rejects_nonfinite_values(text):
    with pytest.raises(ValueError):
        codec.json_loads(text)


@pytest.mark.parametrize('packet', [
    {'$g2': 'unknown'},
    {'$g2': 'bytes', 'encoding': 'base64', 'data': '%%%?'},
    {'$g2': 'bytes', 'encoding': 'hex', 'data': '00'},
    {'$g2': 'ndarray', 'dtype': 'O', 'shape': [1], 'data': [0]},
    {'$g2': 'ndarray', 'dtype': '<c8', 'shape': [1], 'data': [0]},
    {'$g2': 'ndarray', 'dtype': '<f4', 'shape': [2], 'data': [0]},
    {'$g2': 'ndarray', 'dtype': '<f4', 'shape': [-1], 'data': []},
    {'$g2': 'ndarray', 'dtype': 'u1', 'shape': [1], 'data': [256]},
    {'$g2': 'ndarray', 'dtype': 'u1', 'shape': [1], 'data': [-1]},
    {'$g2': 'ndarray', 'dtype': '<f4', 'shape': [1], 'data': ['1.0']},
    {'$g2': 'scalar', 'dtype': '<f4', 'data': [1.]},
    {'$g2': 'scalar', 'dtype': '<f4', 'data': 1., 'extra': True},
])
def test_json_rejects_malformed_typed_values(packet):
    import json
    with pytest.raises((ValueError, TypeError)):
        codec.json_loads(json.dumps(packet))


@pytest.mark.parametrize('packet,error', [({'$g2': 'ordinary-user-field'}, ValueError), ({1: 'key'}, TypeError), (np.array([object()], dtype=object), ValueError)])
def test_json_encoder_rejects_reserved_tags_and_unsupported_types(packet, error):
    with pytest.raises(error):
        codec.json_dumps(packet)


@pytest.mark.parametrize('encoding', ['raw', 'jpeg'])
def test_json_and_msgpack_feed_the_same_protocol_parser(encoding):
    from g2_policy_protocol import protocol as wire
    meta = wire.build_metadata(policy_hz=30, arm='left',
                               image_encoding=encoding, rtc_supported=True)
    image = np.zeros((3, 5, 3), dtype=np.uint8) if encoding == 'raw' else b'jpeg'
    request = wire.build_request(request_id=42, prompt='demo', state=np.arange(20, dtype=np.float32),
                                 images=dict.fromkeys(meta['required_cameras'], image), metadata=meta,
                                 rtc_enabled=True, inference_delay=2, prev_chunk_left_over=np.ones((2, 20)))
    first = wire.parse_request(codec.json_loads(codec.json_dumps(request)), metadata=meta)
    second = wire.parse_request(codec.unpackb(codec.packb(request)), metadata=meta)
    np.testing.assert_array_equal(first['state'], second['state'])
    np.testing.assert_array_equal(first['prev_chunk_left_over'], second['prev_chunk_left_over'])
    response = wire.build_response(request_id=42, actions=np.ones((2, 20)), metadata=meta)
    np.testing.assert_array_equal(wire.parse_response(codec.json_loads(codec.json_dumps(response)), request_id=42, metadata=meta),
                                  wire.parse_response(codec.unpackb(codec.packb(response)), request_id=42, metadata=meta))


def test_json_rejects_extended_float_without_recursing():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip('platform has no extended float')
    with pytest.raises(ValueError, match='unsupported JSON dtype'):
        codec.json_dumps(np.longdouble(1))
    with pytest.raises(ValueError, match='unsupported JSON dtype'):
        codec.json_dumps(np.ones(2, dtype=np.longdouble))
