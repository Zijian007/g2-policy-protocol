import json
from importlib.resources import files
from pathlib import Path
import runpy

import jsonschema
import numpy as np
import pytest

from g2_policy_protocol import codec, protocol as p


def metadata(**kwargs):
    return p.build_metadata(policy_hz=30, **kwargs)


def request(meta=None, **kwargs):
    meta = meta or metadata()
    image = np.zeros((3, 5, 3), dtype=np.uint8) if meta['image_encoding'] == 'raw' else b'jpeg'
    args = dict(request_id=42, prompt='demo', state=np.arange(20, dtype=np.float32),
                images=dict.fromkeys(meta['required_cameras'], image), metadata=meta)
    args.update(kwargs)
    return p.build_request(**args)


def test_generated_constants_are_exactly_from_the_only_schema():
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/generate_protocol.py'))
    assert (Path(p.__file__).with_name('_generated.py')).read_text() == module['render']()
    assert p.POSE_DIM == len(p.POSE_FIELDS) == 20
    assert p.JOINT_DIM == len(p.JOINT_FIELDS) == 16
    assert p.POSE_FIELDS[9] == 'left_gripper' and p.POSE_FIELDS[10] == 'right_x'


@pytest.mark.parametrize('encoding', ['msgpack', 'json'])
@pytest.mark.parametrize('image_encoding', ['raw', 'jpeg'])
def test_every_message_agrees_with_json_schema(encoding, image_encoding):
    schema = json.loads(files('g2_policy_protocol').joinpath('g2.schema.json').read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    meta = metadata(image_encoding=image_encoding, rtc_supported=True)
    messages = [meta, request(meta, rtc_enabled=True, inference_delay=2, prev_chunk_left_over=np.zeros((1, 20))),
                p.build_response(request_id=42, actions=np.ones((2, 20))),
                p.build_error(request_id=42, message='model failed')]
    for message in messages:
        validator.validate(json.loads(codec.json_dumps(message)))
        decoded = codec.loads(codec.dumps(message, encoding=encoding), encoding=encoding)
        if message['type'] == 'metadata':
            p.validate_metadata(decoded, policy_hz=30)
        elif message['type'] == 'obs':
            p.parse_request(decoded, metadata=meta)
        elif message['type'] == 'action':
            p.parse_response(decoded, request_id=42)
        else:
            with pytest.raises(RuntimeError, match='model failed'):
                p.parse_response(decoded, request_id=42)


@pytest.mark.parametrize('changes', [{'protocol': 'g2.policy.v1'}, {'policy_hz': 0}, {'policy_hz': float('nan')},
    {'required_cameras': ['unknown']}, {'image_encoding': 'png'}, {'image_sizes': {'head_color': [0, 1]}},
    {'execution_steps': 51}, {'arm': 'bad'}, {'rtc_supported': 1}])
def test_incompatible_capabilities_are_rejected(changes):
    with pytest.raises(ValueError):
        p.validate_metadata({**metadata(), **changes})


@pytest.mark.parametrize('changes', [{'state': np.zeros(16)}, {'state': np.full(20, np.nan)}, {'request_id': True},
    {'prompt': ''}, {'images': {}}, {'rtc_enabled': True}, {'inference_delay': 1},
    {'prev_chunk_left_over': np.zeros((1, 20))}, {'observation_time_ns': -1}, {'unexpected': 1}])
def test_invalid_observations_never_reach_model(changes):
    meta = metadata()
    raw = request(meta)
    with pytest.raises((ValueError, TypeError)):
        p.parse_request({**raw, **changes}, metadata=meta)


def test_single_arm_required_cameras_and_optional_joint_state():
    meta = metadata(arm='left', requires_joint_state=True)
    with pytest.raises(ValueError, match='joint_state'):
        request(meta)
    raw = request(meta, joint_state=np.zeros(16))
    raw['images']['hand_right_color'] = None
    parsed = p.parse_request(raw, metadata=meta)
    assert parsed['state'].shape == (20,) and parsed['joint_state'].shape == (16,)
    assert set(parsed['images']) == {'head_color', 'hand_left_color'}


def test_request_ids_and_model_errors_cannot_be_confused():
    result = p.build_response(request_id=4, actions=np.zeros((1, 20)))
    with pytest.raises(ValueError, match='request_id'):
        p.parse_response(result, request_id=5)
    with pytest.raises(RuntimeError, match='broken'):
        p.parse_response(p.build_error(request_id=5, message='broken'), request_id=5)


@pytest.mark.parametrize('actions', [np.zeros(20), np.zeros((0, 20)), np.zeros((2, 10)), np.zeros((2, 32)), np.full((1, 20), np.inf)])
def test_model_adapters_must_return_finite_absolute_pose20(actions):
    with pytest.raises(ValueError):
        p.build_response(request_id=42, actions=actions)


def test_encoding_is_fixed_for_the_entire_connection():
    for path, expected in [('/', 'msgpack'), ('/?encoding=json', 'json'), ('/policy?x=1&encoding=msgpack', 'msgpack')]:
        assert codec.encoding_from_path(path) == expected
    for path in ('/?encoding=other', '/?encoding=json&encoding=msgpack'):
        with pytest.raises(ValueError):
            codec.encoding_from_path(path)
    with pytest.raises(ValueError):
        codec.loads('{}', encoding='msgpack')
    with pytest.raises(ValueError):
        codec.loads(b'{}', encoding='json')


def test_decoded_jpeg_dimensions_are_checked_before_inference():
    meta = metadata(image_encoding='jpeg', image_sizes={'head_color':[224,224]})
    images = dict.fromkeys(meta['required_cameras'],np.zeros((8,12,3),np.uint8))
    with pytest.raises(ValueError,match='size disagrees'):
        p.validate_images(images,metadata=meta,decoded=True)


def test_metadata_numpy_scalars_still_emit_schema_valid_json():
    meta = p.build_metadata(policy_hz=np.float32(30),action_horizon=np.int64(10),
                            image_sizes={'head_color':[np.int64(224),np.int64(224)]})
    schema = json.loads(files('g2_policy_protocol').joinpath('g2.schema.json').read_text())
    jsonschema.Draft202012Validator(schema).validate(json.loads(codec.json_dumps(meta)))


@pytest.mark.parametrize('kwargs', [
    {'request_id': True},
    {'server_timing': {'infer_ms': -1}},
    {'policy_timing': {'infer_ms': float('nan')}},
])
def test_response_builder_keeps_validation(kwargs):
    args = dict(request_id=42, actions=np.zeros((2, 20)))
    with pytest.raises(ValueError):
        p.build_response(**{**args, **kwargs})


def test_response_builder_normalizes_arrays_and_numpy_scalars():
    response = p.build_response(request_id=np.int64(42), actions=[[1] * 20],
                                server_timing={'infer_ms': np.float32(2)})
    assert type(response['request_id']) is int
    assert response['actions'].dtype == np.float32
    assert response['actions'].shape == (1, 20)
    assert type(response['server_timing']['infer_ms']) is float


@pytest.mark.parametrize('field', [
    'schema_version', 'state_layout', 'action_layout', 'frame', 'rotation',
    'action_semantics', 'encoding', 'policy_type',
])
def test_redundant_metadata_fields_are_not_part_of_v2(field):
    meta = metadata()
    assert field not in meta
    schema = json.loads(files('g2_policy_protocol').joinpath('g2.schema.json').read_text())
    invalid = {**meta, field: 'obsolete'}
    with pytest.raises(ValueError, match='unknown'):
        p.validate_metadata(invalid)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(invalid)


def test_fixed_semantics_are_generated_from_schema_instead_of_sent():
    assert p.PROTOCOL == 'g2.policy.v2'
    assert (p.FRAME, p.ROTATION, p.ACTION_SEMANTICS) == ('base_link', 'rot6d', 'absolute')
    assert p.ENCODING_VALUES == ('msgpack', 'json')
    p.validate_metadata(metadata(extensions={'policy_type': 'my_custom_model'}))
