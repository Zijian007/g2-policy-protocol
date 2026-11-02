"""g2.policy.v3 messages: round trips, static/dynamic rejection and framework independence."""

import json
import subprocess
import sys

import numpy as np
import pytest

from g2_policy_protocol import codec
from g2_policy_protocol import protocol as p


def metadata(**kwargs):
    return p.build_metadata(policy_hz=30, **kwargs)


def request(meta, **kwargs):
    image = np.zeros((3, 5, 3), dtype=np.uint8) if meta['image_encoding'] == 'raw' else b'jpeg'
    args = dict(request_id=42, prompt='demo', state=np.arange(20, dtype=np.float32),
                images=dict.fromkeys(meta['required_cameras'], image), metadata=meta)
    args.update(kwargs)
    return p.build_request(**args)


def roundtrip(message, encoding):
    return codec.loads(codec.dumps(message, encoding=encoding), encoding=encoding)


def test_vector_layout_constants():
    assert p.POSE_DIM == len(p.POSE_FIELDS) == 20
    assert p.JOINT_DIM == len(p.JOINT_FIELDS) == 16
    assert p.POSE_FIELDS[9] == 'left_gripper' and p.POSE_FIELDS[10] == 'right_x'
    assert p.build_action_spec('joint_position')['fields'] == list(p.JOINT_FIELDS)


@pytest.mark.parametrize('quantity', ['pose_target', 'joint_position'])
@pytest.mark.parametrize('encoding', ['json', 'msgpack'])
@pytest.mark.parametrize('image_encoding', ['raw', 'jpeg'])
def test_every_message_round_trips(quantity, encoding, image_encoding):
    meta = metadata(action_spec=p.build_action_spec(quantity), image_encoding=image_encoding,
                    rtc_supported=True, reset_supported=True, requires_joint_state=True,
                    extensions={'policy_type': 'custom'})
    p.validate_metadata(roundtrip(meta, encoding), policy_hz=30)
    dim = meta['action_spec']['dim']
    obs = request(meta, joint_state=np.zeros(16), rtc_enabled=True, inference_delay=1,
                  prev_chunk_left_over=np.ones((3, dim)), observation_time_ns=7)
    parsed = p.parse_request(roundtrip(obs, encoding), metadata=meta)
    assert parsed['prev_chunk_left_over'].shape == (3, dim)
    np.testing.assert_array_equal(parsed['state'], obs['state'])
    response = p.build_response(request_id=42, actions=np.ones((4, dim)), metadata=meta)
    assert p.parse_response(roundtrip(response, encoding), request_id=42, metadata=meta).shape == (4, dim)
    with pytest.raises(RuntimeError, match='model failed'):
        p.parse_response(roundtrip(p.build_error(request_id=42, message='model failed'), encoding),
                         request_id=42, metadata=meta)
    assert roundtrip(p.build_reset(request_id=8, metadata=meta), encoding)['type'] == 'reset'
    p.parse_reset_ack(roundtrip(p.build_reset_ack(request_id=8), encoding), request_id=8)


def test_optional_observations_can_be_omitted():
    meta = metadata(required_cameras=[], observation_spec=dict(
        pose=False, joint=False, prompt=False, history_steps=1, history_hz=30, context_keys=[]))
    parsed = p.parse_request(request(meta, state=None, prompt=''), metadata=meta)
    assert 'state' not in parsed and parsed['prompt'] == '' and parsed['images'] == {}


def _with_action_spec(**changes):
    return {'action_spec': {**p.build_action_spec(), **changes}}


@pytest.mark.parametrize('changes', [
    {'protocol': 'g2.policy.v2'},
    {'policy_hz': 0},
    {'policy_hz': float('nan')},
    {'arm': 'bad'},
    {'required_cameras': ['unknown']},
    {'required_cameras': ['head_color', 'head_color']},
    {'image_encoding': 'png'},
    {'image_sizes': {'unknown': [2, 3]}},
    {'image_sizes': {'head_color': [2]}},
    {'image_sizes': {'head_color': [2, 0]}},
    {'rtc_supported': 1},
    {'requires_joint_state': 0},
    {'extensions': []},
    {'extensions': {'response_mode': 'full_chunck'}},
    {'timing': {'action_dt': 1 / 30}},
    {'timing': {'action_dt': 0.1, 'first_target_offset_s': 0}},
    {'execution_steps': 51},
    {'schema_version': 'obsolete'},
    _with_action_spec(control_quantity='eef_twist'),
    _with_action_spec(dim=32),
    _with_action_spec(frame='world'),
    _with_action_spec(ee_frame='tcp'),
    _with_action_spec(representation='component_delta'),
    _with_action_spec(units=['rad'] * 20),
])
def test_invalid_metadata_is_rejected(changes):
    meta = metadata()
    with pytest.raises(ValueError):
        p.validate_metadata({**meta, **changes})


@pytest.mark.parametrize('changes', [
    {'protocol': 'g2.policy.v4'},
    {'new_capability': True},
    {'timing': {'action_dt': 1 / 30}},
    _with_action_spec(units=['rad'] * 20),
])
def test_version_drift_errors_name_the_likely_cause(changes):
    with pytest.raises(ValueError, match='different g2-policy-protocol versions'):
        p.validate_metadata({**metadata(), **changes})
    response = p.build_response(request_id=1, actions=np.zeros((1, 20)), metadata=metadata())
    with pytest.raises(ValueError, match='different g2-policy-protocol versions'):
        p.parse_response({**response, 'new_field': 1}, request_id=1, metadata=metadata())


def test_ordinary_value_errors_do_not_blame_versions():
    with pytest.raises(ValueError) as error:
        p.validate_metadata({**metadata(), 'image_sizes': {'head_color': [2, 0]}})
    assert 'versions' not in str(error.value)


def test_context_keys_must_be_known_and_history_must_be_complete():
    meta = metadata()
    for keys in (['unknown'], []):
        spec = {**meta['observation_spec'], 'history_steps': 2, 'context_keys': keys}
        with pytest.raises(ValueError):
            p.validate_metadata({**meta, 'observation_spec': spec})


@pytest.mark.parametrize('changes', [
    {'state': np.zeros(16)},
    {'state': np.full(20, np.nan)},
    {'request_id': True},
    {'request_id': -1},
    {'prompt': ''},
    {'prompt': 1},
    {'images': {}},
    {'images': []},
    {'images': {'unknown': None}},
    {'rtc_enabled': True},
    {'rtc_enabled': 1},
    {'inference_delay': 1},
    {'prev_chunk_left_over': np.zeros((1, 20))},
    {'observation_time_ns': -1},
    {'unexpected': 1},
])
def test_invalid_observations_never_reach_model(changes):
    meta = metadata()
    with pytest.raises(ValueError):
        p.parse_request({**request(meta), **changes}, metadata=meta)


@pytest.mark.parametrize('changes', [
    {'actions': np.zeros(20)},
    {'actions': np.zeros((0, 20))},
    {'actions': np.zeros((2, 10))},
    {'actions': np.zeros((2, 32))},
    {'actions': np.zeros((51, 20))},
    {'actions': np.full((1, 20), np.inf)},
    {'request_id': True},
    {'server_timing': []},
    {'server_timing': {'infer_ms': -1}},
    {'policy_timing': {'infer_ms': float('nan')}},
    {'policy_timing': {'infer_ms': True}},
    {'rtc_info': []},
])
def test_invalid_actions_are_rejected(changes):
    meta = metadata()
    response = p.build_response(request_id=42, actions=np.zeros((2, 20)), metadata=meta)
    with pytest.raises(ValueError):
        p.parse_response({**response, **changes}, request_id=42, metadata=meta)


@pytest.mark.parametrize('changes', [{'code': ''}, {'code': 7}, {'message': ''}, {'message': []}])
def test_invalid_errors_are_rejected(changes):
    error = p.build_error(request_id=42, message='failed')
    with pytest.raises(ValueError):
        p.parse_response({**error, **changes}, request_id=42, metadata=metadata())


def test_request_ids_and_model_errors_cannot_be_confused():
    meta = metadata()
    result = p.build_response(request_id=4, actions=np.zeros((1, 20)), metadata=meta)
    with pytest.raises(ValueError, match='request_id'):
        p.parse_response(result, request_id=5, metadata=meta)
    with pytest.raises(RuntimeError, match='broken'):
        p.parse_response(p.build_error(request_id=5, message='broken'), request_id=5, metadata=meta)
    with pytest.raises(ValueError):
        p.parse_reset_ack(p.build_reset_ack(request_id=3), request_id=4)


def test_single_arm_required_cameras_and_optional_joint_state():
    meta = metadata(arm='left', requires_joint_state=True)
    with pytest.raises(ValueError, match='joint_state'):
        request(meta)
    raw = request(meta, joint_state=np.zeros(16))
    raw['images']['hand_right_color'] = None
    parsed = p.parse_request(raw, metadata=meta)
    assert parsed['state'].shape == (20,) and parsed['joint_state'].shape == (16,)
    assert set(parsed['images']) == {'head_color', 'hand_left_color'}


def test_history_context_and_timestamps():
    meta = metadata(required_cameras=['hand_left_color'], image_sizes={'hand_left_color': [2, 3]})
    meta['observation_spec'].update(history_steps=2, context_keys=['pose_history', 'image_history', 'object_pose'])
    p.validate_metadata(meta)
    context = dict(pose_history=np.zeros((2, 20)), object_pose=np.zeros(20),
                   image_history={'hand_left_color': np.zeros((2, 2, 3, 3), np.uint8)})
    raw = dict(type='obs', request_id=1, prompt='demo', state=np.zeros(20),
               images={'hand_left_color': np.zeros((2, 3, 3), np.uint8)},
               observation_time_ns=10**9, context=context)
    assert p.parse_request(raw, metadata=meta)['context']['pose_history'].shape == (2, 20)
    for mutation in ({'context': {}},
                     {'context': {**context, 'pose_history': np.zeros((3, 20))}},
                     {'context': {**context, 'object_pose': np.full(20, np.nan)}}):
        with pytest.raises(ValueError):
            p.parse_request({**raw, **mutation}, metadata=meta)
    with pytest.raises(ValueError, match='observation_time_ns'):
        p.parse_request({k: v for k, v in raw.items() if k != 'observation_time_ns'}, metadata=meta)
    with pytest.raises(ValueError, match='reset'):
        p.build_reset(request_id=3, metadata=meta)


def test_decoded_jpeg_dimensions_are_checked_before_inference():
    meta = metadata(image_encoding='jpeg', image_sizes={'head_color': [224, 224]})
    images = dict.fromkeys(meta['required_cameras'], np.zeros((8, 12, 3), np.uint8))
    with pytest.raises(ValueError, match='size disagrees'):
        p.validate_images(images, metadata=meta, decoded=True)


@pytest.mark.parametrize('encoding', ['json', 'msgpack'])
@pytest.mark.parametrize('image_encoding', ['raw', 'jpeg'])
def test_history_roundtrip_without_duplicate_current_images(encoding, image_encoding):
    meta = metadata(required_cameras=['head_color'], image_encoding=image_encoding, observation_spec=dict(
        pose=False, joint=False, prompt=False, history_steps=2, history_hz=10, context_keys=['image_history']))
    frames = [b'first JPEG', b'current JPEG'] if image_encoding == 'jpeg' else np.zeros((2, 3, 5, 3), np.uint8)
    history = {'head_color': frames}
    payload = request(meta, images=None, context={'image_history': history}, observation_time_ns=10**9)
    parsed = p.parse_request(roundtrip(payload, encoding), metadata=meta)
    assert 'images' not in payload and 'images' not in parsed
    np.testing.assert_array_equal(parsed['context']['image_history']['head_color'], frames)
    # New servers still accept older clients sending the current frame separately.
    legacy = {**payload, 'images': {'head_color': frames[-1]}}
    assert 'images' in p.parse_request(roundtrip(legacy, encoding), metadata=meta)


def test_images_cannot_be_omitted_without_history():
    meta = metadata()
    with pytest.raises(ValueError, match='requires images or context.image_history'):
        request(meta, images=None)


@pytest.mark.parametrize('image_encoding,frames', [
    ('jpeg', [b'one']), ('jpeg', b'not a list'), ('jpeg', [b'', b'jpeg']),
    ('jpeg', [b'jpeg', 'text']), ('jpeg', np.zeros((2, 2, 3, 3), np.uint8)),
    ('raw', [b'first', b'current']),
])
def test_image_history_representation_mismatch_explains_versions(image_encoding, frames):
    meta = metadata(required_cameras=['head_color'], image_encoding=image_encoding, observation_spec=dict(
        pose=False, joint=False, prompt=False, history_steps=2, history_hz=10, context_keys=['image_history']))
    with pytest.raises(ValueError, match='different g2-policy-protocol versions'):
        request(meta, images=None, context={'image_history': {'head_color': frames}}, observation_time_ns=10**9)


def test_builders_normalize_numpy_scalars():
    meta = p.build_metadata(policy_hz=np.float32(30), action_horizon=np.int64(10),
                            image_sizes={'head_color': [np.int64(224), np.int64(224)]})
    decoded = json.loads(codec.json_dumps(meta))
    p.validate_metadata(decoded)
    assert type(decoded['action_horizon']) is int and decoded['image_sizes']['head_color'] == [224, 224]
    response = p.build_response(request_id=np.int64(42), actions=[[1] * 20], metadata=meta,
                                server_timing={'infer_ms': np.float32(2)})
    assert type(response['request_id']) is int and type(response['server_timing']['infer_ms']) is float
    assert response['actions'].dtype == np.float32 and response['actions'].shape == (1, 20)


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


def test_shared_tools_import_without_any_policy_framework():
    result = subprocess.run([sys.executable, '-c', '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('openpi', 'torch', 'jax', 'wholebody', 'lerobot'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from g2_policy_protocol.data import encoding, dataset, contract
from g2_policy_protocol import protocol, validation
assert encoding.ActionCodec and dataset.TemporalDataset
assert contract.G2.pose.names == protocol.POSE_FIELDS
assert contract.G2.joint.names == protocol.JOINT_FIELDS
'''], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


def test_context_parsers_cover_every_context_key():
    from g2_policy_protocol._constants import CONTEXT_KEYS

    covered = {*p._CONTEXT_VECTORS, *p._CONTEXT_HISTORIES, 'base_to_parent', 'image_history'}
    assert covered == set(CONTEXT_KEYS)
