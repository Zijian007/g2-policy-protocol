"""Shared contract must work without any model framework or robot runtime."""
import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from g2_policy_protocol import codec
from g2_policy_protocol import protocol as p

QUANTITIES = ('pose_target', 'joint_position', 'joint_velocity', 'joint_torque')


def metadata(quantity='pose_target', **kwargs):
    return p.build_metadata(policy_hz=15, required_cameras=[],
        action_spec=p.build_action_spec(quantity),
        observation_spec=dict(pose=False, joint=False, prompt=False, history_steps=1,
                              history_hz=30, context_keys=[]), **kwargs)


@pytest.mark.parametrize('quantity', QUANTITIES)
@pytest.mark.parametrize('encoding', ['json', 'msgpack'])
def test_all_typed_actions_and_optional_observations_round_trip(quantity, encoding):
    meta = metadata(quantity, rtc_supported=True, reset_supported=True)
    p.validate_metadata(codec.loads(codec.dumps(meta, encoding=encoding), encoding=encoding))
    dim = meta['action_spec']['dim']
    request = p.build_request(request_id=7, state=None, prompt='', images={}, metadata=meta,
                            rtc_enabled=True, inference_delay=1, prev_chunk_left_over=np.ones((3, dim)))
    parsed = p.parse_request(codec.loads(codec.dumps(request, encoding=encoding), encoding=encoding), metadata=meta)
    assert 'state' not in parsed and parsed['prompt'] == ''
    assert parsed['prev_chunk_left_over'].shape == (3, dim)
    response = p.build_response(request_id=7, actions=np.ones((4, dim)), metadata=meta)
    assert p.parse_response(codec.loads(codec.dumps(response, encoding=encoding), encoding=encoding),
                            request_id=7, metadata=meta).shape == (4, dim)
    with pytest.raises(ValueError):
        p.parse_response({**response, 'actions': np.ones((4, dim + 1))}, request_id=7, metadata=meta)
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / 'src/g2_policy_protocol/g2.schema.json').read_text())
    for message in (meta, request, response, p.build_reset(request_id=8, metadata=meta), p.build_reset_ack(request_id=8)):
        jsonschema.validate(json.loads(codec.json_dumps(message)), schema)


@pytest.mark.parametrize('key,value', [('dim', 32), ('frame', 'world'), ('ee_frame', 'tcp'),
                                     ('representation', 'component_delta'), ('units', ['rad'] * 20)])
def test_model_internal_or_wrong_wire_contracts_are_rejected(key, value):
    meta = metadata()
    meta['action_spec'][key] = value
    with pytest.raises(ValueError):
        p.validate_metadata(meta)


def test_history_refs_timestamps_and_reset_validation():
    meta = metadata()
    meta['required_cameras'] = ['hand_left_color']  # head camera is not universally required.
    meta['image_sizes'] = {'hand_left_color': [2, 3]}
    meta['observation_spec'].update(pose=True, history_steps=2,
                                  context_keys=['pose_history', 'image_history', 'object_pose'])
    p.validate_metadata(meta)
    image = np.zeros((2, 3, 3), np.uint8)
    context = dict(pose_history=np.zeros((2, 20)), object_pose=np.zeros(20),
                   image_history={'hand_left_color': np.zeros((2, 2, 3, 3), np.uint8)})
    request = dict(type='obs', request_id=1, state=np.zeros(20), images={'hand_left_color': image},
                   observation_time_ns=1000000000, context=context)
    parsed = p.parse_request(request, metadata=meta)
    assert parsed['context']['pose_history'].shape == (2, 20)
    for mutation in ({'observation_time_ns': -1}, {'context': {}},
                     {'context': {**context, 'pose_history': np.zeros((3, 20))}},
                     {'context': {**context, 'object_pose': np.full(20, np.nan)}}):
        with pytest.raises(ValueError):
            p.parse_request({**request, **mutation}, metadata=meta)
    with pytest.raises(ValueError, match='observation_time_ns'):
        p.parse_request({k: v for k, v in request.items() if k != 'observation_time_ns'}, metadata=meta)
    with pytest.raises(ValueError, match='reset'):
        p.build_reset(request_id=3, metadata=meta)
    p.parse_reset_ack(p.build_reset_ack(request_id=3), request_id=3)
    with pytest.raises(ValueError):
        p.parse_reset_ack(p.build_reset_ack(request_id=3), request_id=4)


def test_schema_changes_drive_validation():
    root = Path(__file__).resolve().parents[1]
    render = runpy.run_path(str(root / 'tools/generate_protocol.py'))['render']
    schema = json.loads((root / 'src/g2_policy_protocol/g2.schema.json').read_text())
    assert render(schema) == (root / 'src/g2_policy_protocol/_generated.py').read_text()
    changed = copy.deepcopy(schema)
    changed['$defs']['metadata']['properties']['action_spec']['properties']['dim']['minimum'] = 100
    ns = {'__package__': 'g2_policy_protocol'}
    exec(compile(render(changed), '<v3-generated>', 'exec'), ns)
    with pytest.raises(ValueError, match='dim'):
        ns['validate_metadata_fields'](metadata())


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
'''], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_unknown_response_mode_is_rejected_during_handshake():
    with pytest.raises(ValueError, match="response_mode"):
        p.build_metadata(policy_hz=30, extensions={"response_mode": "full_chunck"})


def test_removed_twist_is_rejected_by_action_spec_metadata_and_json_schema():
    with pytest.raises(ValueError, match='unsupported control_quantity'):
        p.build_action_spec('eef_twist')
    meta = metadata()
    meta['action_spec']['control_quantity'] = 'eef_twist'
    with pytest.raises(ValueError, match='unsupported value'):
        p.validate_metadata(meta)
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / 'src/g2_policy_protocol/g2.schema.json').read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(json.loads(codec.json_dumps(meta)), schema)
