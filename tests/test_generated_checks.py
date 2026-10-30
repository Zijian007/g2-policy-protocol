"""Schema/Python agreement and proof that changing the source changes checks."""
import json
import runpy
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from g2_policy_protocol import codec
from g2_policy_protocol import protocol as p

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def schema():
    return json.loads((ROOT / 'src/g2_policy_protocol/g2.schema.json').read_text())


def messages():
    meta = p.build_metadata(policy_hz=30)
    return {
        'metadata': meta,
        'obs': p.build_request(request_id=1, prompt='demo', state=np.zeros(20),
                               images=dict.fromkeys(p.CAMERA_IDS, np.zeros((2, 3, 3), np.uint8)), metadata=meta),
        'action': p.build_response(request_id=1, actions=np.zeros((2, 20)), metadata=meta),
        'error': p.build_error(request_id=1, message='failed'),
    }


@pytest.mark.parametrize('kind, changes', [
    ('metadata', {'required_cameras': ['unknown']}),
    ('metadata', {'required_cameras': ['head_color', 'head_color']}),
    ('metadata', {'image_sizes': {'unknown': [2, 3]}}),
    ('metadata', {'image_sizes': {'head_color': [2]}}),
    ('metadata', {'image_sizes': {'head_color': [2, 0]}}),
    ('metadata', {'requires_joint_state': 0}),
    ('metadata', {'extensions': []}),
    ('obs', {'request_id': True}),
    ('obs', {'request_id': -1}),
    ('obs', {'prompt': 1}),
    ('obs', {'rtc_enabled': 1}),
    ('obs', {'observation_time_ns': -1}),
    ('obs', {'images': []}),
    ('obs', {'images': {'unknown': None}}),
    ('action', {'server_timing': []}),
    ('action', {'server_timing': {'infer_ms': -1}}),
    ('action', {'policy_timing': {'infer_ms': True}}),
    ('action', {'rtc_info': []}),
    ('error', {'code': ''}),
    ('error', {'code': 7}),
    ('error', {'message': ''}),
    ('error', {'message': []}),
])
def test_static_rejections_agree_with_schema(schema, kind, changes):
    base = messages()
    payload = {**base[kind], **changes}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(json.loads(codec.json_dumps(payload)))
    with pytest.raises(ValueError):
        if kind == 'metadata':
            p.validate_metadata(payload)
        elif kind == 'obs':
            p.parse_request(payload, metadata=base['metadata'])
        else:
            p.parse_response(payload, request_id=1, metadata=base['metadata'])


def generate(schema):
    render = runpy.run_path(str(ROOT / 'tools/generate_protocol.py'))['render']
    namespace = {'__package__': 'g2_policy_protocol'}
    exec(compile(render(schema), '<generated-test>', 'exec'), namespace)
    return namespace


def test_changed_schema_bounds_enums_and_required_fields_change_generated_checks(schema):
    schema['$defs']['request_id']['minimum'] = 2
    schema['$defs']['metadata']['properties']['arm']['enum'] = ['both']
    schema['$defs']['obs']['required'].append('observation_time_ns')
    checks = generate(schema)
    with pytest.raises(ValueError):
        checks['validate_request_id'](1)
    checks['validate_request_id'](2)
    with pytest.raises(ValueError):
        checks['validate_metadata_fields'](p.build_metadata(policy_hz=30, arm='left'))
    obs = {**messages()['obs'], 'request_id': 2}
    with pytest.raises(ValueError, match='observation_time_ns'):
        checks['validate_obs_fields'](obs)
    checks['validate_obs_fields']({**obs, 'observation_time_ns': 1})


def test_new_unsupported_rules_fail_generation_instead_of_being_ignored(schema):
    schema['$defs']['obs']['properties']['prompt']['pattern'] = '^demo$'
    with pytest.raises(ValueError, match='unsupported static keywords'):
        generate(schema)


def test_inconsistent_layout_dimensions_fail_generation(schema):
    schema['$defs']['pose_values']['minItems'] = 21
    with pytest.raises(ValueError, match='dimensions disagree'):
        generate(schema)
