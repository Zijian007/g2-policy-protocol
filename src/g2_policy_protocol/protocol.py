"""Framework-independent G2 messages. Structure/constants come from g2.schema.json.

Decode with codec before parsing. Metadata is checked once on connection; per
request checks cover finite arrays, dynamic image requirements and RTC semantics.
"""
from collections.abc import Mapping

import numpy as np

from ._generated import *  # Generated public contract constants.
from .codec import validate_image


def _vector(value, dim, name):
    if value is None:
        raise ValueError(f'{name} is required')
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (dim,) or not np.isfinite(arr).all():
        raise ValueError(f'{name} must be finite ({dim},), got {arr.shape}')
    return np.ascontiguousarray(arr)


def _actions(value):
    if value is None:
        raise ValueError('actions are required')
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != ACTION_NDIM or len(arr) < ACTION_MIN_ROWS or arr.shape[1] != POSE_DIM or not np.isfinite(arr).all():
        raise ValueError(f'actions must be finite nonempty (T, {POSE_DIM}), got {arr.shape}')
    return np.ascontiguousarray(arr)


def cameras_for_arm(arm):
    if arm == 'both':
        return CAMERA_IDS
    if arm in SIDES:
        return (CAMERA_IDS[0], CAMERA_IDS[SIDES.index(arm) + 1])
    raise ValueError(f'unsupported arm: {arm!r}')


def build_metadata(*, policy_hz, action_horizon=50, arm='both',
                   image_encoding='raw', image_resize='resize',
                   image_sizes=None, required_cameras=None, rtc_supported=False,
                   requires_joint_state=False, execution_steps=None, extensions=None):
    meta = dict(type=METADATA_TYPE, protocol=PROTOCOL, policy_hz=policy_hz,
                action_horizon=action_horizon, arm=arm, image_encoding=image_encoding,
                image_resize=image_resize, image_sizes={} if image_sizes is None else dict(image_sizes),
                required_cameras=list(cameras_for_arm(arm) if required_cameras is None else required_cameras),
                rtc_supported=rtc_supported, requires_joint_state=requires_joint_state)
    if execution_steps is not None:
        meta['execution_steps'] = execution_steps
    if extensions is not None:
        meta['extensions'] = dict(extensions)
    validate_metadata(meta)
    meta['policy_hz'] = float(meta['policy_hz'])
    meta['action_horizon'] = int(meta['action_horizon'])
    meta['required_cameras'] = [str(key) for key in meta['required_cameras']]
    meta['image_sizes'] = {str(key): [int(value) for value in size] for key, size in meta['image_sizes'].items()}
    if execution_steps is not None:
        meta['execution_steps'] = int(execution_steps)
    return meta


def validate_metadata(meta, *, policy_hz=None, require_rtc=False):
    validate_metadata_fields(meta)
    if policy_hz is not None:
        validate_policy_hz(policy_hz)
        if not np.isclose(meta['policy_hz'], policy_hz):
            raise ValueError('server policy_hz disagrees with configured policy_hz')
    if meta.get('execution_steps', meta['action_horizon']) > meta['action_horizon']:
        raise ValueError('execution_steps exceeds action_horizon')
    if require_rtc and not meta['rtc_supported']:
        raise ValueError('server does not support RTC')


def build_request(*, request_id, prompt, state, images, metadata, joint_state=None,
                  rtc_enabled=False, inference_delay=0, prev_chunk_left_over=None,
                  observation_time_ns=None):
    payload = dict(type=OBS_TYPE, request_id=request_id, prompt=prompt, state=state, images=images,
                   rtc_enabled=rtc_enabled, inference_delay=inference_delay,
                   prev_chunk_left_over=prev_chunk_left_over)
    if joint_state is not None:
        payload['joint_state'] = joint_state
    if observation_time_ns is not None:
        payload['observation_time_ns'] = observation_time_ns
    return parse_request(payload, metadata=metadata)


def parse_request(payload, *, metadata):
    validate_obs_fields(payload)
    result = dict(payload)
    result['request_id'] = int(payload['request_id'])
    prompt = payload['prompt']
    if not prompt.strip():
        raise ValueError('prompt must be nonempty text')
    result['state'] = _vector(payload['state'], POSE_DIM, 'state')
    if 'joint_state' in payload:
        result['joint_state'] = _vector(payload['joint_state'], JOINT_DIM, 'joint_state')
    elif metadata['requires_joint_state']:
        raise ValueError('server requires joint_state')
    result['images'] = validate_images(payload['images'], metadata=metadata)
    rtc = payload.get('rtc_enabled', False)
    if rtc and not metadata['rtc_supported']:
        raise ValueError('invalid or unsupported RTC request')
    result['rtc_enabled'] = rtc
    result['inference_delay'] = int(payload.get('inference_delay', 0))
    prefix = payload.get('prev_chunk_left_over')
    result['prev_chunk_left_over'] = None if prefix is None else _actions(prefix)
    if not rtc and (result['inference_delay'] or prefix is not None):
        raise ValueError('RTC fields require rtc_enabled=true')
    if 'observation_time_ns' in payload:
        result['observation_time_ns'] = int(payload['observation_time_ns'])
    return result


def validate_images(images, *, metadata, decoded=False):
    """Check negotiated camera requirements before/after adapter JPEG decoding."""
    validate_images_fields(images)
    if any(images.get(key) is None for key in metadata['required_cameras']):
        raise ValueError('missing required cameras')
    result = {}
    for key, image in images.items():
        if image is None:
            continue
        image = validate_image(image, encoding='raw' if decoded else metadata['image_encoding'])
        expected = metadata['image_sizes'].get(key)
        if isinstance(image, np.ndarray) and expected is not None and tuple(image.shape[:2]) != tuple(expected):
            raise ValueError(f'{key} size disagrees with metadata image_sizes')
        result[key] = image
    return result


def build_response(*, request_id, actions, server_timing=None, policy_timing=None, rtc_info=None):
    result = dict(type=ACTION_TYPE, request_id=request_id, actions=actions)
    for key, block in (('server_timing', server_timing), ('policy_timing', policy_timing), ('rtc_info', rtc_info)):
        if block is not None:
            result[key] = dict(block)
    result['actions'] = parse_response(result, request_id=request_id)
    result['request_id'] = int(request_id)
    for key in ('server_timing', 'policy_timing'):
        if key in result:
            result[key] = {name: float(value) for name, value in result[key].items()}
    return result


def build_error(*, request_id, message, code='inference_error'):
    result = dict(type=ERROR_TYPE, request_id=request_id, code=code, message=message)
    validate_error_fields(result)
    result['request_id'] = int(request_id)
    return result


def parse_response(payload, *, request_id):
    if not isinstance(payload, Mapping):
        raise ValueError('response must be an object')
    kind = payload.get('type')
    if kind == ACTION_TYPE:
        validate_action_fields(payload)
    elif kind == ERROR_TYPE:
        validate_error_fields(payload)
    else:
        raise ValueError('expected action or error response')
    validate_request_id(request_id)
    if payload['request_id'] != request_id:
        raise ValueError('response request_id does not match observation')
    if kind == ERROR_TYPE:
        raise RuntimeError(f'policy server {payload["code"]}: {payload["message"]}')
    return _actions(payload['actions'])
