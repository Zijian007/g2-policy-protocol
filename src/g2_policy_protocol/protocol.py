"""Framework-independent G2 messages. Structure/constants come from the schemas.

Decode with codec before parsing. Handshake metadata is checked once; each request
then uses that metadata to pick v2 pose20 or v3 action_spec / observation_spec.
"""
from collections.abc import Mapping

import numpy as np

from ._generated import *  # Generated public v2 contract constants.
from ._validation import finite_action_rows, finite_vector
from .codec import validate_image
from . import v3


def cameras_for_arm(arm):
    if arm == 'both':
        return CAMERA_IDS
    if arm in SIDES:
        return (CAMERA_IDS[0], CAMERA_IDS[SIDES.index(arm) + 1])
    raise ValueError(f'unsupported arm: {arm!r}')


def is_v3(metadata):
    return isinstance(metadata, Mapping) and metadata.get('protocol') == v3.PROTOCOL


def action_contract(metadata=None):
    """Wire control type. v2 and missing metadata are canonical pose20."""
    return metadata['action_spec'] if is_v3(metadata) else v3.action_spec()


def observation_contract(metadata):
    return metadata['observation_spec'] if is_v3(metadata) else dict(
        pose=True, joint=metadata.get('requires_joint_state', False), prompt=True,
        history_steps=1, history_hz=metadata.get('policy_hz', 30), context_keys=[])


def parse_actions(value, metadata=None):
    """Finite nonempty (T, D) actions. D and horizon come from the handshake."""
    if is_v3(metadata):
        return v3.actions(value, metadata)
    return finite_action_rows(value, POSE_DIM)


def parse_rtc(payload, metadata):
    """Shared RTC flags. v3 also bounds delay against horizon and prefix length."""
    enabled = payload.get('rtc_enabled', False)
    delay = int(payload.get('inference_delay', 0))
    raw_prefix = payload.get('prev_chunk_left_over')
    if enabled and not metadata['rtc_supported']:
        raise ValueError('server does not support RTC')
    if not enabled and (delay or raw_prefix is not None):
        raise ValueError('RTC fields require rtc_enabled=true')
    prefix = None if raw_prefix is None else parse_actions(raw_prefix, metadata)
    if is_v3(metadata):
        if delay >= metadata['action_horizon']:
            raise ValueError('inference_delay exceeds action horizon')
        if prefix is not None and delay > len(prefix):
            raise ValueError('inference_delay exceeds RTC prefix')
    return enabled, delay, prefix


def build_metadata(*, policy_hz, action_horizon=50, arm='both',
                   image_encoding='raw', image_resize='resize',
                   image_sizes=None, required_cameras=None, rtc_supported=False,
                   requires_joint_state=False, execution_steps=None, extensions=None, version=2,
                   action_spec=None, observation_spec=None, timing=None, reset_supported=False):
    meta = dict(type=METADATA_TYPE, protocol=PROTOCOL, policy_hz=policy_hz,
                action_horizon=action_horizon, arm=arm, image_encoding=image_encoding,
                image_resize=image_resize, image_sizes={} if image_sizes is None else dict(image_sizes),
                required_cameras=list(cameras_for_arm(arm) if required_cameras is None else required_cameras),
                rtc_supported=rtc_supported, requires_joint_state=requires_joint_state)
    if execution_steps is not None:
        meta['execution_steps'] = execution_steps
    if extensions is not None:
        meta['extensions'] = dict(extensions)
    if version == 3:
        meta['protocol'] = v3.PROTOCOL
        meta['action_spec'] = v3.action_spec() if action_spec is None else dict(action_spec)
        meta['observation_spec'] = dict(pose=True, joint=requires_joint_state, prompt=True,
                                        history_steps=1, history_hz=policy_hz, context_keys=[]) if observation_spec is None else dict(observation_spec)
        meta['timing'] = dict(action_dt=1 / policy_hz, first_target_offset_s=0) if timing is None else dict(timing)
        meta['reset_supported'] = reset_supported
    elif version != 2:
        raise ValueError('unsupported protocol version')
    validate_metadata(meta)
    meta['policy_hz'] = float(meta['policy_hz'])
    meta['action_horizon'] = int(meta['action_horizon'])
    meta['required_cameras'] = [str(key) for key in meta['required_cameras']]
    meta['image_sizes'] = {str(key): [int(value) for value in size] for key, size in meta['image_sizes'].items()}
    if execution_steps is not None:
        meta['execution_steps'] = int(execution_steps)
    return meta


def validate_metadata(meta, *, policy_hz=None, require_rtc=False):
    (v3.validate_metadata if is_v3(meta) else validate_metadata_fields)(meta)
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
                  observation_time_ns=None, context=None):
    payload = dict(type=OBS_TYPE, request_id=request_id, prompt=prompt, state=state, images=images,
                   rtc_enabled=rtc_enabled, inference_delay=inference_delay,
                   prev_chunk_left_over=prev_chunk_left_over)
    if is_v3(metadata) and state is None:
        payload.pop('state')
    if context is not None:
        payload['context'] = context
    if joint_state is not None:
        payload['joint_state'] = joint_state
    if observation_time_ns is not None:
        payload['observation_time_ns'] = observation_time_ns
    return parse_request(payload, metadata=metadata)


def parse_request(payload, *, metadata):
    if is_v3(metadata):
        return v3.parse_request(payload, metadata)
    validate_obs_fields(payload)
    result = dict(payload)
    result['request_id'] = int(payload['request_id'])
    prompt = payload['prompt']
    if not prompt.strip():
        raise ValueError('prompt must be nonempty text')
    result['state'] = finite_vector(payload['state'], POSE_DIM, 'state')
    if 'joint_state' in payload:
        result['joint_state'] = finite_vector(payload['joint_state'], JOINT_DIM, 'joint_state')
    elif metadata['requires_joint_state']:
        raise ValueError('server requires joint_state')
    result['images'] = validate_images(payload['images'], metadata=metadata)
    rtc_enabled, inference_delay, prefix = parse_rtc(payload, metadata)
    result['rtc_enabled'] = rtc_enabled
    result['inference_delay'] = inference_delay
    result['prev_chunk_left_over'] = prefix
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


def build_response(*, request_id, actions, server_timing=None, policy_timing=None, rtc_info=None, metadata=None):
    result = dict(type=ACTION_TYPE, request_id=request_id, actions=actions)
    for key, block in (('server_timing', server_timing), ('policy_timing', policy_timing), ('rtc_info', rtc_info)):
        if block is not None:
            result[key] = dict(block)
    result['actions'] = parse_response(result, request_id=request_id, metadata=metadata)
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


def parse_response(payload, *, request_id, metadata=None):
    """Parse action/error. Pass handshake metadata so v3 uses action_spec.dim.

    Omitting metadata keeps the v2 pose20 check.
    """
    if not isinstance(payload, Mapping):
        raise ValueError('response must be an object')
    kind = payload.get('type')
    if kind == ACTION_TYPE:
        (v3.fields.validate_action_fields if is_v3(metadata) else validate_action_fields)(payload)
    elif kind == ERROR_TYPE:
        validate_error_fields(payload)
    else:
        raise ValueError('expected action or error response')
    validate_request_id(request_id)
    if payload['request_id'] != request_id:
        raise ValueError('response request_id does not match observation')
    if kind == ERROR_TYPE:
        raise RuntimeError(f'policy server {payload["code"]}: {payload["message"]}')
    return parse_actions(payload['actions'], metadata)


def build_reset(*, request_id, metadata):
    if not is_v3(metadata) or not metadata['reset_supported']:
        raise ValueError('server does not support episode reset')
    return v3.reset_message(request_id)


def parse_reset_ack(payload, *, request_id):
    v3.fields.validate_reset_ack_fields(payload)
    if payload['request_id'] != request_id:
        raise ValueError('reset acknowledgement request_id mismatch')
    return dict(payload)
