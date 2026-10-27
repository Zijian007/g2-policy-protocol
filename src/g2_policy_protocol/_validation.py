"""Small native-value checks used by schema-generated code; no schema interpreter."""
from collections.abc import Mapping
import math

import numpy as np


def finite_vector(value, dim, name):
    if value is None:
        raise ValueError(f'{name} is required')
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (dim,) or not np.isfinite(arr).all():
        raise ValueError(f'{name} must be finite ({dim},), got {arr.shape}')
    return np.ascontiguousarray(arr)


def finite_action_rows(value, dim, name='actions'):
    if value is None:
        raise ValueError(f'{name} are required')
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or not len(arr) or arr.shape[1] != dim or not np.isfinite(arr).all():
        raise ValueError(f'{name} must be finite nonempty (T, {dim}), got {arr.shape}')
    return np.ascontiguousarray(arr)


def integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def number(value, name, minimum=None, exclusive_minimum=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value)
            or (minimum is not None and value < minimum)
            or (exclusive_minimum is not None and value <= exclusive_minimum)):
        raise ValueError(f'{name} must be a finite number within its bounds')
    return float(value)


def text(value, name, min_length):
    if not isinstance(value, str) or len(value) < min_length:
        raise ValueError(f'{name} must be text of length >= {min_length}')


def object_fields(value, name, fields=None, required=()):
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be an object')
    if fields is not None:
        missing = set(required) - value.keys()
        unknown = value.keys() - fields
        if missing or unknown:
            raise ValueError(f'{name}: missing={sorted(missing)}, unknown={sorted(unknown)}')


def array(value, name, minimum=None, maximum=None, unique=False):
    if (not isinstance(value, (list, tuple))
            or (minimum is not None and len(value) < minimum)
            or (maximum is not None and len(value) > maximum)):
        raise ValueError(f'{name} must be an array within its length bounds')
    if unique:
        try:
            repeated = len(set(value)) != len(value)
        except TypeError as exc:
            raise ValueError(f'{name} must contain unique scalar values') from exc
        if repeated:
            raise ValueError(f'{name} must contain unique values')
