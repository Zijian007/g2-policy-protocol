"""Native-value checks and rigid calibration validation; no model/runtime dependencies."""

import math
from collections.abc import Mapping

import numpy as np


def finite_vector(value, dim, name):
    if value is None:
        raise ValueError(f"{name} is required")
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (dim,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite ({dim},), got {arr.shape}")
    return np.ascontiguousarray(arr)


def finite_action_rows(value, dim, name="actions"):
    if value is None:
        raise ValueError(f"{name} are required")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or not len(arr) or arr.shape[1] != dim or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite nonempty (T, {dim}), got {arr.shape}")
    return np.ascontiguousarray(arr)


def integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def number(value, name, minimum=None, exclusive_minimum=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value)
            or (minimum is not None and value < minimum)
            or (exclusive_minimum is not None and value <= exclusive_minimum)):
        raise ValueError(f"{name} must be a finite number within its bounds")
    return float(value)


def text(value, name, min_length):
    if not isinstance(value, str) or len(value) < min_length:
        raise ValueError(f"{name} must be text of length >= {min_length}")


def boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")


def one_of(value, allowed, name):
    if value not in allowed:
        raise ValueError(f"{name} has an unsupported value")


def object_fields(value, name, fields=None, required=()):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if fields is not None:
        missing = set(required) - value.keys()
        unknown = value.keys() - fields
        if missing or unknown:
            raise ValueError(f"{name}: missing={sorted(missing)}, unknown={sorted(unknown)}")


def array(value, name, minimum=None, maximum=None, unique=False):
    if (not isinstance(value, (list, tuple))
            or (minimum is not None and len(value) < minimum)
            or (maximum is not None and len(value) > maximum)):
        raise ValueError(f"{name} must be an array within its length bounds")
    if unique:
        try:
            repeated = len(set(value)) != len(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain unique scalar values") from exc
        if repeated:
            raise ValueError(f"{name} must contain unique values")


def validate_rigid_transform(matrix, name):
    if (
        not isinstance(matrix, list)
        or len(matrix) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
    ):
        raise ValueError(f"{name} must be a 4x4 rigid transform")
    if any(type(v) not in (int, float) or not math.isfinite(v) for row in matrix for v in row):
        raise ValueError(f"{name} must contain finite numbers")
    if any(abs(a - b) > 1e-6 for a, b in zip(matrix[3], [0, 0, 0, 1], strict=True)):
        raise ValueError(f"{name} must have homogeneous last row [0,0,0,1]")
    for i in range(3):
        for j in range(3):
            if abs(sum(matrix[k][i] * matrix[k][j] for k in range(3)) - int(i == j)) > 1e-5:
                raise ValueError(f"{name} rotation must be orthonormal")
    r = matrix
    determinant = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(determinant - 1) > 1e-5:
        raise ValueError(f"{name} rotation determinant must be +1")
