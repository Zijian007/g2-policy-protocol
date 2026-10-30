"""MessagePack and strict JSON for the same messages; no image libraries.

JPEG is transported as opaque bytes. Encoding/decoding and resizing belong to
the caller, before build_request / after parse_request respectively.
"""
import base64
import functools
import json
import math
from collections.abc import Mapping

import msgpack
import numpy as np

from ._generated import (
    ENCODING_VALUES,
    IMAGE_CHANNELS,
    IMAGE_DTYPE,
    IMAGE_MIN_HW,
    IMAGE_NDIM,
)

_RAW_IMAGE_DTYPE = np.dtype(IMAGE_DTYPE)


def _dtype(value) -> np.dtype:
    dtype = np.dtype(value)
    if dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype: {dtype}")
    return dtype


def _shape(value):
    if not isinstance(value, (list, tuple)) or any(type(n) is not int or n < 0 for n in value):
        raise ValueError("invalid ndarray shape")
    return tuple(value)


def pack_array(obj):
    if isinstance(obj, np.ndarray):
        dtype = _dtype(obj.dtype)
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        dtype = _dtype(obj.dtype)
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": dtype.str}
    raise TypeError(f"cannot encode {type(obj).__name__}")


def unpack_array(obj):
    if b"__ndarray__" in obj:
        dtype = _dtype(obj[b"dtype"])
        shape = _shape(obj[b"shape"])
        data = obj[b"data"]
        if not isinstance(data, bytes) or len(data) != math.prod(shape) * dtype.itemsize:
            raise ValueError("ndarray byte length disagrees with dtype/shape")
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    if b"__npgeneric__" in obj:
        return _dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def _json_prepare(value):
    # Walk containers before json.dumps so np.float64 (also a Python float)
    # retains its dtype, and user dictionaries cannot collide with typed tags.
    if isinstance(value, (np.ndarray, np.generic)):
        dtype = _json_dtype(value.dtype)
        scalar = isinstance(value, np.generic)
        result = {"$g2": "scalar" if scalar else "ndarray", "dtype": dtype.str,
                  "data": _json_prepare(value.item() if scalar else value.tolist())}
        if not scalar:
            result["shape"] = list(value.shape)
        return result
    if isinstance(value, bytes):
        return {"$g2": "bytes", "encoding": "base64",
                "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        if "$g2" in value:
            raise ValueError("$g2 is reserved for JSON typed values")
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON message keys must be strings")
        return {key: _json_prepare(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_prepare(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _json_dtype(value):
    dtype = _dtype(value)
    if dtype.kind not in "biufUS" or (dtype.kind == "f" and dtype.itemsize > 8):
        raise ValueError(f"unsupported JSON dtype: {dtype}")
    return dtype


def _json_restore(obj):
    if "$g2" not in obj:
        return obj
    tag = obj["$g2"]
    keys = {"$g2", "encoding", "data"} if tag == "bytes" else {"$g2", "dtype", "data"}
    if tag == "ndarray":
        keys.add("shape")
    if tag not in ("bytes", "ndarray", "scalar") or obj.keys() != keys:
        raise ValueError("invalid JSON typed value")
    if tag == "bytes":
        if obj["encoding"] != "base64" or not isinstance(obj["data"], str):
            raise ValueError("bytes require a Base64 string")
        return base64.b64decode(obj["data"], validate=True)
    dtype = _json_dtype(obj["dtype"])
    shape = _shape(obj["shape"]) if tag == "ndarray" else ()
    raw = np.asarray(obj["data"])
    if raw.shape != shape:
        # Nested JSON lists lose trailing dimensions for empty arrays.
        if raw.size != 0 or math.prod(shape) != 0:
            raise ValueError("JSON data disagrees with dtype/shape")
    if raw.size:
        if dtype.kind in "biuf" and raw.dtype.kind not in "biuf":
            raise ValueError("numeric array requires numeric JSON data")
        if dtype.kind in "iu" and (raw.dtype.kind not in "iu" or
                np.any(raw < np.iinfo(dtype).min) or np.any(raw > np.iinfo(dtype).max)):
            raise ValueError("integer JSON data is outside dtype range")
        if dtype.kind == "b" and raw.dtype.kind != "b":
            raise ValueError("boolean array requires JSON booleans")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(obj["data"], dtype=dtype).reshape(shape)
    if dtype.kind == "f" and not np.isfinite(result).all():
        raise ValueError("JSON array contains non-finite values")
    return result[()] if tag == "scalar" else result


def _finite_json_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    return value


def _reject_json_constant(text):
    raise ValueError(f"invalid JSON number: {text}")


def json_dumps(value) -> str:
    """Encode text frames: typed arrays/scalars, Base64 bytes, finite numbers."""
    return json.dumps(_json_prepare(value), allow_nan=False, ensure_ascii=False, separators=(",", ":"))


def json_loads(value: str | bytes | bytearray):
    """Restore the same arrays/bytes consumed by both protocol parsers."""
    return json.loads(value, object_hook=_json_restore,
                      parse_float=_finite_json_float, parse_constant=_reject_json_constant)


def validate_image(value, *, encoding: str):
    """Validate representation, without allocating pixel copies or decoding JPEG."""
    if encoding == "jpeg":
        if not isinstance(value, bytes) or not value:
            raise ValueError("jpeg image must be nonempty bytes")
        return value
    if encoding != "raw":
        raise ValueError(f"unsupported image encoding: {encoding!r}")
    if (not isinstance(value, np.ndarray) or value.dtype != _RAW_IMAGE_DTYPE
            or value.ndim != IMAGE_NDIM or value.shape[-1] != IMAGE_CHANNELS
            or value.shape[0] < IMAGE_MIN_HW[0] or value.shape[1] < IMAGE_MIN_HW[1]):
        raise ValueError("raw image must be nonempty RGB uint8 HxWx3")
    return value


def dumps(message, *, encoding='msgpack'):
    """Encode one connection frame; the encoding is fixed at connect time."""
    if encoding == 'msgpack':
        return packb(message)
    if encoding == 'json':
        return json_dumps(message)
    raise ValueError(f'unsupported encoding: {encoding!r}')


def loads(frame, *, encoding='msgpack'):
    if encoding == 'msgpack' and isinstance(frame, bytes):
        return unpackb(frame)
    if encoding == 'json' and isinstance(frame, str):
        return json_loads(frame)
    raise ValueError('frame type does not match connection encoding')


def encoding_from_path(path):
    """Servers use ?encoding=json; existing no-query connections use msgpack."""
    from urllib.parse import parse_qs, urlsplit
    values = parse_qs(urlsplit(path).query).get('encoding', ['msgpack'])
    if len(values) != 1 or values[0] not in ENCODING_VALUES:
        raise ValueError('encoding must be json or msgpack')
    return values[0]
