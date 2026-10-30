"""Portable, pickle-free observation fixtures for offline server integration."""

from pathlib import Path

import numpy as np
from PIL import Image

from . import protocol as p
from .images import encode_jpeg


def synthetic_sample():
    state = np.zeros(p.POSE_DIM, dtype=np.float32)
    state[[3, 7, 13, 17]] = 1
    state[:3] = [0.35, 0.2, 0.5]
    state[10:13] = [0.35, -0.2, 0.5]
    y, x = np.indices((240, 320))
    images = {}
    for index, camera in enumerate(p.CAMERA_IDS):
        image = np.empty((240, 320, 3), dtype=np.uint8)
        image[..., 0] = x % 256
        image[..., 1] = y % 256
        image[..., 2] = 40 + index * 70
        images[camera] = image
    return dict(
        state=state, joint_state=np.zeros(p.JOINT_DIM, dtype=np.float32),
        prompt="保持当前位姿", provenance="synthetic; not recorded from a robot",
        images=images,
    )


def validate_sample(sample):
    for key, shape in (("state", (p.POSE_DIM,)), ("joint_state", (p.JOINT_DIM,))):
        if key not in sample and key == "joint_state":
            continue
        value = np.asarray(sample[key], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"sample {key} must be finite {shape}")
    for key in ("prompt", "provenance"):
        if not isinstance(sample[key], str) or not sample[key].strip():
            raise ValueError(f"sample {key} must be nonempty text")
    if not sample["images"]:
        raise ValueError("sample must contain at least one image")
    for camera, image in sample["images"].items():
        if camera not in p.CAMERA_IDS:
            raise ValueError(f"unknown camera: {camera}")
        p.validate_image(image, encoding="raw")
    return sample


def save_sample(path, sample):
    validate_sample(sample)
    fields = {key: sample[key] for key in ("state", "prompt", "provenance")}
    if "joint_state" in sample:
        fields["joint_state"] = sample["joint_state"]
    fields.update(sample["images"])
    # File handle prevents NumPy from silently appending a second .npz suffix.
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **fields)


def load_sample(path):
    with np.load(path, allow_pickle=False) as data:
        allowed = {"state", "joint_state", "prompt", "provenance", *p.CAMERA_IDS}
        if set(data.files) - allowed:
            raise ValueError(f"unknown sample fields: {set(data.files) - allowed}")
        sample = {key: data[key].copy() for key in ("state", "joint_state") if key in data}
        for key in ("prompt", "provenance"):
            value = data[key]
            if value.shape != () or value.dtype.kind != "U":
                raise ValueError(f"sample {key} must be a scalar Unicode string")
            sample[key] = value.item()
        sample["images"] = {key: data[key].copy() for key in p.CAMERA_IDS if key in data}
    return validate_sample(sample)


def prepare_images(sample, metadata):
    """Prepare only required cameras: resize/letterbox, then JPEG when negotiated."""
    result = {}
    bilinear = Image.Resampling.BILINEAR
    for camera in metadata["required_cameras"]:
        if camera not in sample["images"]:
            raise ValueError(f"sample missing required camera: {camera}")
        image = sample["images"][camera]
        target = metadata["image_sizes"].get(camera)
        if target is not None and tuple(image.shape[:2]) != tuple(target):
            height, width = target
            source = Image.fromarray(image)
            if metadata["image_resize"] == "letterbox":
                ratio = min(height / image.shape[0], width / image.shape[1])
                h, w = max(1, int(image.shape[0] * ratio)), max(1, int(image.shape[1] * ratio))
                resized = np.asarray(source.resize((w, h), resample=bilinear))
                image = np.zeros((height, width, 3), dtype=np.uint8)
                y, x = (height - h) // 2, (width - w) // 2
                image[y:y + h, x:x + w] = resized
            else:
                image = np.asarray(source.resize((width, height), resample=bilinear))
        result[camera] = encode_jpeg(image) if metadata["image_encoding"] == "jpeg" else image
    return result


def synthetic_context(sample, metadata):
    """Repeat fixture values for protocol diagnostics only; never a robot history provider."""
    spec = metadata["observation_spec"]
    result = {}
    for key in spec["context_keys"]:
        if key == "image_history":
            raw = prepare_images(sample, {**metadata, "image_encoding": "raw"})
            result[key] = {
                camera: np.repeat(image[None], spec["history_steps"], axis=0)
                for camera, image in raw.items()
            }
        elif key == "base_to_parent":
            result[key] = np.eye(4)
        else:
            if "pose" in key:
                value = sample["state"]
            else:
                value = sample["joint_state"] if "joint_state" in sample else np.zeros(p.JOINT_DIM, np.float32)
            result[key] = np.tile(value, (spec["history_steps"], 1)) if key.endswith("_history") else value
    return result

