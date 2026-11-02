"""Image resizing and JPEG helpers for adapters and diagnostics; the protocol core never decodes images."""

from io import BytesIO

import numpy as np
from PIL import Image

from . import protocol as p


class ImageDecodeError(ValueError):
    """Invalid image input, reported as invalid_request by PolicyServer."""


def resize_rgb(image, size, mode="letterbox"):
    """Resize an HWC RGB image to (height, width), preserving its dtype/range.

    ``letterbox`` keeps the aspect ratio and centers it on black padding;
    ``resize`` stretches. Both match metadata ``image_resize``.
    """
    image = np.asarray(image)
    height, width = size
    if image.shape == (height, width, 3):
        return image.copy()
    if mode not in ("letterbox", "resize"):
        raise ValueError(f"unsupported resize mode: {mode!r}")
    floating = np.issubdtype(image.dtype, np.floating)
    pixels = np.clip(np.rint(image * 255) if floating else image, 0, 255).astype(np.uint8)
    source = Image.fromarray(pixels)
    if mode == "resize":
        result = source.resize((width, height), Image.Resampling.BILINEAR)
    else:
        scale = min(width / source.width, height / source.height)
        resized = source.resize((max(1, int(source.width * scale)), max(1, int(source.height * scale))),
                                Image.Resampling.BILINEAR)
        result = Image.new("RGB", (width, height))
        result.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    result = np.asarray(result).copy()
    return (result.astype(image.dtype) / 255) if floating else result.astype(image.dtype)


def encode_jpeg(image):
    p.validate_image(image, encoding="raw")
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def decode_images(images, metadata):
    decoded = {}
    for camera, image in images.items():
        if metadata["image_encoding"] == "jpeg":
            image = _decode_jpeg(camera, image)
        decoded[camera] = image
    try:
        return p.validate_images(decoded, metadata=metadata, decoded=True)
    except ValueError as exc:
        raise ImageDecodeError(str(exc)) from exc


def decode_image_history(history, metadata):
    """Decode validated wire histories to RGB uint8 (L,H,W,3), checking frame sizes."""
    if metadata["image_encoding"] == "raw":
        return history
    decoded = {}
    for camera, frames in history.items():
        images = [_decode_jpeg(f"image_history.{camera}[{index}]", frame)
                  for index, frame in enumerate(frames)]
        size = metadata["image_sizes"].get(camera, images[0].shape[:2])
        if any(image.shape[:2] != tuple(size) for image in images):
            raise ImageDecodeError(f"image_history.{camera} size disagrees with metadata or other frames")
        decoded[camera] = np.stack(images)
    return decoded


def _decode_jpeg(name, data):
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format != "JPEG":
                raise ImageDecodeError(f"{name}: expected JPEG payload")
            return np.asarray(source.convert("RGB")).copy()
    except OSError as exc:
        raise ImageDecodeError(f"{name}: failed to decode JPEG") from exc
