"""Outgoing image preparation and incoming wire-to-backend RGB conversion.

The client calls prepare_images; the server worker calls decode_observation.
Protocol parsing validates representations without decoding JPEG.
"""

from collections.abc import Mapping
from io import BytesIO
from typing import cast

import numpy as np
from PIL import Image

from .. import protocol as p
from ..types import (
    Metadata,
    Observation,
    ObservationRequest,
    RGBImage,
    WireImage,
    WireImageHistory,
)


class ImageDecodeError(ValueError):
    """Invalid image input, reported as invalid_request by PolicyServer."""


def prepare_images(images: Mapping[str, RGBImage] | None, metadata: Metadata) -> dict[str, WireImage]:
    """Prepare required RGB uint8 current frames for the negotiated wire format."""
    images = {} if images is None else images
    result = {}
    for camera in metadata["required_cameras"]:
        if images.get(camera) is None:
            raise ValueError(f"missing required camera {camera}")
        image = p.validate_image(images[camera], encoding="raw")
        size = metadata["image_sizes"].get(camera)
        if size is not None:
            image = resize_rgb(image, size, mode=metadata["image_resize"])
        result[camera] = encode_jpeg(image) if metadata["image_encoding"] == "jpeg" else image
    return result


def decode_observation(request: ObservationRequest, metadata: Metadata) -> Observation:
    """Convert a validated wire request to backend RGB data, without mutation.

    images holds current HWC frames, or LHWC histories when supplied. Histories
    already contain the current frame, so a duplicate current image is not
    decoded. Other context values remain references/history arrays unchanged.
    Call from the model worker, not the network event loop.
    """
    context = dict(request["context"])
    history = context.pop("image_history", None)
    images = decode_images(request["images"], metadata) if history is None else decode_image_history(history, metadata)
    return cast(Observation, {**request, "images": images, "context": context})


def resize_rgb(image: np.ndarray, size: tuple[int, int] | list[int], mode: str = "letterbox") -> np.ndarray:
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
        resized = source.resize(
            (max(1, int(source.width * scale)), max(1, int(source.height * scale))), Image.Resampling.BILINEAR
        )
        result = Image.new("RGB", (width, height))
        result.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    result = np.asarray(result).copy()
    return (result.astype(image.dtype) / 255) if floating else result.astype(image.dtype)


def encode_jpeg(image: RGBImage) -> bytes:
    p.validate_image(image, encoding="raw")
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def decode_images(images: Mapping[str, WireImage], metadata: Metadata) -> dict[str, RGBImage]:
    decoded = {}
    for camera, image in images.items():
        if metadata["image_encoding"] == "jpeg":
            image = _decode_jpeg(camera, image)
        decoded[camera] = image
    try:
        return cast(dict[str, RGBImage], p.validate_images(decoded, metadata=metadata, decoded=True))
    except ValueError as exc:
        raise ImageDecodeError(str(exc)) from exc


def decode_image_history(history: dict[str, WireImageHistory], metadata: Metadata) -> dict[str, RGBImage]:
    """Decode validated wire histories to RGB uint8 (L,H,W,3), checking frame sizes."""
    if metadata["image_encoding"] == "raw":
        return cast(dict[str, RGBImage], history)
    decoded = {}
    for camera, frames in history.items():
        images = [_decode_jpeg(f"image_history.{camera}[{index}]", frame) for index, frame in enumerate(frames)]
        size = metadata["image_sizes"].get(camera, images[0].shape[:2])
        if any(image.shape[:2] != tuple(size) for image in images):
            raise ImageDecodeError(f"image_history.{camera} size disagrees with metadata or other frames")
        decoded[camera] = np.stack(images)
    return decoded


def _decode_jpeg(name: str, data: bytes) -> RGBImage:
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format != "JPEG":
                raise ImageDecodeError(f"{name}: expected JPEG payload")
            return np.asarray(source.convert("RGB")).copy()
    except OSError as exc:
        raise ImageDecodeError(f"{name}: failed to decode JPEG") from exc
