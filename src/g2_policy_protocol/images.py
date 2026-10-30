"""JPEG helpers for adapters and the check tool; the protocol core never decodes images."""

from io import BytesIO

import numpy as np
from PIL import Image

from . import protocol as p


def resize_rgb(image, size):
    """Letterbox an HWC RGB image to (height, width), preserving its dtype/range."""
    image = np.asarray(image)
    height, width = size
    if image.shape == (height, width, 3):
        return image.copy()
    floating = np.issubdtype(image.dtype, np.floating)
    pixels = np.clip(np.rint(image * 255) if floating else image, 0, 255).astype(np.uint8)
    source = Image.fromarray(pixels)
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
            with Image.open(BytesIO(image)) as source:
                if source.format != "JPEG":
                    raise ValueError(f"{camera}: expected JPEG payload")
                image = np.asarray(source.convert("RGB")).copy()
        decoded[camera] = image
    return p.validate_images(decoded, metadata=metadata, decoded=True)
