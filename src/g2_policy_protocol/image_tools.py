"""Optional Pillow helpers for integration tools; core protocol never decodes images."""

from io import BytesIO

import numpy as np

from . import protocol as p


def encode_jpeg(image):
    from PIL import Image

    p.validate_image(image, encoding="raw")
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def decode_images(images, metadata):
    from PIL import Image

    decoded = {}
    for camera, image in images.items():
        if metadata["image_encoding"] == "jpeg":
            with Image.open(BytesIO(image)) as source:
                if source.format != "JPEG":
                    raise ValueError(f"{camera}: expected JPEG payload")
                image = np.asarray(source.convert("RGB")).copy()
        decoded[camera] = image
    return p.validate_images(decoded, metadata=metadata, decoded=True)
