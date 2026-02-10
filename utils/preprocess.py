# utils/preprocess.py
from __future__ import annotations
import cv2


def smart_resize(image, min_side: int = 700, max_side: int = 1600):
    """
    Resize image to improve small-face detection:
    - Upscale small images
    - Downscale huge images
    Keeps aspect ratio.

    Returns: (resized_image, scale_factor)
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)
    max_dim = max(h, w)

    scale = 1.0

    if min_dim < min_side:
        scale = min_side / float(min_dim)

    if max_dim * scale > max_side:
        scale = max_side / float(max_dim)

    if abs(scale - 1.0) > 1e-6:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )

    return image, scale
