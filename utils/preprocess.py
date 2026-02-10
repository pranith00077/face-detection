# utils/preprocess.py
from __future__ import annotations
import cv2


def aggressive_resize(image, target_min_side=900, max_side=1800):
    """
    Aggressively upscale very small images so tiny faces become detectable.
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)
    max_dim = max(h, w)

    scale = 1.0

    # 🔥 Strong upscaling for tiny images
    if min_dim < target_min_side:
        scale = target_min_side / float(min_dim)

    # Prevent insane memory usage
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


def generate_scales(image):
    """
    Multi-scale pyramid for extreme small-face detection.
    """
    h, w = image.shape[:2]

    scales = [
        (image, 1.0),                         # original
        (cv2.resize(image, None, fx=1.3, fy=1.3), 1.3),
        (cv2.resize(image, None, fx=1.6, fy=1.6), 1.6),
    ]

    return scales
