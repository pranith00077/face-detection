# services/image_service.py
from __future__ import annotations

import os
from typing import List

import cv2

from detector.face_detector import FaceBox


def crop_faces(bgr_image, faces: List[FaceBox]):
    """
    Returns list of cropped face images in the same order as faces.
    """
    crops = []
    h, w = bgr_image.shape[:2]

    for f in faces:
        x1 = max(0, f.x)
        y1 = max(0, f.y)
        x2 = min(w, f.x + f.w)
        y2 = min(h, f.y + f.h)

        crop = bgr_image[y1:y2, x1:x2]
        crops.append(crop)

    return crops


def save_face_crops(face_crops, out_dir: str) -> List[str]:
    """
    Saves crops as face_1.jpg, face_2.jpg ...
    Returns list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    saved_paths: List[str] = []
    for idx, crop in enumerate(face_crops, start=1):
        if crop is None or crop.size == 0:
            continue
        path = os.path.join(out_dir, f"face_{idx}.jpg")
        cv2.imwrite(path, crop)
        saved_paths.append(path)

    return saved_paths
