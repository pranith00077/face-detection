# utils/draw.py
from __future__ import annotations
from typing import List

import cv2
from detector.face_detector import FaceBox


def draw_faces(
    bgr_image,
    faces: List[FaceBox],
    draw_confidence: bool = True,
    thickness: int = 2,
):
    """
    Draw bounding boxes and confidence values on a BGR image.
    Returns a new annotated image.
    """
    output = bgr_image.copy()

    for face in faces:
        x, y, w, h = face.x, face.y, face.w, face.h
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            thickness,
        )

        if draw_confidence:
            label = f"{face.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )
            ly = max(y - th - baseline - 6, 0)
            cv2.rectangle(
                output,
                (x, ly),
                (x + tw + 6, ly + th + baseline + 6),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                output,
                label,
                (x + 3, ly + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return output
