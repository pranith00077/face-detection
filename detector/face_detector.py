# detector/face_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import mediapipe as mp


@dataclass(frozen=True)
class FaceBox:
    """Pixel-space bounding box."""
    x: int
    y: int
    w: int
    h: int
    confidence: float


class FaceDetector:
    """
    ML Face Detector using MediaPipe Face Detection.

    Notes:
    - MediaPipe returns relative bounding boxes in [0,1] range.
    - We convert them to pixel-space (x, y, w, h).
    """

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 0):
        """
        Args:
            min_confidence: Minimum confidence threshold [0.0 - 1.0]
            model_selection:
                0 -> short-range model (best for selfie/webcam)
                1 -> full-range model (better for far faces)
        """
        self.min_confidence = float(min_confidence)
        self.model_selection = int(model_selection)

        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_confidence,
        )

    def detect(self, bgr_image) -> List[FaceBox]:
        """
        Detect faces in a BGR image (OpenCV format).

        Returns:
            List of FaceBox objects (pixel coordinates + confidence)
        """
        if bgr_image is None:
            raise ValueError("Input image is None")

        h, w = bgr_image.shape[:2]
        if h == 0 or w == 0:
            return []

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        faces: List[FaceBox] = []
        if not results.detections:
            return faces

        for det in results.detections:
            score = float(det.score[0]) if det.score else 0.0
            if score < self.min_confidence:
                continue

            rel_box = det.location_data.relative_bounding_box

            # Convert relative -> absolute pixels
            x = int(rel_box.xmin * w)
            y = int(rel_box.ymin * h)
            bw = int(rel_box.width * w)
            bh = int(rel_box.height * h)

            # Clamp to image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))

            faces.append(FaceBox(x=x, y=y, w=bw, h=bh, confidence=score))

        # sort: highest confidence first
        faces.sort(key=lambda f: f.confidence, reverse=True)
        return faces

    def close(self) -> None:
        """Free MediaPipe resources."""
        if self._detector:
            self._detector.close()
            self._detector = None
