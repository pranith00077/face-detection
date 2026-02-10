from dataclasses import dataclass
from typing import List
import cv2
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceDetector as MPFaceDetector,
    FaceDetectorOptions,
    RunningMode,
)


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float


class FaceDetector:
    def __init__(self, min_confidence=0.3):
        options = FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
            ),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )

        self.detector = MPFaceDetector.create_from_options(options)

    def detect(self, image) -> List[FaceBox]:
        if image is None:
            return []

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        from mediapipe.framework.formats import image_format_pb2
        from mediapipe.tasks.python.vision.core import VisionRunningMode
        from mediapipe.python import Image as MPImage

        mp_img = MPImage(image_format=image_format_pb2.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        faces = []
        if result.detections:
            for det in result.detections:
                box = det.bounding_box
                faces.append(
                    FaceBox(
                        x=box.origin_x,
                        y=box.origin_y,
                        w=box.width,
                        h=box.height,
                        confidence=det.categories[0].score,
                    )
                )
        return faces

    def close(self):
        if self.detector:
            self.detector.close()
