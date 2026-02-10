# tests/test_detector.py
import os
import cv2

from detector.face_detector import FaceDetector
from utils.draw import draw_faces
from services.image_service import crop_faces, save_face_crops


if __name__ == "__main__":
    # Project root directory (…/face-detector-ml)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Input image
    img_path = os.path.join(BASE_DIR, "samples", "test1.jpg")

    # Output folders
    annotated_dir = os.path.join(BASE_DIR, "outputs", "annotated")
    faces_dir = os.path.join(BASE_DIR, "outputs", "faces")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {img_path}")

    # Detect faces (ML)
    detector = FaceDetector(min_confidence=0.5, model_selection=0)
    faces = detector.detect(img)
    detector.close()

    # Print results
    print(f"Faces detected: {len(faces)}")
    for i, f in enumerate(faces, 1):
        print(f"{i}. x={f.x}, y={f.y}, w={f.w}, h={f.h}, conf={f.confidence:.2f}")

    # Draw + save annotated image
    annotated = draw_faces(img, faces, draw_confidence=True)
    annotated_path = os.path.join(annotated_dir, "result.jpg")
    cv2.imwrite(annotated_path, annotated)
    print(f"Saved annotated output: {annotated_path}")

    # Crop + save faces
    crops = crop_faces(img, faces)
    saved_paths = save_face_crops(crops, faces_dir)

    print(f"Saved {len(saved_paths)} cropped faces to: {faces_dir}")
    for p in saved_paths:
        print(f" - {p}")
