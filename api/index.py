# api/index.py
from __future__ import annotations

import os
import base64
import cv2
from flask import Flask, render_template, request

from detector.face_detector import FaceDetector, FaceBox
from utils.draw import draw_faces
from services.image_service import crop_faces
from utils.preprocess import smart_resize  # keep your small-face resizing

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

# ---- Upload constraints ----
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_MB = 4  # keep <= ~4.5MB Vercel body limit
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)


def is_allowed_filename(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower().strip())
    return ext in ALLOWED_EXTENSIONS


def to_data_url(bgr_image, quality: int = 85) -> str:
    """Convert BGR image (OpenCV) to base64 data URL."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", bgr_image, encode_params)
    if not ok:
        raise ValueError("Failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename.strip() == "":
            return render_template("index.html", error="No image uploaded")

        filename = file.filename.strip()
        if not is_allowed_filename(filename):
            return render_template(
                "index.html",
                error=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        # Read bytes (serverless-safe: no disk write)
        file_bytes = file.read()
        if not file_bytes:
            return render_template("index.html", error="Empty file")

        if len(file_bytes) > MAX_UPLOAD_BYTES:
            return render_template(
                "index.html",
                error=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB."
            )

        # Decode with OpenCV
        import numpy as np
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return render_template("index.html", error="Invalid image (cannot decode)")

        # Small-face friendly resize
        resized_img, scale = smart_resize(img, min_side=700, max_side=1400)

        detector = FaceDetector(min_confidence=0.3, model_selection=1)
        faces_resized = detector.detect(resized_img)
        detector.close()

        # Map boxes back to original image coords
        faces: list[FaceBox] = []
        if scale != 1.0:
            inv = 1.0 / scale
            for f in faces_resized:
                faces.append(
                    FaceBox(
                        x=int(f.x * inv),
                        y=int(f.y * inv),
                        w=int(f.w * inv),
                        h=int(f.h * inv),
                        confidence=f.confidence,
                    )
                )
        else:
            faces = faces_resized

        # No face case
        if len(faces) == 0:
            return render_template(
                "index.html",
                faces=0,
                no_face=True,
                message="No face detected. Try a clearer image or higher resolution.",
                uploaded_name=filename,
            )

        # Annotate (in memory)
        annotated = draw_faces(img, faces, draw_confidence=True)
        result_image_data = to_data_url(annotated, quality=85)

        # Crops (in memory)
        crops = crop_faces(img, faces)
        face_images_data = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            face_images_data.append(to_data_url(crop, quality=90))

        return render_template(
            "index.html",
            faces=len(faces),
            no_face=False,
            result_image_data=result_image_data,
            face_images_data=face_images_data,
            uploaded_name=filename,
        )

    return render_template("index.html")
