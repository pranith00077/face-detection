# app.py
from __future__ import annotations

import os
import cv2
from flask import Flask, render_template, request, send_from_directory

from detector.face_detector import FaceDetector, FaceBox
from utils.draw import draw_faces
from services.image_service import crop_faces, save_face_crops
from utils.preprocess import aggressive_resize, generate_scales

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ANNOTATED_DIR = os.path.join(BASE_DIR, "outputs", "annotated")
FACES_DIR = os.path.join(BASE_DIR, "outputs", "faces")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

app = Flask(__name__)


@app.route("/outputs/<path:filename>")
def outputs(filename: str):
    return send_from_directory(os.path.join(BASE_DIR, "outputs"), filename)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename.strip() == "":
            return render_template("index.html", error="No image uploaded")

        upload_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(upload_path)

        img = cv2.imread(upload_path)
        if img is None:
            return render_template("index.html", error="Invalid image")

        # -------------------------------------------------
        # 🔥 STEP 1: aggressive upscaling for tiny images
        # -------------------------------------------------
        base_img, base_scale = aggressive_resize(img)

        # -------------------------------------------------
        # 🔥 STEP 2: multi-scale detection pyramid
        # -------------------------------------------------
        detector = FaceDetector(min_confidence=0.25, model_selection=1)

        detected_faces: list[FaceBox] = []

        for scaled_img, scale in generate_scales(base_img):
            faces_scaled = detector.detect(scaled_img)

            # Map boxes back to original image
            inv = 1.0 / (scale * base_scale)
            for f in faces_scaled:
                detected_faces.append(
                    FaceBox(
                        x=int(f.x * inv),
                        y=int(f.y * inv),
                        w=int(f.w * inv),
                        h=int(f.h * inv),
                        confidence=f.confidence,
                    )
                )

        detector.close()

        # -------------------------------------------------
        # 🔥 STEP 3: Remove overlapping duplicates
        # -------------------------------------------------
        final_faces: list[FaceBox] = []

        def iou(a: FaceBox, b: FaceBox) -> float:
            xa = max(a.x, b.x)
            ya = max(a.y, b.y)
            xb = min(a.x + a.w, b.x + b.w)
            yb = min(a.y + a.h, b.y + b.h)
            inter = max(0, xb - xa) * max(0, yb - ya)
            union = (a.w * a.h) + (b.w * b.h) - inter
            return inter / union if union else 0.0

        for f in sorted(detected_faces, key=lambda x: x.confidence, reverse=True):
            if all(iou(f, kept) < 0.4 for kept in final_faces):
                final_faces.append(f)

        # -------------------------------------------------
        # ❌ NO FACE CASE
        # -------------------------------------------------
        if not final_faces:
            return render_template(
                "index.html",
                faces=0,
                no_face=True,
                message="No face detected. Even after aggressive scaling. Try a clearer frontal image.",
            )

        # Clear previous crops
        for fn in os.listdir(FACES_DIR):
            try:
                os.remove(os.path.join(FACES_DIR, fn))
            except Exception:
                pass

        # Draw + save
        annotated = draw_faces(img, final_faces)
        cv2.imwrite(os.path.join(ANNOTATED_DIR, "result.jpg"), annotated)

        crops = crop_faces(img, final_faces)
        saved_paths = save_face_crops(crops, FACES_DIR)

        return render_template(
            "index.html",
            faces=len(final_faces),
            no_face=False,
            result_image="/outputs/annotated/result.jpg",
            face_images=["/outputs/faces/" + os.path.basename(p) for p in saved_paths],
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
