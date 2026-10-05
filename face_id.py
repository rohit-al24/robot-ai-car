#!/usr/bin/env python3
"""
Teach the robot to recognize faces.
  python3 face_id.py capture rohit    # saves 60 photos of your face (run again in other light)
  python3 face_id.py train            # trains the model (takes seconds)
  python3 face_id.py test             # live check: prints name and score (lower score = better match)
"""
import json
import os
import sys
import time
import cv2
import numpy as np
from vision import Camera

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "dataset")
MODEL = os.path.join(BASE, "face_model.yml")
LABELS = os.path.join(BASE, "labels.json")
THRESHOLD = 75        # score below this = recognized. Tune it using 'test'.
FACE_SIZE = (100, 100)


def need_face_module():
    if not hasattr(cv2, "face"):
        sys.exit("cv2.face is missing. Try: pip install opencv-contrib-python "
                 "(inside a venv made with --system-site-packages)")


class FaceRecognizer:
    def __init__(self):
        # Load frontal and profile cascades
        f_path = os.path.join(BASE, "haarcascade_frontalface_default.xml")
        if not os.path.exists(f_path):
            f_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        p_path = os.path.join(BASE, "haarcascade_profileface.xml")
        if not os.path.exists(p_path):
            p_path = os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")

        self.frontal = cv2.CascadeClassifier(f_path)
        self.profile = cv2.CascadeClassifier(p_path)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Pre-compute elliptical mask for face feature isolation
        self.mask = np.zeros(FACE_SIZE, dtype=np.uint8)
        cv2.ellipse(self.mask, (FACE_SIZE[0] // 2, FACE_SIZE[1] // 2),
                    (int(FACE_SIZE[0] * 0.44), int(FACE_SIZE[1] * 0.48)), 0, 0, 360, 255, -1)

        self.model = None
        self.names = {}
        if os.path.exists(MODEL) and os.path.exists(LABELS):
            need_face_module()
            self.model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
            self.model.read(MODEL)
            with open(LABELS) as f:
                self.names = {int(k): v for k, v in json.load(f).items()}

    def detect(self, gray):
        """Detect faces at all angles: frontal, left profile, and right profile."""
        boxes = []

        # 1. Frontal detection
        front = self.frontal.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(40, 40))
        for b in front:
            boxes.append(tuple(b))

        # 2. Left profile detection
        if not self.profile.empty():
            left_p = self.profile.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(40, 40))
            for b in left_p:
                boxes.append(tuple(b))

            # 3. Right profile detection (flip image horizontally)
            flipped = cv2.flip(gray, 1)
            right_p = self.profile.detectMultiScale(flipped, scaleFactor=1.12, minNeighbors=4, minSize=(40, 40))
            w_img = gray.shape[1]
            for (rx, ry, rw, rh) in right_p:
                orig_x = w_img - rx - rw
                boxes.append((orig_x, ry, rw, rh))

        if not boxes:
            return []

        # Filter overlapping boxes
        def overlap(b1, b2):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = ix * iy
            union = w1 * h1 + w2 * h2 - inter
            return inter / (union + 1e-6)

        filtered = []
        # Sort by size (largest first)
        boxes = sorted(boxes, key=lambda f: -f[2] * f[3])
        for b in boxes:
            if not any(overlap(b, fb) > 0.35 for fb in filtered):
                filtered.append(b)

        return filtered

    def prepare(self, gray, box):
        """Preprocesses face with CLAHE lighting normalization + elliptical masking."""
        x, y, w, h = box
        # Pad slightly to avoid cutting off chin/forehead
        pad = int(w * 0.08)
        ih, iw = gray.shape
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(iw, x + w + pad), min(ih, y + h + pad)

        crop = gray[y0:y1, x0:x1]
        resized = cv2.resize(crop, FACE_SIZE)
        enhanced = self.clahe.apply(resized)

        # Apply elliptical mask: keep face details, fill background with neutral tone
        masked = np.full(FACE_SIZE, 128, dtype=np.uint8)
        masked[self.mask > 0] = enhanced[self.mask > 0]
        return masked

    def identify(self, gray, box):
        if self.model is None:
            return "unknown", 999.0
        prep = self.prepare(gray, box)
        label, score = self.model.predict(prep)
        name = self.names.get(label, "unknown") if score < THRESHOLD else "unknown"
        return name, score


def capture(name, count=60):
    rec = FaceRecognizer()
    cam = Camera()
    folder = os.path.join(DATASET, name)
    os.makedirs(folder, exist_ok=True)
    start = len(os.listdir(folder))
    saved = 0
    print(f"Starting multi-angle face capture for '{name}'...")
    print("-> Turn your head slowly: Left, Right, Up, Down, Smile, Neutral.")
    print("Press 'q' or ESC in the window to finish early.")

    last_save = time.time()
    while saved < count:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.05)
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = rec.detect(gray)

        now = time.time()
        for i, (x, y, w, h) in enumerate(faces):
            color = (0, 255, 0) if i == 0 else (255, 200, 0)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display, f"Face #{i+1}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if len(faces) and (now - last_save >= 0.20):
            face_img = rec.prepare(gray, faces[0])
            fn = os.path.join(folder, f"{start + saved:04d}.png")
            cv2.imwrite(fn, face_img)
            saved += 1
            last_save = now
            print(f"Captured {saved}/{count} (multi-angle)", end="\r", flush=True)

        cv2.putText(display, f"Capturing '{name}': {saved}/{count} (Turn head left/right)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("Face Registration (All Angles)", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyAllWindows()
    cam.release()
    print(f"\nDone! {saved} multi-angle photos saved to {folder}")
    if saved > 0:
        print("Now train model: & .venv\\Scripts\\python.exe face_id.py train")


def train():
    need_face_module()
    if not os.path.exists(DATASET) or not os.listdir(DATASET):
        sys.exit(f"No dataset found at {DATASET}.\nRun: & .venv\\Scripts\\python.exe face_id.py capture yourname")

    images, labels, names = [], [], {}
    for idx, person in enumerate(sorted(os.listdir(DATASET))):
        folder = os.path.join(DATASET, person)
        if not os.path.isdir(folder):
            continue
        names[idx] = person
        for fn in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Original
            images.append(img)
            labels.append(idx)

            # Augmentation 1: Horizontal Flip (covers opposite angle)
            flipped = cv2.flip(img, 1)
            images.append(flipped)
            labels.append(idx)

            # Augmentation 2: Slight rotation tilt (+6 deg, -6 deg)
            h, w = img.shape
            for angle in (-6, 6):
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), borderValue=128)
                images.append(rotated)
                labels.append(idx)

    if not images:
        sys.exit("No photos found. Run: & .venv\\Scripts\\python.exe face_id.py capture yourname")

    print(f"Training LBPH model on {len(images)} augmented samples for: {', '.join(names.values())}...")
    model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    model.train(images, np.array(labels))
    model.write(MODEL)
    with open(LABELS, "w") as f:
        json.dump(names, f, indent=2)
    print("Training complete! Model saved to face_model.yml")
    print("Now run: & .venv\\Scripts\\python.exe face_id.py test")


def draw_cute_stars(img, cx, cy, size=14, color=(0, 240, 255)):
    """Draws cute sparkling 4-pointed stars on the face frame."""
    import math
    for angle in (0, math.pi / 4):
        p1 = (int(cx - size * math.cos(angle)), int(cy - size * math.sin(angle)))
        p2 = (int(cx + size * math.cos(angle)), int(cy + size * math.sin(angle)))
        cv2.line(img, p1, p2, color, 2)


def test():
    rec = FaceRecognizer()
    cam = Camera()
    print("Live Face ID test started. Press 'q' or ESC in the window to exit.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = rec.detect(gray)

            for (x, y, w, h) in faces:
                name, score = rec.identify(gray, (x, y, w, h))
                if name != "unknown":
                    color = (0, 255, 0)
                    label = f"* {name.upper()} * ({score:.0f})"
                    # Draw cute corner stars
                    draw_cute_stars(display, x - 8, y - 8, size=12, color=(0, 240, 255))
                    draw_cute_stars(display, x + w + 8, y - 8, size=12, color=(0, 240, 255))
                    draw_cute_stars(display, x + w // 2, y - 18, size=10, color=(255, 230, 100))
                else:
                    color = (0, 120, 255)
                    label = f"Stranger ({score:.0f})"

                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display, label, (x, max(25, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                print(f"{name:12s} score={score:5.1f}", end="\r", flush=True)

            cv2.imshow("Face ID Test (Multi-Angle + Cute Stars)", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        cv2.destroyAllWindows()
        cam.release()
        print("\nStopped.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "capture" and len(sys.argv) > 2:
        capture(sys.argv[2])
    elif cmd == "train":
        train()
    elif cmd == "test":
        test()
    else:
        print(__doc__)
