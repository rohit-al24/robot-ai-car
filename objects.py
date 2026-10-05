#!/usr/bin/env python3
"""
Real-time Neural Object Detector & ArUco Charging Dock Tracker.
Detects 80 everyday objects + ArUco Docking Markers.
"""
import os
import time
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OBJECT_MODEL = os.path.join(BASE, "efficientdet_lite0.tflite")

# Category color map
CATEGORY_COLORS = {
    # Cards / Documents
    "card": (255, 180, 0),
    "book": (200, 180, 50),
    # Tech / Devices
    "cell phone": (255, 0, 200),
    "laptop": (255, 100, 0),
    "mouse": (200, 150, 0),
    "keyboard": (200, 150, 0),
    "remote": (180, 0, 255),
    "tv": (255, 50, 0),
    # Animals
    "cat": (0, 255, 120),
    "dog": (0, 255, 120),
    "bird": (0, 255, 200),
    # Play / Toys / Sports
    "sports ball": (0, 230, 255),
    "frisbee": (0, 230, 255),
    "teddy bear": (255, 150, 220),
    # Food / Drinks
    "bottle": (255, 200, 50),
    "cup": (255, 200, 50),
    "apple": (0, 100, 255),
    "banana": (0, 230, 255),
    # Person
    "person": (0, 255, 0),
    # Dock
    "dock": (0, 255, 255)
}

# Per-class confidence thresholds to eliminate common false positives
CLASS_THRESHOLDS = {
    "remote": 0.58,
    "cell phone": 0.50,
    "mouse": 0.45,
    "keyboard": 0.45,
    "book": 0.42,
    "cup": 0.38,
    "bottle": 0.38,
}
DEFAULT_MIN_SCORE = 0.38


class ObjectDetector:
    def __init__(self, score_threshold=0.35, max_results=8):
        self.detector = None
        if os.path.exists(OBJECT_MODEL):
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import vision, BaseOptions
                options = vision.ObjectDetectorOptions(
                    base_options=BaseOptions(model_asset_path=OBJECT_MODEL),
                    running_mode=vision.RunningMode.IMAGE,
                    score_threshold=score_threshold,
                    max_results=max_results
                )
                self.detector = vision.ObjectDetector.create_from_options(options)
                print("[ObjectDetector] MediaPipe Neural Object Detector Active (80 COCO Classes + Card Filter)")
            except Exception as e:
                print(f"[ObjectDetector] Load warning: {e}")

        # Setup ArUco Marker Detector for Charging Dock
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except Exception:
            # Fallback for older OpenCV versions
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_detector = None

    def _refine_label(self, bgr_frame, label, score, bx, by, bw, bh):
        """
        Differentiates between actual cell phone / remote vs card / planar paper
        using aspect ratio, color profile, and structural characteristics.
        """
        min_thresh = CLASS_THRESHOLDS.get(label, DEFAULT_MIN_SCORE)
        if score < min_thresh:
            # Check if this low/medium confidence item is actually a card
            if label in ("remote", "cell phone", "book") and bw > 25 and bh > 25:
                ar = max(bw, bh) / float(max(1, min(bw, bh)))
                # Standard ID/Credit/Playing card aspect ratio is ~1.4 - 1.7
                if 1.25 <= ar <= 1.85:
                    return "card", max(score, 0.65)
            return None, 0.0

        # Feature analysis on the cropped object
        crop = bgr_frame[by:by + bh, bx:bx + bw]
        if crop.size > 0:
            ar = max(bw, bh) / float(max(1, min(bw, bh)))
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray_crop))

            # Disambiguate 'remote'
            if label == "remote":
                # Real TV remotes are long & slender (ar >= 2.1) and usually dark
                if ar < 1.95:
                    if 1.25 <= ar <= 1.85 or mean_brightness > 90:
                        return "card", 0.72  # Correct misclassified card
                    elif score < 0.65:
                        return None, 0.0     # Discard false remote

            # Disambiguate 'cell phone' vs 'card'
            elif label == "cell phone":
                # If it has standard card aspect ratio and bright paper surface (not dark screen)
                if 1.30 <= ar <= 1.75 and mean_brightness > 115 and score < 0.75:
                    return "card", 0.70

        return label, score

    def detect_objects(self, bgr_frame):
        """
        Detects objects & ArUco dock in frame.
        Returns list of dicts:
          [{"label": "bottle", "score": 0.85, "box": (x, y, w, h), "center": (cx, cy), "type": "coco"}, ...]
        """
        results = []
        h, w, _ = bgr_frame.shape

        # 1. Detect COCO Objects via MediaPipe Neural Network
        if self.detector is not None:
            import mediapipe as mp
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detect_res = self.detector.detect(mp_img)

            for detection in detect_res.detections:
                cat = detection.categories[0]
                raw_label = cat.category_name.lower()
                raw_score = cat.score
                bb = detection.bounding_box
                bx, by, bw, bh = int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)

                # Filter out of bounds
                bx = max(0, min(w - 1, bx))
                by = max(0, min(h - 1, by))
                bw = min(w - bx, bw)
                bh = min(h - by, bh)

                if bw > 18 and bh > 18:
                    label, score = self._refine_label(bgr_frame, raw_label, raw_score, bx, by, bw, bh)
                    if label is not None:
                        results.append({
                            "label": label,
                            "score": round(score, 2),
                            "box": (bx, by, bw, bh),
                            "center": (bx + bw // 2, by + bh // 2),
                            "type": "object"
                        })

        # 2. Detect ArUco Charging Dock Markers
        try:
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            if hasattr(self, 'aruco_detector') and self.aruco_detector is not None:
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None and len(ids) > 0:
                flat_ids = np.ravel(ids)
                for i, marker_corners in enumerate(corners):
                    pts = marker_corners.reshape(-1, 2)
                    bx = int(np.min(pts[:, 0]))
                    by = int(np.min(pts[:, 1]))
                    bw = int(np.max(pts[:, 0]) - bx)
                    bh = int(np.max(pts[:, 1]) - by)
                    marker_id = int(flat_ids[i])
                    results.append({
                        "label": f"dock [ID:{marker_id}]",
                        "score": 1.0,
                        "box": (bx, by, bw, bh),
                        "center": (bx + bw // 2, by + bh // 2),
                        "type": "dock",
                        "corners": pts
                    })
        except Exception:
            pass

        return results


if __name__ == "__main__":
    from vision import Camera

    obj_det = ObjectDetector()
    cam = Camera()

    print("\n[LIVE NEURAL OBJECT & ARUCO DOCK DETECTOR]")
    print("-> Point camera at objects: Phone, Bottle, Cup, Laptop, Dog, Cat, Ball, Book, etc.")
    print("Press 'q' or ESC in the window to exit.\n")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue

            display = frame.copy()
            objects = obj_det.detect_objects(frame)

            for obj in objects:
                label = obj["label"]
                score = obj["score"]
                x, y, w, h = obj["box"]

                color = CATEGORY_COLORS.get(label.split()[0], (0, 200, 255))
                if obj["type"] == "dock":
                    color = (0, 255, 255)
                    # Draw ArUco dock corners
                    pts = obj["corners"].astype(np.int32)
                    cv2.polylines(display, [pts], True, color, 3)

                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                tag = f"{label.upper()} ({int(score * 100)}%)"
                cv2.putText(display, tag, (x, max(22, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Object count HUD
            info = f"Objects Detected: {len(objects)}"
            cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow("Neural Object & Dock Tracker", display)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    finally:
        cv2.destroyAllWindows()
        cam.release()
