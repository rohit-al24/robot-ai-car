"""Camera helper + simple hand-wave detector (works on 32-bit Pi, no extra libraries)."""
import time
import cv2

# 0 = USB webcam
# "http://PHONE_IP:8080/video" = phone running IP Webcam
# "picam" = Raspberry Pi ribbon camera (needs python3-picamera2)
CAMERA_SOURCE = 0


class Camera:
    def __init__(self, source=CAMERA_SOURCE, size=(320, 240)):
        self.size = size
        self.picam = None
        self.cap = None

        # 1. First try Raspberry Pi Official CSI Ribbon Camera (libcamera / Picamera2)
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            cfg = self.picam.create_video_configuration(
                main={"size": size, "format": "BGR888"})
            self.picam.configure(cfg)
            self.picam.start()
            print("[Camera] Raspberry Pi CSI Ribbon Camera (Picamera2) Active!")
            return
        except Exception:
            self.picam = None

        # 2. Fallback to USB Webcam / V4L2 / DirectShow
        try:
            # Try V4L2 backend on Linux (Raspberry Pi) first for fastest FPS
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            if self.cap.isOpened():
                print("[Camera] USB / V4L2 Video Device Active!")
        except Exception as e:
            print(f"[Camera Warning] {e}")

    def read(self):
        if self.picam is not None:
            frame, ok = self.picam.capture_array(), True
        else:
            ok, frame = self.cap.read()
        if ok and (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        return ok, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.picam is not None:
            self.picam.stop()


class WaveDetector:
    """Detects side-to-side motion (a waving hand) from frame differences.
    Don't use it while the robot itself is driving: the whole picture moves."""

    def __init__(self):
        self.prev = None
        self.history = []   # (time, x position of motion)

    def update(self, gray):
        small = cv2.GaussianBlur(cv2.resize(gray, (160, 120)), (15, 15), 0)
        if self.prev is None:
            self.prev = small
            return False
        diff = cv2.absdiff(self.prev, small)
        self.prev = small
        mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        area = cv2.countNonZero(mask) / mask.size
        now = time.time()
        self.history = [h for h in self.history if now - h[0] < 1.5]
        if area < 0.02 or area > 0.5:      # too little motion, or whole view moving
            return False
        m = cv2.moments(mask, binaryImage=True)
        if m["m00"] == 0:
            return False
        self.history.append((now, m["m10"] / m["m00"]))
        xs = [h[1] for h in self.history]
        if len(xs) < 8:
            return False
        steps = [b - a for a, b in zip(xs, xs[1:]) if abs(b - a) > 2]
        changes = sum(1 for a, b in zip(steps, steps[1:]) if a * b < 0)
        if changes >= 3:
            self.history.clear()
            return True
        return False
