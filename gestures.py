#!/usr/bin/env python3
"""
Real-time Neural Hand Gesture Recognizer for Pet Robot Interaction.
Supports:
- "COME_HERE" / "PERCH" (Open palm stretched out like calling a parrot to perch or puppy)
- "STOP" / "STAY" (Raised stop palm or closed fist)
- "PRAISE" / "GOOD_BOY" (Thumbs Up)
- "PLAY" / "DANCE" (Victory / Peace sign)
- "I_LOVE_YOU" (🤟 Sign)
- "STEER_DIRECTION" (Pointing finger direction)
"""
import os
import time
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
GESTURE_MODEL = os.path.join(BASE, "gesture_recognizer.task")

# Mapping of AI Gestures to Pet Robot Actions & Eye Moods
GESTURE_ROBOT_MAP = {
    "Open_Palm": {
        "action": "COME_HERE / PERCH",
        "description": "Owner calling pet / stretching hand for parrot perch (spread fingers)",
        "mood": "star",
        "sound": "happy_chirp",
        "drive": "FORWARD"
    },
    "HIGH_FIVE": {
        "action": "HIGH_FIVE / HI_FI",
        "description": "Owner offering a stiff High-Five palm slap!",
        "mood": "high_five",
        "sound": "high_five_slap",
        "drive": "FORWARD_SLOW"
    },
    "Closed_Fist": {
        "action": "STAY / FIST_BUMP",
        "description": "Stop / waiting for fist bump",
        "mood": "neutral",
        "sound": "ready",
        "drive": "STOP"
    },
    "Thumb_Up": {
        "action": "PRAISE / GOOD_PET",
        "description": "Owner praising the pet",
        "mood": "praise_thumbs",
        "sound": "proud",
        "drive": "SPIN_HAPPY"
    },
    "Thumb_Down": {
        "action": "SCOLD / CORRECTION",
        "description": "Owner disapproving",
        "mood": "sad",
        "sound": "whimper",
        "drive": "BACKWARD"
    },
    "Victory": {
        "action": "PLAY_TIME / DANCE",
        "description": "Owner signaling playful trick",
        "mood": "star",
        "sound": "playful",
        "drive": "WIGGLE"
    },
    "ILoveYou": {
        "action": "AFFECTION / LOVE",
        "description": "Owner showing love to pet",
        "mood": "love",
        "sound": "purr",
        "drive": "APPROACH_SLOW"
    },
    "HEART_HANDS": {
        "action": "LOVE_HEART / TWO_HAND_HEART",
        "description": "Owner making a two-handed love heart gesture (🫶)!",
        "mood": "heart_hands",
        "sound": "love_purr",
        "drive": "APPROACH_SLOW"
    },
    "FINGER_GUN": {
        "action": "PISTOL_AIM / SURRENDER",
        "description": "Owner aiming a pistol (👉 / 🔫) at robot! Hands up in fear!",
        "mood": "surprised",
        "sound": "fear_whimper",
        "drive": "STOP"
    },
    "GUN_FIRED": {
        "action": "PISTOL_FIRED / PLAY_DEAD",
        "description": "Owner shot/fired pistol upwards! Robot plays dead (X_X)!",
        "mood": "dead",
        "sound": "shot_faint",
        "drive": "STOP"
    },
    "Pointing_Up": {
        "action": "LOOK_UP / ATTENTION",
        "description": "Owner pointing attention up",
        "mood": "curious",
        "sound": "inquire",
        "drive": "LOOK_UP"
    }
}

# Hand skeleton connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


class GestureDetector:
    def __init__(self, min_hand_confidence=0.30, num_hands=2, stability_duration=2.0):
        self.recognizer = None
        self.stability_duration = stability_duration
        self._candidate_gesture = None
        self._candidate_start_time = 0.0
        self._stable_gesture = None
        self._stable_results = []
        self._last_raw_results = []

        if os.path.exists(GESTURE_MODEL):
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import vision, BaseOptions
                options = vision.GestureRecognizerOptions(
                    base_options=BaseOptions(model_asset_path=GESTURE_MODEL),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=num_hands,
                    min_hand_detection_confidence=min_hand_confidence,
                    min_hand_presence_confidence=min_hand_confidence
                )
                self.recognizer = vision.GestureRecognizer.create_from_options(options)
                print(f"[GestureDetector] MediaPipe Recognizer Active (Stability Filter: {self.stability_duration}s hold)")
            except Exception as e:
                print(f"[GestureDetector] Initialization warning: {e}")

    def _classify_landmarks_geom(self, lms_raw):
        """
        Robust geometric finger analyzer that determines hand gesture directly from 21 landmarks.
        Differentiates HIGH_FIVE (stiff joined fingers) vs Open_Palm (spread fingers for perch/calling).
        """
        if len(lms_raw) != 21:
            return "Open_Palm", 0.70

        # Extract landmark coordinates
        w_pt = np.array([lms_raw[0].x, lms_raw[0].y])  # Wrist
        t_tip = np.array([lms_raw[4].x, lms_raw[4].y]) # Thumb tip
        t_mcp = np.array([lms_raw[2].x, lms_raw[2].y]) # Thumb mcp
        i_tip = np.array([lms_raw[8].x, lms_raw[8].y]) # Index tip
        i_pip = np.array([lms_raw[6].x, lms_raw[6].y]) # Index pip
        m_tip = np.array([lms_raw[12].x, lms_raw[12].y]) # Middle tip
        m_pip = np.array([lms_raw[10].x, lms_raw[10].y]) # Middle pip
        r_tip = np.array([lms_raw[16].x, lms_raw[16].y]) # Ring tip
        r_pip = np.array([lms_raw[14].x, lms_raw[14].y]) # Ring pip
        p_tip = np.array([lms_raw[20].x, lms_raw[20].y]) # Pinky tip
        p_pip = np.array([lms_raw[18].x, lms_raw[18].y]) # Pinky pip

        # Determine extension based on distance from wrist
        def is_extended(tip, pip, wrist):
            return np.linalg.norm(tip - wrist) > (np.linalg.norm(pip - wrist) * 1.15)

        idx_open = is_extended(i_tip, i_pip, w_pt)
        mid_open = is_extended(m_tip, m_pip, w_pt)
        rng_open = is_extended(r_tip, r_pip, w_pt)
        pnk_open = is_extended(p_tip, p_pip, w_pt)
        thm_open = np.linalg.norm(t_tip - np.array([lms_raw[17].x, lms_raw[17].y])) > \
                   np.linalg.norm(t_mcp - np.array([lms_raw[17].x, lms_raw[17].y])) * 1.10

        open_count = sum([idx_open, mid_open, rng_open, pnk_open])

        # Calculate Finger Spread (Distance between adjacent fingertips normalized by hand length)
        hand_len = max(0.01, np.linalg.norm(m_tip - w_pt))
        spread_idx_mid = np.linalg.norm(i_tip - m_tip)
        spread_mid_rng = np.linalg.norm(m_tip - r_tip)
        spread_rng_pnk = np.linalg.norm(r_tip - p_tip)
        total_spread = (spread_idx_mid + spread_mid_rng + spread_rng_pnk) / hand_len

        # 1. 4 or 3 fingers extended: check if fingers are joined (High Five) or spread (Perch / Call)
        if open_count >= 3:
            # If fingers are joined and stiff (small spread) -> High Five!
            if total_spread < 0.42 and idx_open and mid_open and rng_open:
                return "HIGH_FIVE", 0.94
            else:
                return "Open_Palm", 0.90
        # 2. Finger Gun / Pistol (👉 / 🔫: Index extended, Thumb cocked, Middle+Ring+Pinky curled)
        if idx_open and not rng_open and not pnk_open:
            # If pointing upwards or tilted back sharply (Gun recoil shot) -> GUN_FIRED!
            is_upward_shot = (i_tip[1] < w_pt[1] - 0.15) and (t_tip[1] < i_tip[1] + 0.08)
            idx_horizontal = abs(i_tip[0] - w_pt[0]) > abs(i_tip[1] - w_pt[1]) * 0.65

            if is_upward_shot and not idx_horizontal:
                return "GUN_FIRED", 0.96
            elif (not mid_open or abs(i_tip[1] - m_tip[1]) > 0.05) and (thm_open or idx_horizontal):
                return "FINGER_GUN", 0.95

        # 3. Closed Fist (Stay / wait)
        if open_count == 0 and not thm_open:
            return "Closed_Fist", 0.88
        # 4. Thumbs Up (Praise)
        elif open_count == 0 and thm_open and (t_tip[1] < w_pt[1]):
            return "Thumb_Up", 0.88
        # 5. Victory / Peace (Play / dance)
        elif idx_open and mid_open and not rng_open and not pnk_open:
            return "Victory", 0.90
        # 6. I Love You (Affection)
        elif thm_open and idx_open and not mid_open and not rng_open and pnk_open:
            return "ILoveYou", 0.90
        # 7. Pointing Up
        elif idx_open and not mid_open and not rng_open and not pnk_open:
            return "Pointing_Up", 0.88

        return "Open_Palm", 0.75

    def detect_gestures(self, bgr_frame):
        """
        Detects hand gestures & landmarks in frame.
        """
        if self.recognizer is None:
            return []

        h, w, _ = bgr_frame.shape
        import mediapipe as mp
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = []
        try:
            recognition_res = self.recognizer.recognize(mp_img)
            if not recognition_res.hand_landmarks:
                return []

            # Check if 2 hands together form a Two-Hand Heart (🫶)
            is_two_hand_heart = False
            if len(recognition_res.hand_landmarks) >= 2:
                h1 = recognition_res.hand_landmarks[0]
                h2 = recognition_res.hand_landmarks[1]
                t1 = np.array([h1[4].x, h1[4].y])
                t2 = np.array([h2[4].x, h2[4].y])
                i1 = np.array([h1[8].x, h1[8].y])
                i2 = np.array([h2[8].x, h2[8].y])
                d_thumbs = np.linalg.norm(t1 - t2)
                d_index = np.linalg.norm(i1 - i2)
                if d_thumbs < 0.22 and d_index < 0.25 and (i1[1] < t1[1] + 0.10 and i2[1] < t2[1] + 0.10):
                    is_two_hand_heart = True

            for i, lms in enumerate(recognition_res.hand_landmarks):
                # Check neural gesture
                gesture_name = "None"
                score = 0.0
                if is_two_hand_heart:
                    gesture_name = "HEART_HANDS"
                    score = 0.96
                else:
                    if i < len(recognition_res.gestures) and recognition_res.gestures[i]:
                        top_gesture = recognition_res.gestures[i][0]
                        gesture_name = top_gesture.category_name
                        score = top_gesture.score

                    # If neural output is Open_Palm or None or low confidence, run geometric check
                    # to accurately differentiate HIGH_FIVE (fingers joined & stiff) vs Open_Palm (spread fingers)
                    if gesture_name in ("None", "Open_Palm") or score < 0.50:
                        geom_name, geom_score = self._classify_landmarks_geom(lms)
                        gesture_name = geom_name
                        score = max(score, geom_score)

                # Handedness (Left / Right)
                handedness = "Hand"
                if i < len(recognition_res.handedness) and recognition_res.handedness[i]:
                    handedness = recognition_res.handedness[i][0].category_name

                # Extract 21 Landmarks in pixel space
                landmarks_px = []
                xs, ys = [], []
                for lm in lms:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    landmarks_px.append((px, py, lm.z))
                    xs.append(px)
                    ys.append(py)

                # Bounding box & Center
                if xs and ys:
                    bx = max(0, min(xs) - 15)
                    by = max(0, min(ys) - 15)
                    bw = min(w - bx, max(xs) - min(xs) + 30)
                    bh = min(h - by, max(ys) - min(ys) + 30)
                    center = (bx + bw // 2, by + bh // 2)
                else:
                    bx, by, bw, bh = 0, 0, 0, 0
                    center = (w // 2, h // 2)

                # Robot action mapping
                action_info = GESTURE_ROBOT_MAP.get(gesture_name, {
                    "action": gesture_name.upper(),
                    "description": "Recognized hand motion",
                    "mood": "star",
                    "sound": "happy_chirp",
                    "drive": "FORWARD"
                })

                results.append({
                    "gesture": gesture_name,
                    "score": round(score, 2),
                    "handedness": handedness,
                    "action_info": action_info,
                    "center": center,
                    "box": (bx, by, bw, bh),
                    "landmarks": landmarks_px
                })
        except Exception as e:
            pass

        self._last_raw_results = results
        now = time.time()

        # Check dominant gesture in raw results
        raw_gesture_name = results[0]["gesture"] if results else None

        if raw_gesture_name and raw_gesture_name != "None":
            if self._candidate_gesture == raw_gesture_name:
                elapsed = now - self._candidate_start_time
                if elapsed >= self.stability_duration:
                    # Successfully held stable for 2.0 seconds!
                    self._stable_gesture = raw_gesture_name
                    self._stable_results = results
                    return results
                else:
                    # Still stabilizing / holding (less than 2s) -> do not trigger action yet!
                    self._stable_gesture = None
                    self._stable_results = []
                    return []
            else:
                # New candidate gesture started
                self._candidate_gesture = raw_gesture_name
                self._candidate_start_time = now
                self._stable_gesture = None
                self._stable_results = []
                return []
        else:
            # No hands or no recognized gesture
            self._candidate_gesture = None
            self._candidate_start_time = 0.0
            self._stable_gesture = None
            self._stable_results = []
            return []

    def get_stability_progress(self):
        """Returns (candidate_gesture_name, progress_0_to_1, is_confirmed)."""
        if not self._candidate_gesture or self._candidate_start_time <= 0:
            return None, 0.0, False
        elapsed = time.time() - self._candidate_start_time
        prog = min(1.0, max(0.0, elapsed / self.stability_duration))
        return self._candidate_gesture, prog, (prog >= 1.0)

    def draw_overlay(self, display, gestures=None):
        """Draws skeleton, gesture cards, and 2-second stability hold timer on frame."""
        # Use raw detections for visual skeleton tracking even while stabilizing
        hands_to_draw = self._last_raw_results if self._last_raw_results else (gestures or [])
        cand_name, prog, confirmed = self.get_stability_progress()

        for hand in hands_to_draw:
            lms = hand["landmarks"]
            bx, by, bw, bh = hand["box"]
            gesture = hand["gesture"]
            score = hand["score"]
            handedness = hand["handedness"]
            info = hand["action_info"]

            # Draw Hand Skeleton Bones
            if len(lms) == 21:
                for p1_idx, p2_idx in HAND_CONNECTIONS:
                    pt1 = (lms[p1_idx][0], lms[p1_idx][1])
                    pt2 = (lms[p2_idx][0], lms[p2_idx][1])
                    cv2.line(display, pt1, pt2, (0, 255, 180), 2, cv2.LINE_AA)

                # Draw Landmark Joints
                for pt in lms:
                    cv2.circle(display, (pt[0], pt[1]), 4, (0, 200, 255), -1)

            # Color palette based on action
            if gesture == "HEART_HANDS":
                color = (255, 105, 180)  # Bright Pink Heart
            elif gesture == "HIGH_FIVE":
                color = (0, 240, 255)    # Bright Gold / Sparkle (High-Five Slap)
            elif gesture == "Open_Palm":
                color = (0, 255, 120)    # Green (Calling/Perch)
            elif gesture == "Thumb_Up":
                color = (0, 230, 255)    # Gold (Praise)
            elif gesture == "ILoveYou":
                color = (255, 100, 220)  # Pink (Affection)
            elif gesture == "Closed_Fist":
                color = (0, 165, 255)    # Orange (Stay)
            elif gesture == "Victory":
                color = (255, 200, 0)    # Cyan/Yellow (Dance)
            else:
                color = (255, 255, 0)

            # Bounding box
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), color, 2)

            # Badge overlay
            action_tag = f"[{handedness.upper()}] {info['action']}"
            badge_y = max(45, by - 22)
            cv2.putText(display, action_tag, (bx, badge_y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Stability Progress Bar (Hold for 2 seconds)
            if cand_name == gesture:
                bar_w = bw
                bar_h = 6
                bar_x = bx
                bar_y = badge_y + 4
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                fill_w = int(bar_w * prog)
                fill_col = (0, 255, 0) if confirmed else (0, 200, 255)
                if fill_w > 0:
                    cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_col, -1)

                status_txt = "LOCKED (2.0s)" if confirmed else f"Hold Steady... ({prog*self.stability_duration:.1f}s/2.0s)"
                cv2.putText(display, status_txt, (bx, bar_y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, fill_col, 1)
            else:
                conf_tag = f"Confidence: {int(score * 100)}% | Reaction: {info['mood'].upper()}"
                cv2.putText(display, conf_tag, (bx, badge_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

        return display


if __name__ == "__main__":
    from vision import Camera

    cam = Camera()
    gesture_det = GestureDetector()

    print("\n[PET ROBOT NEURAL HAND GESTURE DETECTOR]")
    print("=" * 60)
    print("Available Interactive Hand Motions:")
    print(" ✋ OPEN PALM   -> Stretched hand (Calling pet / Parrot perch)")
    print(" 👍 THUMBS UP   -> Praise / Good pet (Sparkle & happy chirp)")
    print(" ✊ CLOSED FIST -> Stay / Fist bump (Halts & prepares)")
    print(" ✌️  VICTORY     -> Play / Dance trick (Playful wiggle)")
    print(" 🤟 I LOVE YOU  -> Affection (Heart eyes & purr)")
    print(" 👎 THUMB DOWN  -> Scold / Correction (Sad eye reaction)")
    print("=" * 60)
    print("Press 'q' or ESC to exit.\n")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue

            display = frame.copy()
            gestures = gesture_det.detect_gestures(frame)
            gesture_det.draw_overlay(display, gestures)

            # HUD Header
            hud = f"Gestures: {len(gestures)}"
            if gestures:
                hud += f" | Active: {gestures[0]['action_info']['action']}"
            cv2.putText(display, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow("Pet Robot Hand Gestures & Motions", display)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    finally:
        cv2.destroyAllWindows()
        cam.release()
