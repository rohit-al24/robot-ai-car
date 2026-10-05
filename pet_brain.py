#!/usr/bin/env python3
"""
Pet brain: camera -> face/hand reactions -> eyes.
  python3 pet_brain.py          # eyes shown in a window (xrdp)
  python3 pet_brain.py --lcd    # eyes shown on the SPI LCD
Later we merge this with pet_robot.py (wheels, servos, docking).
"""
import sys
import threading
import time
import traceback
import cv2
import numpy as np
from eyes import Eyes, run_tk, run_lcd
from vision import Camera, WaveDetector
from face_id import FaceRecognizer
from emotions import EmotionDetector
from objects import ObjectDetector, CATEGORY_COLORS
from gestures import GestureDetector
from naughty_ai import NaughtyAI


def vision_loop(eyes, stop, show_cam=False, frame_holder=None):
    try:
        cam = Camera()
        rec = FaceRecognizer()
        wave = WaveDetector()
        emo_det = EmotionDetector()
        obj_det = ObjectDetector()
        gesture_det = GestureDetector()
        ai = NaughtyAI(speech_cooldown=5.0)

        last_face = time.time()
        hold_until = 0.0
        last_name = None
        last_emo = None
        last_gesture = None
        current = None          # (name, cx, cy, emotion)
        
        # 2-Second Face / Stranger / Emotion Stability Lock
        candidate_face_key = None
        candidate_face_start = 0.0
        stable_face = None      # (name, cx, cy, emotion)
        face_locked = False
        
        detected_objects = []
        detected_gestures = []
        n = 0

        while not stop.is_set():
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue
            n += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            now = time.time()
            waved = wave.update(gray)

            # Detect hand gestures
            detected_gestures = gesture_det.detect_gestures(frame)

            # Detect multi-angle faces
            faces = rec.detect(gray)
            if len(faces):
                box = faces[0]
                name, score = rec.identify(gray, box)
                emotion, conf, _ = emo_det.detect_emotion(frame)
                current = (name, box[0] + box[2] / 2, box[1] + box[3] / 2, emotion)

                # Face Stability Lock Check (Must stay present & consistent for 2.0 seconds)
                face_key = (name, emotion)
                if candidate_face_key == face_key:
                    face_elapsed = now - candidate_face_start
                    if face_elapsed >= 2.0:
                        stable_face = current
                        face_locked = True
                    else:
                        stable_face = None
                        face_locked = False
                else:
                    candidate_face_key = face_key
                    candidate_face_start = now
                    stable_face = None
                    face_locked = False
            else:
                current = None
                candidate_face_key = None
                candidate_face_start = 0.0
                stable_face = None
                face_locked = False

            # Detect objects every 2 frames for top performance
            if n % 2 == 0:
                detected_objects = obj_det.detect_objects(frame)

            # 1. Highest priority: Hand Gestures (Direct Owner Commands like Calling / Parrot Perch / High-Five)
            if detected_gestures:
                g = detected_gestures[0]
                g_info = g["action_info"]
                gcx, gcy = g["center"]
                eyes.look((gcx - w / 2) / (w / 2), (gcy - h / 2) / (h / 2))

                # If single hand High-Five, match the hand side on screen
                if g["gesture"] == "HIGH_FIVE":
                    if len(detected_gestures) == 1:
                        side = "left" if g["handedness"] == "Left" else "right"
                        eyes.set_mood("high_five", hand_side=side)
                    else:
                        eyes.set_mood("high_five", hand_side="both")
                elif g["gesture"] == "FINGER_GUN":
                    # Pistol aimed -> Fear & Surrender (hands raised up trembling)
                    eyes.set_mood("surrender")
                    hold_until = now + 1.6
                elif g["gesture"] == "GUN_FIRED":
                    # Pistol fired upwards -> Play dead (X_X eyes & fall down, recover)
                    eyes.set_mood("dead")
                    hold_until = now + 4.0
                elif g["gesture"] == "Thumb_Up":
                    eyes.trigger_praise_sequence()
                    hold_until = now + 3.0
                else:
                    eyes.set_mood(g_info["mood"])

                if g["gesture"] not in ("FINGER_GUN", "GUN_FIRED", "Thumb_Up"):
                    hold_until = now + 1.2

                if g["gesture"] != last_gesture:
                    print(f">> Hand Motion [{g_info['action']}] detected! (Reaction Eyes: {g_info['mood']})")
                    last_gesture = g["gesture"]
            else:
                last_gesture = None
                if waved:
                    print(">> Wave detected! (Surprised reaction)")
                    eyes.set_mood("surprised")
                    hold_until = now + 1.5
                elif now > hold_until:
                    if current:
                        cx, cy = current[1], current[2]
                        # Look toward the face position (-1..1) continuously
                        eyes.look((cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2))

                    if stable_face:
                        name, cx, cy, emotion = stable_face

                        # Expression & Identity Mood Map (Triggered only after 2s lock!)
                        if name != "unknown":
                            if emotion == "happy":
                                # Registered user happy reaction -> energetic glowing happy eyes with floating love hearts!
                                eyes.set_mood("registered_happy")
                            elif emotion == "cute":
                                eyes.set_mood("registered_shy")
                            elif emotion == "sad":
                                eyes.set_mood("sad")
                            elif emotion == "angry":
                                eyes.set_mood("angry")
                            elif emotion == "surprised":
                                eyes.set_mood("cute_surprised")
                            elif emotion == "wink":
                                eyes.set_mood("wink")
                            elif emotion == "sleepy":
                                eyes.set_mood("sleepy")
                            else:
                                # Seeing registered user -> Adorable Shy reaction with rosy blushing cheeks and soft eyes!
                                eyes.set_mood("registered_shy")

                            if (name, emotion) != (last_name, last_emo):
                                print(f">> [LOCKED 2.0s] {name} is {emotion.upper()}! (Eyes: {eyes.mood})")
                                last_name, last_emo = name, emotion
                        else:
                            if emotion == "happy":
                                eyes.set_mood("happy")  # Normal happy eyes for strangers
                            elif emotion in ("cute", "surprised", "wink", "sleepy", "angry", "sad"):
                                eyes.set_mood(emotion)
                            else:
                                eyes.set_mood("curious")
                            if (last_name, last_emo) != ("unknown", emotion):
                                print(f">> [LOCKED 2.0s] Stranger ({emotion.upper()}) detected! (Eyes: {eyes.mood})")
                                last_name, last_emo = "unknown", emotion
                        last_face = now
                    elif detected_objects and not current:
                        # No human face, but objects/dock detected -> Track closest object!
                        lead_obj = detected_objects[0]
                        ocx, ocy = lead_obj["center"]
                        olabel = lead_obj["label"].split()[0]
                        eyes.look((ocx - w / 2) / (w / 2), (ocy - h / 2) / (h / 2))

                        if lead_obj["type"] == "dock":
                            eyes.set_mood("neutral")
                        elif olabel in ("sports ball", "frisbee", "teddy bear"):
                            eyes.set_mood("star")
                        elif olabel in ("cat", "dog", "bird"):
                            eyes.set_mood("love")
                        else:
                            eyes.set_mood("curious")
                    elif not current:
                        eyes.look_auto()
                        last_name = None
                        last_emo = None
                        eyes.set_mood("sleepy" if (now - last_face > 45) else "neutral")

            # Trigger Naughty AI Voice Banter asynchronously ONLY on stable/locked recognition
            ai.trigger_reaction_async(
                face_info=stable_face,
                emotion=stable_face[3] if stable_face else None,
                gestures=detected_gestures,
                objects=detected_objects,
                bgr_frame=frame
            )

            if show_cam:
                display = frame.copy()

                # Draw Detected Objects & Dock
                for obj in detected_objects:
                    ox, oy, ow, oh = obj["box"]
                    olabel = obj["label"]
                    ocolor = CATEGORY_COLORS.get(olabel.split()[0], (0, 200, 255))
                    if obj["type"] == "dock":
                        ocolor = (0, 255, 255)
                        pts = obj["corners"].astype(np.int32)
                        cv2.polylines(display, [pts], True, ocolor, 3)
                    else:
                        cv2.rectangle(display, (ox, oy), (ox + ow, oy + oh), ocolor, 2)
                    cv2.putText(display, f"{olabel.upper()}", (ox, max(18, oy - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ocolor, 2)

                # Draw Face & Emotion Overlay with 2-second lock progress
                if current and len(faces):
                    bx, by, bw, bh = faces[0]
                    name, _, _, emotion = current
                    col = (0, 255, 0) if name != "unknown" else (0, 150, 255)
                    cv2.rectangle(display, (bx, by), (bx + bw, by + bh), col, 2)

                    # Face Lock Progress Bar
                    f_prog = min(1.0, max(0.0, (now - candidate_face_start) / 2.0)) if candidate_face_start > 0 else 0.0
                    bar_w = bw
                    bar_y = max(35, by - 12)
                    cv2.rectangle(display, (bx, bar_y), (bx + bar_w, bar_y + 5), (50, 50, 50), -1)
                    if f_prog > 0:
                        cv2.rectangle(display, (bx, bar_y), (bx + int(bar_w * f_prog), bar_y + 5), (0, 255, 0) if face_locked else (0, 200, 255), -1)

                    status_str = "LOCKED" if face_locked else f"Locking... ({f_prog * 2.0:.1f}s/2.0s)"
                    label = f"{name.upper()} [{emotion.upper()}] - {status_str}"
                    cv2.putText(display, label, (bx, max(22, by - 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)

                # Draw Detected Gestures & Stability Progress
                gesture_det.draw_overlay(display, detected_gestures)

                # Live AI Speech Subtitles HUD & Trigger Reason
                if ai.last_comment:
                    cv2.rectangle(display, (0, h - 50), (w, h), (15, 15, 15), -1)
                    cv2.putText(display, f"GRIFFIN: {ai.last_comment[:60]}", (8, h - 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    if ai.last_reason:
                        cv2.putText(display, f"REASON: [{ai.last_reason[:65]}]", (8, h - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 120), 1)

                # Status Header HUD
                g_str = detected_gestures[0]["action_info"]["action"] if detected_gestures else "None"
                hud = f"Objects: {len(detected_objects)} | Gesture: {g_str} | Lang: English"
                cv2.putText(display, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

                if frame_holder is not None:
                    frame_holder[0] = display

            time.sleep(0.02)
        cam.release()
    except Exception:
        traceback.print_exc()


def main():
    show_cam = "--cam" in sys.argv
    eyes = Eyes()
    stop = threading.Event()
    frame_holder = [None] if show_cam else None

    threading.Thread(target=vision_loop, args=(eyes, stop, show_cam, frame_holder), daemon=True).start()
    try:
        if "--lcd" in sys.argv:
            run_lcd(eyes)
        else:
            get_frame_func = (lambda: frame_holder[0]) if show_cam else None
            run_tk(eyes, get_cam_frame=get_frame_func)
    finally:
        stop.set()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
