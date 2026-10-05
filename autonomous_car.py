#!/usr/bin/env python3
"""
Autonomous Driving Car Main Controller.
Integrates:
- Raspberry Pi Camera (Vision)
- Vision-based Drivable Floor & Corridor Path Detection (PathDetector)
- Obstacle Detection & Avoidance
- Motor Driver Differential Steering (MotorDriver)
"""
import sys
import time
import cv2
import numpy as np

from vision import Camera
from path_detection import PathDetector
from motor_driver import MotorDriver


def run_autonomous_car(show_display=False, base_speed=45):
    cam = Camera(source=0, size=(320, 240))
    path_finder = PathDetector(history_weight=0.70)
    motors = MotorDriver()

    print("\n" + "=" * 60)
    print("🚀 AUTONOMOUS DRIVING CAR INITIALIZED")
    print(f"Base Cruising Speed: {base_speed}%")
    print("Controls: Press 'q' or ESC in video window to stop.")
    print("=" * 60 + "\n")

    # Safety start delay (allows user to place car down)
    for i in range(3, 0, -1):
        print(f"Starting in {i} seconds...")
        time.sleep(1.0)
    print("🚗 DRIVING ENGAGED!\n")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.02)
                continue

            # Analyze frame for drivable path and obstacles
            res = path_finder.detect_path(frame)

            steering = res["steering"]       # -1.0 (left) to +1.0 (right)
            action = res["action"]           # 'FORWARD', 'STEER LEFT', 'STEER RIGHT', 'OBSTACLE AHEAD'
            clearance = res["clearance"]     # 0.0 to 1.0 fraction clear
            overlay = res["overlay"]

            # Autonomous Navigation Decision Logic
            if clearance < 0.22 or "OBSTACLE" in action:
                # Obstacle directly ahead: Stop, reverse briefly, and spin search for clear path
                print("⚠️ [OBSTACLE DETECTED] -> Braking & Re-routing!")
                motors.stop()
                time.sleep(0.2)
                motors.backward(speed=40)
                time.sleep(0.5)
                # Spin toward side with better clearance
                if steering < 0:
                    motors.spin_right(speed=45)
                else:
                    motors.spin_left(speed=45)
                time.sleep(0.4)
            else:
                # Modulate speed based on curvature (slow down on sharp corners)
                curv_factor = 1.0 - (abs(steering) * 0.40)
                active_speed = int(base_speed * curv_factor)

                # Drive with differential steering
                motors.steer(throttle=active_speed, steering=steering)

            # Display real-time telemetry
            if show_display:
                cv2.imshow("Autonomous Car Vision HUD", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    print("User stop requested.")
                    break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nEmergency Stop Triggered.")
    finally:
        motors.cleanup()
        cam.release()
        if show_display:
            cv2.destroyAllWindows()
        print("Autonomous car shutdown cleanly.")


if __name__ == "__main__":
    show_ui = "--gui" in sys.argv or "--cam" in sys.argv
    speed = 45
    for arg in sys.argv:
        if arg.startswith("--speed="):
            try:
                speed = int(arg.split("=")[-1])
            except ValueError:
                pass

    run_autonomous_car(show_display=show_ui, base_speed=speed)
