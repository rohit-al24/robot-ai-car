#!/usr/bin/env python3
"""
Real-time Drivable Floor & Corridor Path Detector for Autonomous Robot Navigation.
Features:
- Drivable floor segmentation using adaptive color/texture modeling & edge boundaries.
- Object & obstacle avoidance integration.
- Target steering angle (-1.0 to +1.0) & waypoint path generation.
- Real-time HUD with corridor overlay, heading vector, and clearance status.
"""
import time
import cv2
import numpy as np


class PathDetector:
    def __init__(self, history_weight=0.75, num_slices=6):
        """
        Args:
            history_weight: Temporal smoothing factor for steering stability (0..1)
            num_slices: Number of horizontal scan slices along the ground corridor
        """
        self.history_weight = history_weight
        self.num_slices = num_slices
        self.smoothed_steering = 0.0
        self.prev_floor_sample = None

    def _sample_floor_seed(self, frame_hsv):
        """
        Samples the bottom center region right in front of robot base
        to adaptively learn current floor color/lighting.
        """
        h, w, _ = frame_hsv.shape
        seed_y1 = int(h * 0.82)
        seed_y2 = int(h * 0.96)
        seed_x1 = int(w * 0.40)
        seed_x2 = int(w * 0.60)

        seed = frame_hsv[seed_y1:seed_y2, seed_x1:seed_x2]
        if seed.size == 0:
            return np.array([0, 0, 100]), np.array([180, 50, 255])

        mean_val = np.mean(seed, axis=(0, 1))
        std_val = np.std(seed, axis=(0, 1))

        # Build dynamic tolerance window around current floor sample
        h_tol = max(18, int(std_val[0] * 2.5))
        s_tol = max(45, int(std_val[1] * 2.8))
        v_tol = max(55, int(std_val[2] * 2.8))

        lower = np.array([
            max(0, mean_val[0] - h_tol),
            max(0, mean_val[1] - s_tol),
            max(20, mean_val[2] - v_tol)
        ], dtype=np.uint8)

        upper = np.array([
            min(180, mean_val[0] + h_tol),
            min(255, mean_val[1] + s_tol),
            min(255, mean_val[2] + v_tol)
        ], dtype=np.uint8)

        return lower, upper

    def detect_path(self, bgr_frame, detected_objects=None):
        """
        Analyzes the frame and returns drivable path metrics and visualization.
        
        Returns:
            dict containing:
              - 'steering': float (-1.0 left to +1.0 right)
              - 'action': string ("FORWARD", "STEER LEFT", "STEER RIGHT", "OBSTACLE AHEAD")
              - 'clearance': float (0.0 to 1.0 fraction of path clear ahead)
              - 'waypoints': list of (x, y) centerpoints from bottom to top
              - 'corridor_poly': polygon coordinates of drivable zone
              - 'overlay': annotated BGR frame
              - 'drivable_mask': binary floor mask
        """
        h, w, _ = bgr_frame.shape
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

        # 1. Adaptively segment drivable floor
        lower_floor, upper_floor = self._sample_floor_seed(hsv)
        floor_mask = cv2.inRange(hsv, lower_floor, upper_floor)

        # 2. Structural edge detection to avoid walls, step edges & furniture borders
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        
        # Floor cannot cross heavy edge obstacles
        floor_mask[edges_dilated > 0] = 0

        # 3. Mask out any detected objects (from neural object detector or ArUco dock)
        if detected_objects:
            for obj in detected_objects:
                bx, by, bw, bh = obj.get("box", (0, 0, 0, 0))
                # Expand box slightly at bottom for safety margin
                pad = 8
                ox1 = max(0, bx - pad)
                oy1 = max(0, by - pad)
                ox2 = min(w, bx + bw + pad)
                oy2 = min(h, by + bh + pad)
                floor_mask[oy1:oy2, ox1:ox2] = 0

        # 4. Perspective Floor ROI Trapezoid (focus on drivable front zone)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_poly = np.array([
            [int(w * 0.02), h],
            [int(w * 0.28), int(h * 0.40)],
            [int(w * 0.72), int(h * 0.40)],
            [int(w * 0.98), h]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, [roi_poly], 255)
        drivable = cv2.bitwise_and(floor_mask, floor_mask, mask=roi_mask)

        # Clean noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        drivable = cv2.morphologyEx(drivable, cv2.MORPH_CLOSE, kernel)
        drivable = cv2.morphologyEx(drivable, cv2.MORPH_OPEN, kernel)

        # 5. Multi-Horizon Scan Slices to find corridor centers & boundaries
        waypoints = []
        left_bounds = []
        right_bounds = []

        y_start = int(h * 0.92)
        y_end = int(h * 0.45)
        y_step = (y_start - y_end) // self.num_slices

        center_x = w // 2
        last_cx = center_x
        clearance_count = 0

        for i in range(self.num_slices):
            curr_y = y_start - i * y_step
            slice_h = max(6, y_step // 2)
            y1 = max(0, curr_y - slice_h)
            y2 = min(h, curr_y + slice_h)

            slice_row = drivable[y1:y2, :]
            row_profile = np.sum(slice_row, axis=0) / 255.0

            # Find connected drivable segment around expected path center
            active_cols = np.where(row_profile > (slice_h * 0.4))[0]

            if len(active_cols) > 15:
                # Group connected spans
                spans = np.split(active_cols, np.where(np.diff(active_cols) > 4)[0] + 1)
                # Find span closest to last known center
                best_span = min(spans, key=lambda s: abs(np.mean(s) - last_cx))
                span_w = best_span[-1] - best_span[0]

                if span_w >= 20:  # Minimum drivable corridor width
                    cx = int(np.mean(best_span))
                    lx = int(best_span[0])
                    rx = int(best_span[-1])
                    last_cx = cx
                    waypoints.append((cx, curr_y))
                    left_bounds.append((lx, curr_y))
                    right_bounds.append((rx, curr_y))
                    clearance_count += 1
                else:
                    break
            else:
                break

        # 6. Calculate Steering Angle & Action Recommendation
        clearance = clearance_count / float(self.num_slices)
        raw_steering = 0.0

        if len(waypoints) >= 2:
            # Weighted average of waypoint offsets (giving more weight to mid-distance horizon)
            offsets = []
            weights = []
            for idx, (pt_x, pt_y) in enumerate(waypoints):
                weight = 1.0 + (idx * 0.5)
                norm_offset = (pt_x - (w / 2.0)) / (w / 2.0)
                offsets.append(norm_offset * weight)
                weights.append(weight)
            raw_steering = float(np.clip(sum(offsets) / sum(weights), -1.0, 1.0))
        elif len(waypoints) == 1:
            raw_steering = float(np.clip((waypoints[0][0] - (w / 2.0)) / (w / 2.0), -1.0, 1.0))
        else:
            raw_steering = 0.0

        # Temporal smoothing
        self.smoothed_steering = (self.history_weight * self.smoothed_steering +
                                  (1.0 - self.history_weight) * raw_steering)

        # Action logic
        if clearance < 0.25:
            action = "OBSTACLE AHEAD"
            status_color = (0, 0, 255)       # Red
        elif self.smoothed_steering < -0.35:
            action = "STEER HARD LEFT"
            status_color = (0, 165, 255)     # Orange
        elif self.smoothed_steering < -0.12:
            action = "STEER LEFT"
            status_color = (0, 220, 255)     # Yellow
        elif self.smoothed_steering > 0.35:
            action = "STEER HARD RIGHT"
            status_color = (0, 165, 255)     # Orange
        elif self.smoothed_steering > 0.12:
            action = "STEER RIGHT"
            status_color = (0, 220, 255)     # Yellow
        else:
            action = "FORWARD"
            status_color = (0, 255, 120)     # Green

        # 7. Render Rich Augmented Reality Overlay
        overlay = bgr_frame.copy()

        # Draw transparent drivable corridor polygon
        if len(left_bounds) >= 2 and len(right_bounds) >= 2:
            poly_pts = left_bounds + right_bounds[::-1]
            poly_np = np.array(poly_pts, dtype=np.int32)
            corridor_layer = np.zeros_like(bgr_frame)
            cv2.fillPoly(corridor_layer, [poly_np], (0, 230, 100))
            # Blend layer
            cv2.addWeighted(corridor_layer, 0.35, overlay, 1.0, 0, overlay)
            # Outline corridor
            cv2.polylines(overlay, [np.array(left_bounds, dtype=np.int32)], False, (0, 255, 180), 2)
            cv2.polylines(overlay, [np.array(right_bounds, dtype=np.int32)], False, (0, 255, 180), 2)

        # Draw Waypoint Path & Guidance Trajectory
        if len(waypoints) >= 2:
            for i in range(len(waypoints) - 1):
                p1 = waypoints[i]
                p2 = waypoints[i + 1]
                cv2.line(overlay, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(overlay, p1, 5, (255, 255, 0), -1)
            cv2.circle(overlay, waypoints[-1], 6, (0, 200, 255), -1)

        # Draw Target Heading Vector Arrow from bottom center
        base_origin = (int(w * 0.5), int(h * 0.96))
        target_len = int(h * 0.28 * max(0.3, clearance))
        target_x = int(base_origin[0] + self.smoothed_steering * (w * 0.35))
        target_y = int(base_origin[1] - target_len)
        cv2.arrowedLine(overlay, base_origin, (target_x, target_y),
                        status_color, 4, cv2.LINE_AA, tipLength=0.25)

        # 8. HUD & Telemetry Dashboard
        # Top Header Bar
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.line(overlay, (0, 55), (w, 55), status_color, 2)

        # Text Status
        steer_pct = int(self.smoothed_steering * 100)
        hud_text = f"ACTION: {action}"
        hud_metrics = f"Steer: {steer_pct:+03d}% | Clearance: {int(clearance*100)}%"
        cv2.putText(overlay, hud_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(overlay, hud_metrics, (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        # Mini Steering Gauge Bar at top-right
        bar_w = 90
        bar_h = 10
        bar_x = w - bar_w - 15
        bar_y = 20
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        mid_x = bar_x + bar_w // 2
        cv2.line(overlay, (mid_x, bar_y - 2), (mid_x, bar_y + bar_h + 2), (180, 180, 180), 1)
        steer_pos = int(mid_x + (self.smoothed_steering * (bar_w // 2)))
        steer_pos = max(bar_x, min(bar_x + bar_w, steer_pos))
        cv2.circle(overlay, (steer_pos, bar_y + bar_h // 2), 6, status_color, -1)

        return {
            "steering": round(self.smoothed_steering, 3),
            "action": action,
            "clearance": round(clearance, 2),
            "waypoints": waypoints,
            "overlay": overlay,
            "drivable_mask": drivable
        }


if __name__ == "__main__":
    from vision import Camera
    from objects import ObjectDetector, CATEGORY_COLORS

    cam = Camera()
    path_det = PathDetector()
    obj_det = ObjectDetector()

    print("\n[AUTONOMOUS ROBOT PATH & CORRIDOR DETECTOR]")
    print("-> Detects walkable floor path, obstacle clearance, & real-time steering angle.")
    print("-> Objects detected are automatically marked as obstacles.")
    print("Press 'q' or ESC in the window to exit.\n")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue

            # 1. Detect any objects/obstacles
            objects = obj_det.detect_objects(frame)

            # 2. Detect drivable corridor path avoiding obstacles
            nav = path_det.detect_path(frame, detected_objects=objects)
            display = nav["overlay"]

            # Draw detected object bounding boxes over overlay
            for obj in objects:
                x, y, w, h = obj["box"]
                lbl = obj["label"]
                color = CATEGORY_COLORS.get(lbl.split()[0], (0, 200, 255))
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display, f"OBSTACLE: {lbl.upper()}", (x, max(18, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.imshow("Robot Drivable Path & Obstacle Detector", display)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    finally:
        cv2.destroyAllWindows()
        cam.release()
