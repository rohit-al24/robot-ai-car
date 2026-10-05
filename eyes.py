#!/usr/bin/env python3
"""
Cute Futuristic Toy Robot Eyes & Compact Robotic Hands.
Rendered on a glossy black glass face panel with luminous cyan/turquoise emissive lighting.

Features:
- No pupils, no eyelashes, no realistic eyeballs.
- Pure smooth cyan/turquoise luminous vector-like geometry (crescent arcs, pill capsules, soft glow).
- Expressive morphing: Happy (^ ^), Curious, Surprised, Sleepy, Excited/Star, Registered Cute (🥹).
- Compact 5-finger soft-toy robotic hands (White polymer + Dark charcoal + Glowing cyan ring).
- Dynamic high-five slap and two-hand love heart presentation (🫶).
"""
import math
import random
import sys
import time
from PIL import Image, ImageDraw

# Color Palette: Deep Black OLED Glass + Emissive Electric Sky-Blue Core & Subtle Glow
BG = (0, 0, 0)
CYAN_CORE = (0, 180, 255)       # Solid electric sky-blue from reference image
CYAN_GLOW = (0, 110, 190)       # Soft outer glow
CYAN_HIGHLIGHT = (80, 215, 255) # Top soft inner glow
POLYMER_WHITE = (242, 248, 255)
CHARCOAL = (35, 40, 48)
CHARCOAL_LIGHT = (55, 62, 74)
PINK_BLUSH = (255, 65, 135)

# LCD wiring (BCM numbers)
LCD_DC = 5
LCD_RST = 6

# Mood Matrix:
# h/w: scale, happy_arc: upward dome arch factor (0.0 to 1.0)
# angry: inward slant angle cut (0.0 to 1.0), sad: outward droop slant (0.0 to 1.0)
# hands: high-five overlay, heart_hands: 🫶 love heart gesture, dead: X_X shot/dead animation,
# shy: adorable shy blinking with glowing rosy blush cheeks, thumbs: robotic cyber paw thumb praise overlay
MOODS = {
    "neutral":        dict(h=1.00, w=1.00, happy_arc=0.00, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=0.0),
    "happy":          dict(h=0.78, w=1.05, happy_arc=1.00, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=0.0),
    "cute":           dict(h=0.92, w=1.05, happy_arc=0.65, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.8, thumbs=0.0, surrender=0.0),
    "cute_blush":     dict(h=0.90, w=1.08, happy_arc=0.85, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=1.0, thumbs=0.0, surrender=0.0),
    "registered_shy": dict(h=0.92, w=1.05, happy_arc=0.60, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=1.0, thumbs=0.0, surrender=0.0),
    "registered_happy":dict(h=0.82, w=1.12, happy_arc=0.95, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.6, thumbs=0.0, surrender=0.0),
    "registered_cute":dict(h=0.92, w=1.05, happy_arc=0.60, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=1.0, thumbs=0.0, surrender=0.0),
    "high_five":      dict(h=0.50, w=0.60, happy_arc=0.80, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=1.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=0.0),
    "heart_hands":    dict(h=0.60, w=0.75, happy_arc=0.85, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=1.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=0.0),
    "surrender":      dict(h=1.35, w=0.95, happy_arc=0.00, hl=1.0, hr=1.0, angry=0.00, sad=0.40, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=1.0),
    "praise_thumbs":  dict(h=0.50, w=0.60, happy_arc=0.00, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=1.0, surrender=0.0),
    "dead":           dict(h=0.80, w=0.80, happy_arc=0.00, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=1.0, shy=0.0, thumbs=0.0, surrender=0.0),
    "sleepy":         dict(h=0.20, w=0.95, happy_arc=0.00, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=0.0),
    "surprised":      dict(h=1.35, w=0.90, happy_arc=0.00, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=0.0),
    "curious":        dict(h=1.00, w=1.00, happy_arc=0.15, hl=1.25, hr=0.75, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=0.0),
    "star":           dict(h=1.10, w=1.10, happy_arc=0.50, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=0.0),
    "cute_surprised": dict(h=1.30, w=1.05, happy_arc=0.20, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=0.0),
    "wink":           dict(h=1.00, w=1.00, happy_arc=0.00, hl=1.15, hr=0.15, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=0.0),
    "love":           dict(h=0.95, w=1.08, happy_arc=0.75, hl=1.0, hr=1.0, angry=0.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.8, thumbs=0.0, surrender=0.0),
    "angry":          dict(h=0.88, w=1.05, happy_arc=0.00, hl=1.0, hr=1.0, angry=1.00, sad=0.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=0.0),
    "sad":            dict(h=0.92, w=0.98, happy_arc=0.00, hl=1.0, hr=1.0, angry=0.00, sad=1.00, hands=0.0, heart_hands=0.0, dead=0.0, shy=0.0, thumbs=0.0, surrender=0.0),
}


class Eyes:
    def __init__(self, width=240, height=240):
        self.w, self.h = width, height
        self.mood = "neutral"
        self.cur = dict(MOODS["neutral"])
        self.hand_side = "both"   # "left", "right", or "both"
        self.gx = self.gy = 0.0
        self.tx = self.ty = 0.0
        self.auto_glance = True
        self.blink_start = None
        self.next_blink = time.time() + 3
        self.next_glance = time.time() + 2
        self.anim_t = 0.0
        self.dead_start = None
        self.dead_duration = 3.5  # Dramatic play-dead fall down, dizzy eyes, then revive!
        self.praise_start = None  # Sequence: Surprised -> Hide eyes & show thumbs up -> Back to normal!

    def set_mood(self, mood, hand_side="both"):
        if mood in MOODS:
            if mood == "dead" and self.mood != "dead":
                self.dead_start = time.time()
            elif mood != "dead":
                self.dead_start = None

            if mood != "praise_thumbs":
                self.praise_start = None

            self.mood = mood
            self.hand_side = hand_side

    def trigger_praise_sequence(self):
        """
        Triggers the praise sequence:
        Directly transitions into cyber thumbs-up with full animation.
        """
        self.praise_start = time.time()
        self.set_mood("praise_thumbs")

    def trigger_shot(self):
        """Triggers dramatic play-dead reaction from a pistol finger shot."""
        self.set_mood("dead")

    def look(self, x, y):
        self.auto_glance = False
        self.tx = max(-1.0, min(1.0, x))
        self.ty = max(-1.0, min(1.0, y))

    def look_auto(self):
        self.auto_glance = True

    def _draw_star_sparkle(self, draw, cx, cy, size, color=CYAN_CORE):
        """Draws a subtle 4-point sparkle highlight."""
        points = []
        angle = (self.anim_t * 2.5) % (2 * math.pi)
        for i in range(8):
            r = size if (i % 2 == 0) else (size * 0.30)
            a = angle + i * (math.pi / 4)
            points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(points, fill=color)

    def _draw_cute_heart(self, draw, cx, cy, size, color):
        """Draws a smooth cute geometric mini heart."""
        points = []
        for t_deg in range(0, 360, 15):
            t = math.radians(t_deg)
            x = 16 * (math.sin(t) ** 3)
            y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
            points.append((cx + (x / 16.0) * size, cy + (y / 16.0) * size))
        draw.polygon(points, fill=color)

    def _draw_cyber_robotic_paw(self, draw, cx, cy, scale, is_left, angle_rad):
        """
        Renders the exact cyan glowing cyber robotic paw design from reference image:
        - Semi-translucent glowing cyan fill with bright cyan outline borders.
        - Curved rounded palm base plate.
        - Two large central segmented finger capsules with pointed tips.
        - Angled thumb fin on the outer/inner side.
        """
        def rot(px, py):
            dx, dy = px - cx, py - cy
            rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            return (cx + rx, cy + ry)

        def draw_glowing_poly(poly_pts):
            # Outer glow aura outline
            draw.polygon(poly_pts, outline=CYAN_GLOW, width=int(max(3, 4 * scale)), fill=(0, 60, 110))
            # Bright cyan border
            draw.polygon(poly_pts, outline=CYAN_CORE, width=int(max(2, 2.2 * scale)), fill=(0, 130, 210))

        sign = 1 if is_left else -1

        # 1. Palm Base (Rounded trapezoid plate)
        pw, ph = scale * 26, scale * 16
        palm_y = cy + scale * 12
        p_corners = [
            rot(cx - pw * 0.45, palm_y - ph * 0.5),
            rot(cx + pw * 0.45, palm_y - ph * 0.5),
            rot(cx + pw * 0.50, palm_y + ph * 0.5),
            rot(cx - pw * 0.50, palm_y + ph * 0.5),
        ]
        draw_glowing_poly(p_corners)

        # 2. Main Double Segmented Finger Capsules (Two vertical capsules with pointed crowns)
        for f_idx, x_off in enumerate([-scale * 5.8, scale * 5.8]):
            fw = scale * 5.2
            fh = scale * 18.0
            fx = cx + x_off
            fy = cy - scale * 2.0

            # Finger outline polygon with tapered crown
            f_pts = [
                rot(fx - fw, fy + fh * 0.48),
                rot(fx - fw, fy - fh * 0.25),
                rot(fx, fy - fh * 0.52),        # Pointed apex
                rot(fx + fw, fy - fh * 0.25),
                rot(fx + fw, fy + fh * 0.48),
            ]
            draw_glowing_poly(f_pts)

            # Center internal joint divider line
            j_p1 = rot(fx - fw * 0.85, fy)
            j_p2 = rot(fx + fw * 0.85, fy)
            draw.line([j_p1, j_p2], fill=CYAN_CORE, width=int(max(2, 2 * scale)))

        # 3. Angled Thumb Claw (Tapered side fin)
        th_w, th_h = scale * 4.5, scale * 13.0
        th_cx = cx + sign * scale * 13.5
        th_cy = cy + scale * 4.0
        th_pts = [
            rot(th_cx - th_w * sign * 0.5, th_cy + th_h * 0.45),
            rot(th_cx + th_w * sign * 0.9, th_cy - th_h * 0.10),
            rot(th_cx + th_w * sign * 1.2, th_cy - th_h * 0.45), # Sharp claw tip
            rot(th_cx - th_w * sign * 0.2, th_cy - th_h * 0.20),
        ]
        draw_glowing_poly(th_pts)

    def _draw_high_five_overlay(self, draw):
        """Draws dynamic single or dual cyber robotic paws high-five slap with impact burst."""
        hands_val = self.cur.get("hands", 0.0)
        if hands_val < 0.05:
            return

        alpha = hands_val
        slap_phase = math.sin(self.anim_t * 6.5)
        scale = (self.w / 240.0) * alpha * 1.35
        hy = self.h * 0.62 - math.sin(self.anim_t * 3.0) * (self.h * 0.04)

        show_left = self.hand_side in ("left", "both")
        show_right = self.hand_side in ("right", "both")

        if show_left and show_right:
            clap_dist = (1.0 - max(0.0, slap_phase)) * (self.w * 0.22)
            left_hx = self.w * 0.26 + clap_dist * 0.5
            right_hx = self.w * 0.74 - clap_dist * 0.5
            left_angle = math.radians(18 - slap_phase * 18)
            right_angle = math.radians(-18 + slap_phase * 18)

            self._draw_cyber_robotic_paw(draw, left_hx, hy, scale, is_left=True, angle_rad=left_angle)
            self._draw_cyber_robotic_paw(draw, right_hx, hy, scale, is_left=False, angle_rad=right_angle)

            if slap_phase > 0.65:
                impact_cx = self.w * 0.5
                impact_cy = hy - self.h * 0.14
                spark_size = (self.w * 0.18) * (slap_phase ** 2) * alpha
                ring_r = spark_size * 1.4
                draw.ellipse((impact_cx - ring_r, impact_cy - ring_r,
                              impact_cx + ring_r, impact_cy + ring_r), outline=CYAN_CORE, width=int(2*scale))
                self._draw_star_sparkle(draw, impact_cx, impact_cy, spark_size, (255, 255, 255))
                self._draw_star_sparkle(draw, impact_cx - spark_size * 0.8, impact_cy - spark_size * 0.5, spark_size * 0.45, CYAN_CORE)
                self._draw_star_sparkle(draw, impact_cx + spark_size * 0.8, impact_cy - spark_size * 0.5, spark_size * 0.45, CYAN_CORE)
        elif show_left:
            reach = max(0.0, slap_phase) * (self.w * 0.28)
            left_hx = self.w * 0.22 + reach
            left_angle = math.radians(24 - slap_phase * 24)
            self._draw_cyber_robotic_paw(draw, left_hx, hy, scale, is_left=True, angle_rad=left_angle)
            if slap_phase > 0.60:
                impact_cx = left_hx + self.w * 0.10
                impact_cy = hy - self.h * 0.08
                spark_size = (self.w * 0.16) * (slap_phase ** 2) * alpha
                self._draw_star_sparkle(draw, impact_cx, impact_cy, spark_size, (255, 255, 255))
        elif show_right:
            reach = max(0.0, slap_phase) * (self.w * 0.28)
            right_hx = self.w * 0.78 - reach
            right_angle = math.radians(-24 + slap_phase * 24)
            self._draw_cyber_robotic_paw(draw, right_hx, hy, scale, is_left=False, angle_rad=right_angle)
            if slap_phase > 0.60:
                impact_cx = right_hx - self.w * 0.10
                impact_cy = hy - self.h * 0.08
                spark_size = (self.w * 0.16) * (slap_phase ** 2) * alpha
                self._draw_star_sparkle(draw, impact_cx, impact_cy, spark_size, (255, 255, 255))

    def _draw_heart_hands_overlay(self, draw):
        """Draws cyber paws forming a Heart Gesture (🫶) with pulsing glowing love heart."""
        heart_val = self.cur.get("heart_hands", 0.0)
        if heart_val < 0.05:
            return

        alpha = heart_val
        scale = (self.w / 240.0) * alpha * 1.30

        left_hx = self.w * 0.36
        right_hx = self.w * 0.64
        hy = self.h * 0.68 - math.sin(self.anim_t * 3.0) * (self.h * 0.03)

        self._draw_cyber_robotic_paw(draw, left_hx, hy, scale, is_left=True, angle_rad=math.radians(35))
        self._draw_cyber_robotic_paw(draw, right_hx, hy, scale, is_left=False, angle_rad=math.radians(-35))

        # Center Pulsing Love Heart
        pulse = math.sin(self.anim_t * 5.5) * 0.12
        h_size = (self.w * 0.15 * (1.0 + pulse)) * alpha
        hcx = self.w * 0.5
        hcy = hy - self.h * 0.15

        self._draw_cute_heart(draw, hcx, hcy, h_size * 1.30, (255, 120, 180))
        self._draw_cute_heart(draw, hcx, hcy, h_size, (255, 40, 115))
        self._draw_cute_heart(draw, hcx, hcy - h_size * 0.15, h_size * 0.45, (255, 200, 230))

        # Floating Mini Love Hearts drifting outwards to give to the user
        for fi in range(3):
            t_drift = (self.anim_t * 1.6 + fi * 0.85) % 2.5
            progress = t_drift / 2.5
            drift_x = hcx + math.sin(t_drift * 3.5 + fi * 2.0) * (self.w * 0.22)
            drift_y = hcy - progress * (self.h * 0.35)
            drift_scale = (self.w * 0.05) * (1.0 - progress * 0.5) * alpha
            if drift_y > self.h * 0.10:
                self._draw_cute_heart(draw, drift_x, drift_y, drift_scale, (255, 100, 180))
                self._draw_star_sparkle(draw, drift_x + drift_scale * 0.8, drift_y - drift_scale * 0.5, drift_scale * 0.4, (255, 255, 255))

    def _draw_cyber_thumbs_up(self, draw):
        """Draws clean glowing cyber robotic fist with an upright thumb giving a thumbs-up (👍) with no stars."""
        thumbs_val = self.cur.get("thumbs", 0.0)
        if thumbs_val < 0.05:
            return

        alpha = thumbs_val
        # Smooth energetic pump / nod animation
        pump = math.sin(self.anim_t * 5.0) * (self.h * 0.04)
        scale = (self.w / 240.0) * alpha * 1.55
        cx = self.w * 0.50
        cy = self.h * 0.56 + pump

        def draw_glowing_poly(poly_pts):
            draw.polygon(poly_pts, outline=CYAN_GLOW, width=int(max(3, 4 * scale)), fill=(0, 60, 110))
            draw.polygon(poly_pts, outline=CYAN_CORE, width=int(max(2, 2.2 * scale)), fill=(0, 140, 220))

        # 1. Closed Cyber Palm Base (Trapezoid core)
        pw, ph = scale * 26, scale * 16
        p_corners = [
            (cx - pw * 0.45, cy + ph * 0.05),
            (cx + pw * 0.45, cy + ph * 0.05),
            (cx + pw * 0.50, cy + ph * 0.85),
            (cx - pw * 0.50, cy + ph * 0.85),
        ]
        draw_glowing_poly(p_corners)

        # 2. Curled Horizontal Finger Pods (Fist knuckles)
        for f_idx, y_off in enumerate([scale * 1.0, scale * 7.5]):
            fx = cx + scale * 3.0
            fy = cy + y_off
            fw = scale * 9.0
            fh = scale * 2.8
            f_pts = [
                (fx - fw, fy - fh),
                (fx + fw, fy - fh),
                (fx + fw, fy + fh),
                (fx - fw, fy + fh),
            ]
            draw_glowing_poly(f_pts)

        # 3. Big Upright Cyber Thumb (Raised high & proud, pointing UP)
        th_w = scale * 5.5
        th_h = scale * 24.0
        th_cx = cx - scale * 6.5
        th_cy = cy - scale * 8.5

        thumb_pts = [
            (th_cx - th_w, th_cy + th_h * 0.45),
            (th_cx - th_w, th_cy - th_h * 0.28),
            (th_cx, th_cy - th_h * 0.55),        # Sharp apex tip
            (th_cx + th_w, th_cy - th_h * 0.28),
            (th_cx + th_w, th_cy + th_h * 0.45),
        ]
        draw_glowing_poly(thumb_pts)

        # Segment divider on thumb
        draw.line([(th_cx - th_w * 0.8, th_cy - scale * 1.5),
                   (th_cx + th_w * 0.8, th_cy - scale * 1.5)], fill=CYAN_CORE, width=int(max(2, 2 * scale)))
        draw.line([(th_cx - th_w * 0.8, th_cy + scale * 5.5),
                   (th_cx + th_w * 0.8, th_cy + scale * 5.5)], fill=CYAN_CORE, width=int(max(2, 2 * scale)))

        # Clean energy aura ring pulses (No stars)
        pulse_r = (scale * 14.0) + ((self.anim_t * 12.0) % (scale * 16.0))
        draw.arc((th_cx - pulse_r, th_cy - th_h * 0.55 - pulse_r * 0.6,
                  th_cx + pulse_r, th_cy - th_h * 0.55 + pulse_r * 0.6),
                 start=180, end=360, fill=CYAN_HIGHLIGHT, width=int(max(1, 1.8 * scale)))

    def _draw_surrender_hands_overlay(self, draw):
        """Draws two cyber robotic paws raised up high in fear/surrender (Hands Up!) when pistol is aimed."""
        surrender_val = self.cur.get("surrender", 0.0)
        if surrender_val < 0.05:
            return

        alpha = surrender_val
        tremble = math.sin(self.anim_t * 18.0) * (self.w * 0.015)  # Scared shivering/trembling!
        scale = (self.w / 240.0) * alpha * 1.25
        hy = self.h * 0.46 + tremble

        left_hx = self.w * 0.22 + tremble
        right_hx = self.w * 0.78 - tremble

        # Two hands raised up high facing the user
        self._draw_cyber_robotic_paw(draw, left_hx, hy, scale, is_left=True, angle_rad=math.radians(-12))
        self._draw_cyber_robotic_paw(draw, right_hx, hy, scale, is_left=False, angle_rad=math.radians(12))

    def _draw_dead_x_eyes(self, draw, cx, cy, ew, eh):
        """Draws dizzy X_X comic play-dead knock-out eyes."""
        rx = ew * 0.40
        ry = eh * 0.40
        stroke_w = int(max(4, ew * 0.18))
        glow_w = stroke_w + 3

        # Glow X
        draw.line([(cx - rx, cy - ry), (cx + rx, cy + ry)], fill=CYAN_GLOW, width=glow_w)
        draw.line([(cx + rx, cy - ry), (cx - rx, cy + ry)], fill=CYAN_GLOW, width=glow_w)

        # Crisp Cyan Core X
        draw.line([(cx - rx, cy - ry), (cx + rx, cy + ry)], fill=CYAN_CORE, width=stroke_w)
        draw.line([(cx + rx, cy - ry), (cx - rx, cy + ry)], fill=CYAN_CORE, width=stroke_w)

    def _draw_eilik_eye(self, draw, cx, cy, ew, eh, is_left, happy_arc, angry_val, sad_val):
        r"""
        Renders an exact Eilik-style solid sky-blue OLED robot eye:
        1. Neutral: Solid vertical squircle/oval pill.
        2. Happy: Upward-curved crescent arch dome (^ ^) with flat or slightly curved bottom.
        3. Angry: Fierce inward diagonal slant slice cut (\ /).
        4. Sad: Melancholy outward diagonal droop slant (/ \).
        """
        rx = ew / 2.0
        ry = eh / 2.0

        if happy_arc > 0.35:
            # === HAPPY / SMILING DOME ARCH (Reference Top-Right) ===
            # Creates an upward curved dome with a flat/slightly arched base
            num_pts = 36
            pts = []
            # Upper dome: semicircle from angle 180° to 360° (top curve)
            # Base: slightly scooped or flat bottom line
            arch_factor = min(1.0, happy_arc)
            # Dome top
            for deg in range(180, 361, 10):
                rad = math.radians(deg)
                px = cx + rx * math.cos(rad)
                py = (cy + ry * 0.20) + (ry * 1.05) * math.sin(rad)
                pts.append((px, py))
            # Base bottom arch (soft upward concavity for smile)
            for deg in range(0, 181, 15):
                rad = math.radians(deg)
                px = cx + rx * math.cos(rad)
                py = (cy + ry * 0.20) + (ry * 0.25 * (1.0 - arch_factor)) * math.sin(rad)
                pts.append((px, py))

            # Outer subtle glow + solid fill
            glow_pts = [(cx + (px - cx) * 1.08, cy + (py - cy) * 1.08) for px, py in pts]
            draw.polygon(glow_pts, fill=CYAN_GLOW)
            draw.polygon(pts, fill=CYAN_CORE)

        elif angry_val > 0.25:
            # === ANGRY INWARD SLANT WEDGE (Reference Bottom-Left) ===
            # Slanted top: left eye has top-right cut low, right eye has top-left cut low
            num_pts = 32
            pts = []
            slant = min(1.0, angry_val)
            cut_drop = ry * 0.95 * slant

            for deg in range(0, 360, 12):
                rad = math.radians(deg)
                px = cx + rx * math.cos(rad)
                py = cy + ry * math.sin(rad)
                
                # Apply inward top slant
                if is_left:
                    # Inner corner (right side of left eye) slants down
                    norm_x = (px - (cx - rx)) / (2 * rx)  # 0 (outer) to 1 (inner)
                    roof_y = (cy - ry) + cut_drop * norm_x
                else:
                    # Inner corner (left side of right eye) slants down
                    norm_x = 1.0 - ((px - (cx - rx)) / (2 * rx)) # 1 (inner) to 0 (outer)
                    roof_y = (cy - ry) + cut_drop * norm_x

                if py < roof_y:
                    py = roof_y
                pts.append((px, py))

            glow_pts = [(cx + (px - cx) * 1.07, cy + (py - cy) * 1.07) for px, py in pts]
            draw.polygon(glow_pts, fill=CYAN_GLOW)
            draw.polygon(pts, fill=CYAN_CORE)

        elif sad_val > 0.25:
            # === SAD OUTWARD DROOP SLANT (Reference Bottom-Right) ===
            # Slanted top: left eye has top-left (outer) cut low, right eye has top-right (outer) cut low
            num_pts = 32
            pts = []
            droop = min(1.0, sad_val)
            droop_drop = ry * 0.90 * droop

            for deg in range(0, 360, 12):
                rad = math.radians(deg)
                px = cx + rx * math.cos(rad)
                py = cy + ry * math.sin(rad)

                if is_left:
                    # Outer corner (left side of left eye) droops down
                    norm_x = 1.0 - ((px - (cx - rx)) / (2 * rx))  # 1 (outer) to 0 (inner)
                    roof_y = (cy - ry) + droop_drop * norm_x
                else:
                    # Outer corner (right side of right eye) droops down
                    norm_x = (px - (cx - rx)) / (2 * rx)  # 0 (inner) to 1 (outer)
                    roof_y = (cy - ry) + droop_drop * norm_x

                if py < roof_y:
                    py = roof_y
                pts.append((px, py))

            glow_pts = [(cx + (px - cx) * 1.07, cy + (py - cy) * 1.07) for px, py in pts]
            draw.polygon(glow_pts, fill=CYAN_GLOW)
            draw.polygon(pts, fill=CYAN_CORE)

        else:
            # === NEUTRAL / DEFAULT SMOOTH OVAL PILL (Reference Top-Left) ===
            glow_pad = max(2, int(ew * 0.08))
            draw.ellipse((cx - rx - glow_pad, cy - ry - glow_pad,
                          cx + rx + glow_pad, cy + ry + glow_pad), fill=CYAN_GLOW)
            draw.ellipse((cx - rx, cy - ry,
                          cx + rx, cy + ry), fill=CYAN_CORE)

    def render(self):
        now = time.time()
        self.anim_t += 0.05

        # Smooth mood transition
        for k, v in MOODS[self.mood].items():
            self.cur[k] += (v - self.cur[k]) * 0.28

        # Random glances when idle
        if self.auto_glance:
            if self.mood in ("neutral", "curious", "star", "cute_surprised", "cute", "touched_teary", "registered_cute"):
                if now > self.next_glance:
                    self.tx = random.uniform(-0.6, 0.6)
                    self.ty = random.uniform(-0.30, 0.30)
                    self.next_glance = now + random.uniform(1.8, 4.5)
            else:
                self.tx = self.ty = 0.0
        self.gx += (self.tx - self.gx) * 0.22
        self.gy += (self.ty - self.gy) * 0.22

        # Fluttery natural blinking
        if self.blink_start is None and now > self.next_blink and self.mood != "sleepy":
            self.blink_start = now
        blink = 1.0
        if self.blink_start is not None:
            blink_dur = 0.18
            t = (now - self.blink_start) / blink_dur
            if t >= 1:
                self.blink_start = None
                self.next_blink = now + random.uniform(2.5, 6.0)
            else:
                blink = max(abs(2 * t - 1), 0.08)

        # Automatic multi-step Thumb Praise Sequence:
        # Step 1: Surprised reaction for 0.7s
        # Step 2: Hide eyes completely and display glowing Cyber Thumb Up for 2.2s
        # Step 3: Return to normal!
        if self.praise_start is not None:
            p_elapsed = now - self.praise_start
            if p_elapsed < 0.7:
                self.mood = "surprised"
            elif p_elapsed < 2.9:
                self.mood = "praise_thumbs"
            else:
                self.mood = "neutral"
                self.praise_start = None

        # Automatic Recovery from play-dead
        if self.mood == "dead" and self.dead_start is not None:
            elapsed = now - self.dead_start
            if elapsed > self.dead_duration:
                self.set_mood("surprised")  # Wakes up with surprised blink!
                self.dead_start = None

        # 1. Deep Glossy Black Glass OLED Face
        img = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(img)

        # Check if hands, love hearts, or praise thumbs are active (eyes hidden when hands/hearts/thumbs are displayed)
        hands_active = (self.cur.get("hands", 0.0) > 0.15 or self.cur.get("heart_hands", 0.0) > 0.15 or self.cur.get("thumbs", 0.0) > 0.15)
        dead_val = self.cur.get("dead", 0.0)
        shy_val = self.cur.get("shy", 0.0)

        if not hands_active:
            # Sizing tailored to the reference toy robot face proportions
            ew = self.w * 0.27 * self.cur["w"]
            base_h = self.h * 0.40
            cy = self.h * 0.50 + self.gy * self.h * 0.08

            # If dead, drop down dramaticaly with dizzy wobble
            if dead_val > 0.1:
                cy += (self.h * 0.18) * dead_val

            # Rosy Blushing Cheeks for shy & cute expressions (Registered owner face recognition)
            if (shy_val > 0.15 or self.mood in ("cute", "cute_blush", "love", "registered_shy", "registered_cute", "registered_happy")) and dead_val < 0.2:
                b_strength = max(shy_val, 0.75)
                bw, bh = ew * 0.65, base_h * 0.22
                blush_col = (int(PINK_BLUSH[0] * 0.85 * b_strength),
                             int(PINK_BLUSH[1] * 0.40 * b_strength),
                             int(PINK_BLUSH[2] * 0.60 * b_strength))
                for bcx in (self.w * 0.15, self.w * 0.85):
                    bcy = cy + base_h * 0.40
                    d.ellipse((bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2), fill=blush_col)
                    # Subtle glowing star sparkle on shy cheeks
                    if shy_val > 0.6:
                        self._draw_star_sparkle(d, bcx, bcy, ew * 0.15, (255, 200, 230))

            # Floating cute hearts for 'cute', 'love', 'registered_happy'
            if self.mood in ("cute", "cute_blush", "love", "registered_happy") and blink > 0.6 and dead_val < 0.2:
                for hi, h_offset in enumerate([-self.w * 0.22, self.w * 0.22]):
                    hcx = self.w * 0.5 + h_offset
                    hcy = cy - base_h * 0.65 + math.sin(self.anim_t * 3.0 + hi) * (self.h * 0.04)
                    self._draw_cute_heart(d, hcx, hcy, self.w * 0.045, (255, 100, 160))

            # 2. Render Reference Luminous Sky-Blue Eyes / Dead X_X Eyes
            happy_arc = self.cur.get("happy_arc", 0.0)
            angry_val = self.cur.get("angry", 0.0)
            sad_val = self.cur.get("sad", 0.0)

            for i, (cx_base, hm) in enumerate(((self.w * 0.31, self.cur["hl"]), (self.w * 0.69, self.cur["hr"]))):
                is_left = (i == 0)
                cx = cx_base + self.gx * self.w * 0.07
                eh = max(base_h * self.cur["h"] * hm * blink, 4)

                if dead_val > 0.4:
                    self._draw_dead_x_eyes(d, cx, cy, ew, eh)
                else:
                    self._draw_eilik_eye(d, cx, cy, ew, eh, is_left, happy_arc, angry_val, sad_val)

        # 3. Render High-Five Hands, Two-Hand Hearts, Surrender Hands & Cyber Thumbs-Up (Exclusively shown when active)
        self._draw_high_five_overlay(d)
        self._draw_heart_hands_overlay(d)
        self._draw_surrender_hands_overlay(d)
        self._draw_cyber_thumbs_up(d)

        return img


def run_tk(eyes, tick=None, fps=20, scale=2):
    import tkinter as tk
    from PIL import ImageTk
    root = tk.Tk()
    root.title("Toy Robot Eyes (Pixar & Eilik 3D Style)")
    label = tk.Label(root, bg="black")
    label.pack()
    delay = int(1000 / fps)

    def step():
        if tick:
            tick()
        img = eyes.render().resize((eyes.w * scale, eyes.h * scale))
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        label.image = photo
        root.after(delay, step)

    step()
    root.mainloop()


def run_lcd(eyes, tick=None, fps=20):
    from luma.core.interface.serial import spi
    from luma.lcd.device import st7789
    serial = spi(port=0, device=0, gpio_DC=LCD_DC, gpio_RST=LCD_RST,
                 bus_speed_hz=32000000)
    device = st7789(serial, width=eyes.w, height=eyes.h, rotate=0)
    try:
        while True:
            if tick:
                tick()
            device.display(eyes.render())
            time.sleep(1 / fps)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    eyes = Eyes()
    names = list(MOODS)
    st = {"i": 0, "t": time.time()}

    def tick():
        if time.time() - st["t"] > 3:
            st["t"] = time.time()
            st["i"] = (st["i"] + 1) % len(names)
            eyes.set_mood(names[st["i"]])
            print("mood:", names[st["i"]])

    (run_lcd if "--lcd" in sys.argv else run_tk)(eyes, tick)
