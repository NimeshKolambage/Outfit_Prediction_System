"""
Body Measurements Module using MediaPipe Pose Landmarks
Detects shoulder width, chest size, and arm length from camera feed.
Works alongside gender_detector.py and server.py.
All processing runs on Flask-served frames — no separate OpenCV windows.

Landmark indices used (MediaPipe Pose):
  11 LEFT_SHOULDER   12 RIGHT_SHOULDER
  13 LEFT_ELBOW      14 RIGHT_ELBOW
  15 LEFT_WRIST      16 RIGHT_WRIST
  23 LEFT_HIP        24 RIGHT_HIP

Calibration strategy:
  Instead of a fixed PIXEL_TO_CM constant (which depends on distance to camera),
  we use the shoulder-to-hip torso height as a biological reference.
  Average torso height (shoulder→hip) ≈ 42 cm (male) / 39 cm (female).
  pixel_to_cm = reference_cm / torso_px  — recalculated every frame.
"""

import cv2
import numpy as np
import math
import os
import glob
import sys
import types

# ── MediaPipe import with TensorFlow bypass ───────────────────
HAS_MEDIAPIPE = False
mp_pose = None
mp_drawing = None

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    HAS_MEDIAPIPE = True
except (ImportError, AttributeError, Exception):
    try:
        for mod_name in ('mediapipe.tasks', 'mediapipe.tasks.python'):
            if mod_name not in sys.modules:
                sys.modules[mod_name] = types.ModuleType(mod_name)
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        HAS_MEDIAPIPE = True
        print("[OK] MediaPipe loaded (TF-tasks bypassed)")
    except Exception as e2:
        print(f"[WARNING] MediaPipe not available: {e2}")
        HAS_MEDIAPIPE = False


# ── Drawing helpers ───────────────────────────────────────────

def _draw_dotted_line(img, pt1, pt2, color, thickness=2, gap=10):
    """Draw a dotted line between two (x,y) points."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    steps = int(length / gap)
    for i in range(steps + 1):
        t = i / max(steps, 1)
        x = int(pt1[0] + dx * t)
        y = int(pt1[1] + dy * t)
        if i % 2 == 0:
            cv2.circle(img, (x, y), thickness, color, -1)


def _draw_landmark_dot(img, cx, cy, radius=7, color=(0, 255, 255), border=(255, 255, 255)):
    """Draw a prominent dot with a white border at a landmark."""
    cv2.circle(img, (cx, cy), radius + 2, border, -1)
    cv2.circle(img, (cx, cy), radius, color, -1)


def _put_label(img, text, x, y, color=(255, 255, 255), bg=(0, 0, 0), scale=0.5):
    """Put text with a dark background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 4, y + baseline + 2), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


# ── Reference torso height (shoulder–hip) in cm ──────────────
REFERENCE_TORSO_CM = {"Male": 42.0, "Female": 39.0}


class BodyMeasurements:
    """Measure shoulder width, chest circumference (est.), and arm length
    using MediaPipe Pose landmarks from a webcam frame.

    Dynamic calibration:  pixel_to_cm is recalculated each frame using the
    shoulder–hip distance (torso height) as a biological reference.
    """

    # Pose landmark indices
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW     = 13
    RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15
    RIGHT_WRIST    = 16
    LEFT_HIP       = 23
    RIGHT_HIP      = 24

    # Size chart (cm) — simplified male/female charts
    SIZE_CHARTS = {
        "Male": {
            "XS": {"shoulder": (38, 41), "chest": (81, 86),  "arm": (58, 61)},
            "S":  {"shoulder": (41, 44), "chest": (86, 91),  "arm": (61, 63)},
            "M":  {"shoulder": (44, 47), "chest": (91, 101), "arm": (63, 65)},
            "L":  {"shoulder": (47, 50), "chest": (101, 111),"arm": (65, 67)},
            "XL": {"shoulder": (50, 54), "chest": (111, 121),"arm": (67, 70)},
        },
        "Female": {
            "XS": {"shoulder": (34, 37), "chest": (76, 81),  "arm": (53, 56)},
            "S":  {"shoulder": (37, 39), "chest": (81, 88),  "arm": (56, 58)},
            "M":  {"shoulder": (39, 42), "chest": (88, 96),  "arm": (58, 60)},
            "L":  {"shoulder": (42, 45), "chest": (96, 104), "arm": (60, 62)},
            "XL": {"shoulder": (45, 48), "chest": (104, 114),"arm": (62, 65)},
        },
    }

    def __init__(self):
        self.pose = None
        if HAS_MEDIAPIPE:
            try:
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("[OK] BodyMeasurements: MediaPipe Pose initialized")
            except Exception as e:
                print(f"[ERROR] BodyMeasurements: Pose init failed: {e}")
        else:
            print("[WARNING] BodyMeasurements: MediaPipe not available")

        # Smoothing buffers (rolling average over N frames)
        self._buf_shoulder = []
        self._buf_chest = []
        self._buf_arm = []
        self._buf_scale = []
        self._buf_max = 12  # frames to average

        # Store the gender for dynamic calibration
        self._current_gender = "Male"

    # ── helpers ──

    @staticmethod
    def _dist(p1, p2, w, h):
        """Euclidean distance between two landmarks in pixel space."""
        return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

    @staticmethod
    def _pt(landmark, w, h):
        """Return (x_px, y_px) integer coords of a landmark."""
        return (int(landmark.x * w), int(landmark.y * h))

    def _smooth(self, buf, value):
        buf.append(value)
        if len(buf) > self._buf_max:
            buf.pop(0)
        return round(sum(buf) / len(buf), 1)

    # ── core measurement ──

    def measure(self, frame_bgr, gender="Male", sleeve_mode="long"):
        """Run pose estimation and return measurements dict.

        Works even with partial body visibility (e.g. only upper body in frame).
        - If hips visible → uses torso height for calibration
        - If only shoulders visible → estimates torso from shoulder width
        - Arms are optional; uses whichever arm(s) are visible
        - sleeve_mode: "short" = shoulder→elbow only, "long" = shoulder→elbow→wrist

        Returns dict with keys:
            detected (bool), shoulder_cm, chest_cm, arm_cm,
            landmarks (raw pose_landmarks), points (dict of pixel coords)
        """
        self._current_gender = gender

        result = {
            "detected": False,
            "shoulder_cm": 0,
            "chest_cm": 0,
            "arm_cm": 0,
            "shoulder_px": 0,
            "arm_px": 0,
            "torso_px": 0,
            "pixel_to_cm": 0,
            "points": {},
        }

        if self.pose is None:
            return result

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pose_res = self.pose.process(rgb)

        if pose_res.pose_landmarks is None:
            return result

        lm = pose_res.pose_landmarks.landmark
        h, w = frame_bgr.shape[:2]

        ls = lm[self.LEFT_SHOULDER]
        rs = lm[self.RIGHT_SHOULDER]
        le = lm[self.LEFT_ELBOW]
        re = lm[self.RIGHT_ELBOW]
        lw = lm[self.LEFT_WRIST]
        rw = lm[self.RIGHT_WRIST]
        lh = lm[self.LEFT_HIP]
        rh = lm[self.RIGHT_HIP]

        # --- Only require BOTH SHOULDERS to be minimally visible ---
        shoulders_ok = (ls.visibility > 0.15) and (rs.visibility > 0.15)
        if not shoulders_ok:
            return result

        shoulder_px = self._dist(ls, rs, w, h)
        if shoulder_px < 20:  # too small
            return result

        # ── Dynamic calibration ──────────────────────────
        # Check if hips are visible for torso-based calibration
        hips_ok = (lh.visibility > 0.15) and (rh.visibility > 0.15)

        if hips_ok:
            left_torso_px  = self._dist(ls, lh, w, h)
            right_torso_px = self._dist(rs, rh, w, h)
            torso_px = (left_torso_px + right_torso_px) / 2.0
            if torso_px < 20:
                torso_px = shoulder_px * 1.15  # fallback
        else:
            # Hips not visible — estimate torso from shoulder width
            # Anthropometric ratio: torso height ≈ 1.15 × shoulder width
            torso_px = shoulder_px * 1.15

        ref_cm = REFERENCE_TORSO_CM.get(gender, 42.0)
        px_to_cm = ref_cm / torso_px
        px_to_cm = self._smooth(self._buf_scale, px_to_cm)

        # ── Shoulder width ───────────────────────────────
        shoulder_cm = self._smooth(self._buf_shoulder, round(shoulder_px * px_to_cm, 1))

        # ── Arm length (use whichever arm(s) are visible) ──
        left_arm_vis  = min(le.visibility, lw.visibility)
        right_arm_vis = min(re.visibility, rw.visibility)

        arm_px = 0
        arm_count = 0

        if sleeve_mode == "short":
            # Short sleeve: measure shoulder → elbow only
            if le.visibility > 0.1:
                arm_px += self._dist(ls, le, w, h)
                arm_count += 1
            if re.visibility > 0.1:
                arm_px += self._dist(rs, re, w, h)
                arm_count += 1
            if arm_count > 0:
                arm_px /= arm_count
                arm_cm = self._smooth(self._buf_arm, round(arm_px * px_to_cm, 1))
            else:
                # Estimate: upper arm ≈ 0.55 × full arm ≈ 0.55 × 1.4 × shoulder
                arm_px = shoulder_px * 0.77
                arm_cm = self._smooth(self._buf_arm, round(arm_px * px_to_cm, 1))
        else:
            # Long sleeve: measure shoulder → elbow → wrist (full arm)
            if left_arm_vis > 0.1:
                arm_px += self._dist(ls, le, w, h) + self._dist(le, lw, w, h)
                arm_count += 1
            if right_arm_vis > 0.1:
                arm_px += self._dist(rs, re, w, h) + self._dist(re, rw, w, h)
                arm_count += 1
            if arm_count > 0:
                arm_px /= arm_count
                arm_cm = self._smooth(self._buf_arm, round(arm_px * px_to_cm, 1))
            else:
                # Estimate arm from shoulder width (anthropometric: arm ≈ 1.4 × shoulder)
                arm_px = shoulder_px * 1.4
                arm_cm = self._smooth(self._buf_arm, round(arm_px * px_to_cm, 1))

        # ── Chest circumference estimate ─────────────────
        chest_width_px = shoulder_px
        chest_depth_px = chest_width_px * 0.65  # anthropometric average
        # Ellipse perimeter (Ramanujan): π(3(a+b) - sqrt((3a+b)(a+3b)))
        a = chest_width_px / 2.0
        b = chest_depth_px / 2.0
        chest_circ_px = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
        chest_cm = self._smooth(self._buf_chest, round(chest_circ_px * px_to_cm, 1))

        # ── Collect pixel coordinates for drawing ────────
        points = {
            "ls": self._pt(ls, w, h), "rs": self._pt(rs, w, h),
        }
        # Add optional landmarks only if visible enough
        if le.visibility > 0.1:
            points["le"] = self._pt(le, w, h)
        if re.visibility > 0.1:
            points["re"] = self._pt(re, w, h)
        if lw.visibility > 0.1:
            points["lw"] = self._pt(lw, w, h)
        if rw.visibility > 0.1:
            points["rw"] = self._pt(rw, w, h)
        if lh.visibility > 0.1:
            points["lh"] = self._pt(lh, w, h)
        if rh.visibility > 0.1:
            points["rh"] = self._pt(rh, w, h)

        # Chest midpoint (25% below shoulder line, or use shoulder midpoint)
        s_mid_x = (points["ls"][0] + points["rs"][0]) // 2
        s_mid_y = (points["ls"][1] + points["rs"][1]) // 2
        chest_offset = int(torso_px * 0.25) if hips_ok else int(shoulder_px * 0.3)
        points["chest_mid"] = (s_mid_x, s_mid_y + chest_offset)

        result.update({
            "detected": True,
            "shoulder_cm": shoulder_cm,
            "chest_cm": chest_cm,
            "arm_cm": arm_cm,
            "shoulder_px": round(shoulder_px),
            "arm_px": round(arm_px),
            "torso_px": round(torso_px),
            "pixel_to_cm": round(px_to_cm, 4),
            "landmarks": pose_res.pose_landmarks,
            "points": points,
        })
        return result

    def predict_size(self, measurements, gender="Male"):
        """Given measurements dict and gender, return best-fit size label."""
        if not measurements.get("detected"):
            return "Unknown"

        chart = self.SIZE_CHARTS.get(gender, self.SIZE_CHARTS["Male"])

        shoulder = measurements["shoulder_cm"]
        chest = measurements["chest_cm"]
        arm = measurements["arm_cm"]

        best_size = "M"
        best_score = float("inf")

        for size_label, ranges in chart.items():
            s_mid = sum(ranges["shoulder"]) / 2
            c_mid = sum(ranges["chest"]) / 2
            a_mid = sum(ranges["arm"]) / 2

            score = (
                abs(shoulder - s_mid) * 2.0
                + abs(chest - c_mid) * 1.5
                + abs(arm - a_mid) * 1.0
            )
            if score < best_score:
                best_score = score
                best_size = size_label

        return best_size

    # ── drawing ──

    def draw_measurements(self, frame, measurements, gender="Male", size="M", sleeve_mode="long"):
        """Draw dotted landmark markers, measurement lines, and info panel."""
        h, w = frame.shape[:2]

        if not measurements.get("detected"):
            # Guide text
            cv2.putText(
                frame, "Stand in frame - arms slightly open",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
            )
            cv2.putText(
                frame, "Ensure shoulders, arms & hips are visible",
                (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
            )
            return frame

        pts = measurements.get("points", {})
        lm_data = measurements.get("landmarks")

        # Colors
        COL_SHOULDER = (0, 255, 255)   # cyan
        COL_CHEST    = (0, 200, 100)   # green
        COL_ARM      = (255, 165, 0)   # orange
        COL_DOT      = (0, 255, 255)   # landmark dots
        COL_SKEL     = (80, 80, 80)    # faint skeleton

        # 1) Draw faint skeleton in background (if mediapipe available)
        if HAS_MEDIAPIPE and lm_data is not None:
            if sleeve_mode == "short":
                # Filter out connections past elbow (wrist/hand landmarks: 15-22)
                # Keep only connections where BOTH endpoints are <= 14 (elbow) or >= 23 (hips+)
                WRIST_HAND_IDS = {15, 16, 17, 18, 19, 20, 21, 22}
                filtered_connections = [
                    c for c in mp_pose.POSE_CONNECTIONS
                    if c[0] not in WRIST_HAND_IDS and c[1] not in WRIST_HAND_IDS
                ]
                mp_drawing.draw_landmarks(
                    frame, lm_data, filtered_connections,
                    mp_drawing.DrawingSpec(color=COL_SKEL, thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=COL_SKEL, thickness=1),
                )
            else:
                mp_drawing.draw_landmarks(
                    frame, lm_data, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COL_SKEL, thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=COL_SKEL, thickness=1),
                )

        # 2) Dotted measurement lines + landmark dots

        # --- Shoulder line (left shoulder ↔ right shoulder) ---
        if "ls" in pts and "rs" in pts:
            _draw_dotted_line(frame, pts["ls"], pts["rs"], COL_SHOULDER, thickness=3, gap=8)
            _draw_landmark_dot(frame, *pts["ls"], radius=6, color=COL_SHOULDER)
            _draw_landmark_dot(frame, *pts["rs"], radius=6, color=COL_SHOULDER)
            # Label at midpoint
            mid_x = (pts["ls"][0] + pts["rs"][0]) // 2
            mid_y = (pts["ls"][1] + pts["rs"][1]) // 2 - 15
            _put_label(frame, f"Shoulder: {measurements['shoulder_cm']} cm",
                       mid_x - 60, mid_y, COL_SHOULDER)

        # --- Chest area (dotted arc across chest) ---
        if "chest_mid" in pts and "ls" in pts and "rs" in pts:
            cm = pts["chest_mid"]
            # Draw dotted horizontal line at chest level from left to right
            chest_left  = (pts["ls"][0], cm[1])
            chest_right = (pts["rs"][0], cm[1])
            _draw_dotted_line(frame, chest_left, chest_right, COL_CHEST, thickness=3, gap=8)
            _draw_landmark_dot(frame, *cm, radius=7, color=COL_CHEST)
            _draw_landmark_dot(frame, *chest_left, radius=5, color=COL_CHEST)
            _draw_landmark_dot(frame, *chest_right, radius=5, color=COL_CHEST)
            _put_label(frame, f"Chest: {measurements['chest_cm']} cm",
                       cm[0] - 50, cm[1] - 15, COL_CHEST)

        # --- Left arm (shoulder → elbow → wrist) ---
        if sleeve_mode == "short":
            # Short sleeve: only draw shoulder → elbow
            if "ls" in pts and "le" in pts:
                _draw_dotted_line(frame, pts["ls"], pts["le"], COL_ARM, thickness=3, gap=8)
                _draw_landmark_dot(frame, *pts["le"], radius=5, color=COL_ARM)
            if "rs" in pts and "re" in pts:
                _draw_dotted_line(frame, pts["rs"], pts["re"], COL_ARM, thickness=3, gap=8)
                _draw_landmark_dot(frame, *pts["re"], radius=5, color=COL_ARM)
        else:
            # Long sleeve: draw shoulder → elbow → wrist
            if "ls" in pts and "le" in pts and "lw" in pts:
                _draw_dotted_line(frame, pts["ls"], pts["le"], COL_ARM, thickness=3, gap=8)
                _draw_dotted_line(frame, pts["le"], pts["lw"], COL_ARM, thickness=3, gap=8)
                _draw_landmark_dot(frame, *pts["le"], radius=5, color=COL_ARM)
                _draw_landmark_dot(frame, *pts["lw"], radius=5, color=COL_ARM)

            # --- Right arm (shoulder → elbow → wrist) ---
            if "rs" in pts and "re" in pts and "rw" in pts:
                _draw_dotted_line(frame, pts["rs"], pts["re"], COL_ARM, thickness=3, gap=8)
                _draw_dotted_line(frame, pts["re"], pts["rw"], COL_ARM, thickness=3, gap=8)
                _draw_landmark_dot(frame, *pts["re"], radius=5, color=COL_ARM)
                _draw_landmark_dot(frame, *pts["rw"], radius=5, color=COL_ARM)

        # Arm label near best visible elbow/wrist
        arm_label_placed = False
        if sleeve_mode == "short":
            label_keys = ("le", "re")  # short sleeve: label near elbow
        else:
            label_keys = ("lw", "rw", "le", "re")  # long sleeve: prefer wrist
        for key in label_keys:
            if key in pts and not arm_label_placed:
                _put_label(frame, f"Arm: {measurements['arm_cm']} cm",
                           pts[key][0] - 40, pts[key][1] + 20, COL_ARM)
                arm_label_placed = True

        # --- Hip dots (if visible) ---
        if "lh" in pts:
            _draw_landmark_dot(frame, *pts["lh"], radius=5, color=(180, 180, 255))
        if "rh" in pts:
            _draw_landmark_dot(frame, *pts["rh"], radius=5, color=(180, 180, 255))

        # 3) Info panel (top-left) with semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (310, 185), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y0 = 30
        gap = 26
        cv2.putText(frame, f"Gender: {gender}", (16, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Shoulder: {measurements['shoulder_cm']} cm", (16, y0 + gap),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_SHOULDER, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Chest:    {measurements['chest_cm']} cm", (16, y0 + gap * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_CHEST, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Arm:      {measurements['arm_cm']} cm", (16, y0 + gap * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_ARM, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Predicted Size: {size}", (16, y0 + gap * 4 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, cv2.LINE_AA)

        return frame


class VirtualTryOn:
    """Overlay clothing images onto the user based on pose landmarks.
    Re-uses helper functions from the existing app.py logic."""

    SHIRT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shirts")

    # Scale presets per predicted size
    SIZE_PRESETS = {
        "XS": {"width_scale": 1.50, "height_scale": 1.30, "y_lift": 0.20},
        "S":  {"width_scale": 1.60, "height_scale": 1.40, "y_lift": 0.19},
        "M":  {"width_scale": 1.70, "height_scale": 1.50, "y_lift": 0.18},
        "L":  {"width_scale": 1.90, "height_scale": 1.70, "y_lift": 0.16},
        "XL": {"width_scale": 2.05, "height_scale": 1.85, "y_lift": 0.15},
    }

    def __init__(self):
        self.shirts = []
        self.shirt_names = []
        self.current_index = 0
        self._load_shirts()

    def _load_shirts(self):
        """Load all PNG shirts from the shirts/ directory."""
        if not os.path.isdir(self.SHIRT_DIR):
            print(f"[WARNING] VirtualTryOn: '{self.SHIRT_DIR}' not found")
            return

        paths = sorted(glob.glob(os.path.join(self.SHIRT_DIR, "*.png")))
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            # Ensure RGBA
            if img.ndim == 3 and img.shape[2] == 3:
                a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
                img = np.concatenate([img, a], axis=2)
            if img.ndim == 3 and img.shape[2] == 4:
                self.shirts.append(img)
                self.shirt_names.append(os.path.basename(p))

        print(f"[OK] VirtualTryOn: Loaded {len(self.shirts)} shirt(s)")

    def get_shirt_list(self):
        return self.shirt_names

    def set_shirt(self, index):
        if 0 <= index < len(self.shirts):
            self.current_index = index

    def next_shirt(self):
        if len(self.shirts) > 0:
            self.current_index = (self.current_index + 1) % len(self.shirts)

    def prev_shirt(self):
        if len(self.shirts) > 0:
            self.current_index = (self.current_index - 1) % len(self.shirts)

    # ----- overlay helpers (from app.py) -----

    @staticmethod
    def _rotate_rgba(img_rgba, angle_deg):
        h, w = img_rgba.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cos = abs(M[0, 0]); sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(
            img_rgba, M, (new_w, new_h),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    @staticmethod
    def _soften_alpha(rgba, sigma=1.2):
        out = rgba.copy()
        out[:, :, 3] = cv2.GaussianBlur(out[:, :, 3], (0, 0), sigma)
        return out

    @staticmethod
    def _overlay(frame, overlay, x, y):
        H, W = frame.shape[:2]
        oh, ow = overlay.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + ow), min(H, y + oh)
        if x1 >= x2 or y1 >= y2:
            return frame
        ox1, oy1 = x1 - x, y1 - y
        roi = frame[y1:y2, x1:x2]
        over = overlay[oy1:oy1 + (y2 - y1), ox1:ox1 + (x2 - x1)]
        alpha = over[:, :, 3].astype(np.float32) / 255.0
        alpha3 = np.dstack([alpha] * 3)
        blended = alpha3 * over[:, :, :3].astype(np.float32) + (1 - alpha3) * roi.astype(np.float32)
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        return frame

    def apply(self, frame, measurements, size="M"):
        """Overlay the current shirt onto the frame using measurements."""
        if len(self.shirts) == 0:
            return frame
        if not measurements.get("detected"):
            return frame

        lm_data = measurements.get("landmarks")
        if lm_data is None:
            return frame

        h, w = frame.shape[:2]
        lm = lm_data.landmark

        ls = lm[BodyMeasurements.LEFT_SHOULDER]
        rs = lm[BodyMeasurements.RIGHT_SHOULDER]
        lh = lm[BodyMeasurements.LEFT_HIP]

        lsx, lsy = int(ls.x * w), int(ls.y * h)
        rsx, rsy = int(rs.x * w), int(rs.y * h)
        lhy = int(lh.y * h)

        shoulder_px = int(math.hypot(lsx - rsx, lsy - rsy))
        torso_px = int(lhy - (lsy + rsy) // 2)

        if shoulder_px < 30 or torso_px < 30:
            return frame

        preset = self.SIZE_PRESETS.get(size, self.SIZE_PRESETS["M"])

        target_w = int(shoulder_px * preset["width_scale"])
        target_h = int(torso_px * preset["height_scale"])

        shirt_rgba = self.shirts[self.current_index]
        shirt_resized = cv2.resize(shirt_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Rotation from shoulder angle
        angle_rad = math.atan2(lsy - rsy, lsx - rsx)
        angle_deg = -math.degrees(angle_rad)
        shirt_rot = self._rotate_rgba(shirt_resized, angle_deg)
        shirt_rot = self._soften_alpha(shirt_rot)

        cx = (lsx + rsx) // 2
        cy = (lsy + rsy) // 2
        x = cx - shirt_rot.shape[1] // 2
        y = cy - int(shirt_rot.shape[0] * preset["y_lift"])

        frame = self._overlay(frame, shirt_rot, x, y)

        # Shirt label
        name = self.shirt_names[self.current_index]
        cv2.putText(frame, f"Shirt: {name}", (w - 260, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
