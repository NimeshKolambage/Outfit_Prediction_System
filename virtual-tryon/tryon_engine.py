"""
Virtual Try-On Engine
Real-time clothing overlay with pose detection and adaptive blending.
"""

import cv2
import numpy as np
import os
import math
import glob
import time
import pickle
from collections import Counter
import mediapipe_compat as mp

# -------------------------
# Configuration
# -------------------------
PERSON_TH = 0.5              # Segmentation threshold

# Lighting and shading
LIGHT_ADAPT_STRENGTH = 0.4   # Luminance adaptation strength
SHADOW_STRENGTH = 0.35       # Body shadow intensity
HIGHLIGHT_STRENGTH = 0.15    # Body highlight intensity

# Size prediction
RATIO_THRESHOLD = 0.72
STABLE_FRAMES = 25  # Increased stability buffer (was 10)
RATIO_CHANGE_THRESHOLD = 0.08  # Minimum ratio change to trigger size update (new)


# Scale presets per size (shirts: S/M/L/XL, pants: numeric waist)
SIZE_PRESETS = {
    # Shirt sizes
    "S":  {"width_scale": 1.50, "height_scale": 1.45, "y_lift": 0.30},
    "M":  {"width_scale": 1.65, "height_scale": 1.55, "y_lift": 0.28},
    "L":  {"width_scale": 1.80, "height_scale": 1.70, "y_lift": 0.26},
    "XL": {"width_scale": 1.95, "height_scale": 1.85, "y_lift": 0.24},
    # Pant sizes (numeric waist inches)
    "28": {"width_scale": 1.40, "height_scale": 1.40, "y_lift": 0.30},
    "30": {"width_scale": 1.50, "height_scale": 1.45, "y_lift": 0.30},
    "32": {"width_scale": 1.60, "height_scale": 1.52, "y_lift": 0.28},
    "34": {"width_scale": 1.70, "height_scale": 1.60, "y_lift": 0.27},
    "36": {"width_scale": 1.80, "height_scale": 1.70, "y_lift": 0.26},
    "38": {"width_scale": 1.95, "height_scale": 1.85, "y_lift": 0.24},
}


# -------------------------
# Image Processing Functions
# -------------------------

def adapt_lighting(shirt_rgba, frame_bgr, x, y, strength=0.4):
    """Adapt shirt luminance to match body lighting using LAB color space."""
    H, W = frame_bgr.shape[:2]
    oh, ow = shirt_rgba.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x1 >= x2 or y1 >= y2:
        return shirt_rgba

    roi = frame_bgr[y1:y2, x1:x2]
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    over = shirt_rgba[(y1 - y):(y2 - y), (x1 - x):(x2 - x)]
    alpha = over[:, :, 3].astype(np.float32) / 255.0
    mask = (alpha > 0.2).astype(np.uint8)

    if cv2.countNonZero(mask) < 100:
        return shirt_rgba

    # Get luminance statistics
    body_l = roi_lab[:, :, 0]
    body_mean = cv2.mean(body_l, mask=mask)[0]
    body_std = max(np.std(body_l[mask > 0]), 1)

    shirt_lab = cv2.cvtColor(over[:, :, :3], cv2.COLOR_BGR2LAB).astype(np.float32)
    shirt_l = shirt_lab[:, :, 0]
    shirt_mean = cv2.mean(shirt_l, mask=mask)[0]
    shirt_std = max(np.std(shirt_l[mask > 0]), 1)

    # Match luminance distribution
    adjusted_l = (shirt_l - shirt_mean) * (body_std / shirt_std) + body_mean
    adjusted_l = adjusted_l * strength + shirt_l * (1 - strength)
    shirt_lab[:, :, 0] = np.clip(adjusted_l, 0, 255)

    adjusted_bgr = cv2.cvtColor(shirt_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    out = shirt_rgba.copy()
    out[(y1 - y):(y2 - y), (x1 - x):(x2 - x), :3] = adjusted_bgr
    return out


def apply_body_shading(shirt_rgba, frame_bgr, x, y, shadow_strength=0.35, highlight_strength=0.15):
    """Apply shadows and highlights based on body luminance for depth effect."""
    H, W = frame_bgr.shape[:2]
    oh, ow = shirt_rgba.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x1 >= x2 or y1 >= y2:
        return shirt_rgba

    roi = frame_bgr[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Create and normalize shadow/highlight maps
    shadow_map = cv2.GaussianBlur(1.0 - roi_gray, (0, 0), 9)
    highlight_map = cv2.GaussianBlur(roi_gray, (0, 0), 9)
    shadow_map = (shadow_map - shadow_map.min()) / (shadow_map.max() - shadow_map.min() + 1e-6)
    highlight_map = (highlight_map - highlight_map.min()) / (highlight_map.max() - highlight_map.min() + 1e-6)

    out = shirt_rgba.copy()
    over = out[(y1 - y):(y2 - y), (x1 - x):(x2 - x)]
    alpha = over[:, :, 3].astype(np.float32) / 255.0
    mask = (alpha > 0.2).astype(np.float32)

    # Apply shading
    shadow_mult = np.clip(1.0 - (shadow_strength * shadow_map), 0.5, 1.0)
    highlight_add = highlight_strength * highlight_map * 30

    rgb = over[:, :, :3].astype(np.float32)
    rgb = rgb * (shadow_mult[..., None] * mask[..., None] + (1 - mask[..., None]))
    rgb = rgb + (highlight_add[..., None] * mask[..., None])

    over[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[(y1 - y):(y2 - y), (x1 - x):(x2 - x)] = over
    return out


def blend_overlay(frame_bgr, shirt_rgba, x, y):
    """Blend shirt onto frame with soft edges."""
    H, W = frame_bgr.shape[:2]
    oh, ow = shirt_rgba.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x1 >= x2 or y1 >= y2:
        return frame_bgr

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    roi = frame_bgr[y1:y2, x1:x2]
    over = shirt_rgba[oy1:oy2, ox1:ox2]
    over_rgb = over[:, :, :3]

    # Create soft-edged alpha
    alpha = over[:, :, 3].astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha_eroded = cv2.erode((alpha * 255).astype(np.uint8), kernel, iterations=1).astype(np.float32) / 255.0
    edge_mask = cv2.GaussianBlur(alpha - alpha_eroded, (5, 5), 0)
    final_alpha = np.clip(alpha_eroded + edge_mask * 0.7, 0, 1)

    # Alpha blend
    alpha3 = np.dstack([final_alpha] * 3)
    blended = alpha3 * over_rgb.astype(np.float32) + (1 - alpha3) * roi.astype(np.float32)
    frame_bgr[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame_bgr


def create_neck_cutout(shirt_rgba, neck_x, neck_y, neck_radius, shirt_x, shirt_y):
    """Create smooth cutout around neck area."""
    h, w = shirt_rgba.shape[:2]
    rel_x, rel_y = neck_x - shirt_x, neck_y - shirt_y

    if not (0 <= rel_x < w and 0 <= rel_y < h):
        return shirt_rgba

    out = shirt_rgba.copy()
    y_coords, x_coords = np.ogrid[:h, :w]

    # Ellipse mask (wider than tall)
    a, b = neck_radius * 1.2, neck_radius * 0.7
    dist = ((x_coords - rel_x) / a) ** 2 + ((y_coords - rel_y) / b) ** 2
    neck_mask = np.clip(1.0 - np.exp(-2 * (dist - 0.8)), 0, 1)

    out[:, :, 3] = (out[:, :, 3].astype(np.float32) * neck_mask).astype(np.uint8)
    return out


# -------------------------
# Helper Classes
# -------------------------

class SizeStabilizer:
    """Stabilize size predictions using voting over recent frames."""
    def __init__(self, initial_size="M", need_frames=25):
        self.stable_size = initial_size
        self.need_frames = need_frames
        self.size_history = []

    def update(self, current_size):
        # Only add to history if it differs from current stable size
        # This prevents rapid oscillation between sizes
        if current_size != self.stable_size:
            self.size_history.append(current_size)
        else:
            # Reinforce current size
            self.size_history.append(self.stable_size)
            
        if len(self.size_history) > self.need_frames:
            self.size_history.pop(0)

        # Required: 70% of recent frames must agree to change size
        if len(self.size_history) >= self.need_frames:
            most_common = Counter(self.size_history).most_common(1)
            if len(self.size_history) > 0:
                top_count = most_common[0][1] if most_common else 0
                confidence = top_count / len(self.size_history)
                # Only change if we have high confidence (>70%)
                if confidence > 0.7:
                    self.stable_size = most_common[0][0]
        
        return self.stable_size


class LandmarkSmoother:
    """Smooth landmark positions using exponential moving average."""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.points = {}

    def update(self, key, x, y):
        if key in self.points:
            px, py = self.points[key]
            self.points[key] = (self.alpha * x + (1 - self.alpha) * px,
                               self.alpha * y + (1 - self.alpha) * py)
        else:
            self.points[key] = (float(x), float(y))
        return int(self.points[key][0]), int(self.points[key][1])

    def reset(self):
        self.points = {}


# -------------------------
# Main Engine
# -------------------------

class TryOnEngine:
    """Virtual try-on engine with pose detection and clothing overlay."""

    SUPPORTED_FORMATS = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp']

    def set_gender_detector(self, gender_detector):
        """Set the gender detector instance for gender-gated try-on."""
        self.gender_detector = gender_detector

    def _validate_distance(self, shoulder_w_px):
        """
        Validate user is at EXACTLY 6 feet (5.8-6.2 ft tolerance).
        Returns: (is_valid: bool, msg: str, color: str)
        Color: "red" or "green" for display
        """
        if shoulder_w_px is None or shoulder_w_px <= 0:
            return False, "Move into frame (shoulders not detected)", "red"
        
        # Calibration: at 6 feet, shoulder width ≈ 145-155 px on typical webcam
        # Approximate conversion: distance_ft ≈ (shoulder_width_inches * focal_length) / shoulder_px
        # For standard setup: ~6ft → ~150px
       # Calibration: at 6 feet, shoulder width ≈ 145-155 px on typical webcam
        # For standard setup: ~6ft → ~150px
        target_px = 150          # ✅ was 100 (wrong for most webcams)
        px_tolerance = 20        # ✅ was 15 (give a bit more room)
        
        min_px = target_px - px_tolerance
        max_px = target_px + px_tolerance
        
        if shoulder_w_px < min_px:
            # Too far
            return False, "TOO FAR - Move closer to 6 feet", "red"
        if shoulder_w_px > max_px:
            # Too close
            return False, "TOO CLOSE - Move back to 6 feet", "red"
        
        # Perfect distance
        return True, "Distance OK (6 feet) - Size Locked ✓", "green"

    def __init__(self, shirt_dir="shirts"):
        self.shirt_dir = shirt_dir
        self.shirt_paths = []
        self.shirt_cache = []
        self.shirt_index = 0

        self.pose = mp.Pose(static_image_mode=False, model_complexity=1)
        self.seg = mp.SelfieSegmentation(model_selection=1)
        self.shirt_stabilizer = SizeStabilizer(initial_size="M", need_frames=STABLE_FRAMES)
        self.pant_stabilizer  = SizeStabilizer(initial_size="M", need_frames=STABLE_FRAMES)
        self.smoother = LandmarkSmoother(alpha=0.5)
        
        # Gender detection phase
        self.gender_detector = None  # Set via set_gender_detector()
        self.gender_countdown_active = True  # Start with gender countdown
        self.gender_countdown_start = time.time()
        self.gender_countdown_duration = 3  # 3 seconds for gender detection
        self.detected_gender = None  # "Male" or "Female"
        self.gender_confirmed = False  # True after gender is determined
        self.female_blocked = False  # True if female detected
        
        # Countdown + One-time detection system
        self.countdown_active = False  # Starts after gender confirmed as Male
        self.countdown_start_time = None  # When countdown began
        self.countdown_duration = 5  # 5 seconds to prepare
        self.detection_ready = False  # Ready to detect after countdown
        self.locked_size = None  # Will be set on one-time detection
        self.size_locked = False  # Flag to track if size is locked
        self.display_state = "GENDER_COUNTDOWN"  # GENDER_COUNTDOWN, GENDER_DETECT, FEMALE_BLOCKED, COUNTDOWN, DETECTING, READY
        self.distance_warning = ""  # Distance validation message
        self.distance_warning_color = "white"  # Color for warning text
        
        # Distance validation states
        self.is_distance_valid = False  # True only when at exactly 6 feet
        self.is_size_locked = False  # True after size is detected once
        
        # Locked sizes (both shirt and pant)
        self.locked_shirt_size = None
        self.locked_pant_size = None
        self.lock_start_time = None  # When sizes were locked
        self.auto_reset_timeout = 30 * 60  # 30 minutes in seconds

        # Load ML size prediction models
        self.shirt_size_model = None
        self.pant_size_model = None
        self._load_size_models()

        self._load_clothing()

    def _load_clothing(self):
        """Load all clothing images from directory."""
        if not os.path.isdir(self.shirt_dir):
            raise FileNotFoundError(f"[ERROR] '{self.shirt_dir}' folder not found.")

        self.shirt_paths = []
        for ext in self.SUPPORTED_FORMATS:
            self.shirt_paths.extend(glob.glob(os.path.join(self.shirt_dir, ext)))
        self.shirt_paths = sorted(self.shirt_paths)

        if not self.shirt_paths:
            raise FileNotFoundError("[ERROR] No image files in shirts folder.")

        print("[INFO] Loading clothing images...")
        valid_paths, self.shirt_cache = [], []

        for path in self.shirt_paths:
            try:
                img = self._load_image(path)
                valid_paths.append(path)
                self.shirt_cache.append(img)
                print(f"  Loaded: {os.path.basename(path)}")
            except Exception as e:
                print(f"  [ERROR] {os.path.basename(path)}: {e}")

        self.shirt_paths = valid_paths
        if not self.shirt_cache:
            raise RuntimeError("[ERROR] No valid clothing images loaded")
        print(f"[OK] Loaded {len(self.shirt_cache)} item(s)")

    def _load_image(self, path):
        """Load and validate clothing image, ensure RGBA format."""
        if os.path.getsize(path) > 10 * 1024 * 1024:
            raise ValueError("File too large (max 10MB)")

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to load image")

        h, w = img.shape[:2]
        if not (50 <= w <= 4000 and 50 <= h <= 4000):
            raise ValueError(f"Invalid dimensions ({w}x{h})")

        # Convert to RGBA if needed
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
            img = np.dstack([img, alpha])
        elif img.ndim != 3 or img.shape[2] != 4:
            raise ValueError("Unsupported format")

        return img
    

    def next_shirt(self):
        self.shirt_index = (self.shirt_index + 1) % len(self.shirt_cache)
        self.smoother.reset()
        self.locked_size = None  # Reset locked size when changing shirt
        self.locked_shirt_size = None
        self.locked_pant_size = None
        self.lock_start_time = None
        self.size_locked = False
        self.is_size_locked = False  # Reset new flag
        self.is_distance_valid = False

    def prev_shirt(self):
        self.shirt_index = (self.shirt_index - 1) % len(self.shirt_cache)
        self.smoother.reset()
        self.locked_size = None  # Reset locked size when changing shirt
        self.locked_shirt_size = None
        self.locked_pant_size = None
        self.lock_start_time = None
        self.size_locked = False
        self.is_size_locked = False  # Reset new flag
        self.is_distance_valid = False
    
    def _update_gender_countdown(self, frame):
        """Update gender detection countdown and detect gender after 3 seconds."""
        if not self.gender_countdown_active:
            return
        
        elapsed = time.time() - self.gender_countdown_start
        remaining = self.gender_countdown_duration - elapsed
        
        if remaining <= 0:
            # Timer expired - detect gender now
            self.gender_countdown_active = False
            self.display_state = "GENDER_DETECT"
            
            detected = "Male"  # Default fallback
            if self.gender_detector is not None and self.gender_detector.model_loaded:
                results = self.gender_detector.detect_gender(frame)
                if results:
                    detected = results[0]['gender']
                    print(f"[OK] Gender detected: {detected} (confidence: {results[0]['confidence']:.2f})")
                else:
                    print("[WARNING] No face detected during gender check, defaulting to Male")
            else:
                print("[WARNING] Gender model not loaded, defaulting to Male")
            
            self.detected_gender = detected
            self.gender_confirmed = True
            
            if detected == "Female":
                self.female_blocked = True
                self.display_state = "FEMALE_BLOCKED"
                print("[INFO] Female detected - try-on unavailable")
            else:
                # Male confirmed - start the distance/size countdown
                self.countdown_active = True
                self.countdown_start_time = time.time()
                self.display_state = "COUNTDOWN"
                print("[INFO] Male confirmed - proceeding to size detection")
        else:
            self.display_state = "GENDER_COUNTDOWN"
    
    def _get_gender_countdown_remaining(self):
        """Get remaining seconds on gender countdown."""
        if not self.gender_countdown_active:
            return 0
        elapsed = time.time() - self.gender_countdown_start
        remaining = self.gender_countdown_duration - elapsed
        return max(0, int(remaining) + 1)

    def _update_countdown(self):
        """Update countdown and transition to detection phase."""
        if self.countdown_active:
            elapsed = time.time() - self.countdown_start_time
            remaining = self.countdown_duration - elapsed
            
            if remaining <= 0:
                # Countdown finished - transition to detection
                self.countdown_active = False
                self.detection_ready = True
                self.display_state = "DETECTING"
                return 0
            else:
                self.display_state = "COUNTDOWN"
                return int(remaining) + 1  # Round up
        return 0
    
    def reset_for_next_user(self):
        """Reset all detection states for next user - press R key."""
        # Reset gender detection phase
        self.gender_countdown_active = True
        self.gender_countdown_start = time.time()
        self.detected_gender = None
        self.gender_confirmed = False
        self.female_blocked = False
        
        # Reset size detection phase
        self.countdown_active = False
        self.countdown_start_time = None
        self.detection_ready = False
        self.locked_size = None
        self.locked_shirt_size = None
        self.locked_pant_size = None
        self.lock_start_time = None
        self.size_locked = False
        self.is_size_locked = False  # Reset size locked flag
        self.is_distance_valid = False  # Reset distance flag
        self.display_state = "GENDER_COUNTDOWN"
        self.smoother.reset()
        print("[INFO] Reset for next user. Gender detection starting (3s)...")

    def reload_clothing(self):
        """Reload clothing from folder."""
        old_name = self.get_current_shirt_name()
        self._load_clothing()
        for i, path in enumerate(self.shirt_paths):
            if os.path.basename(path) == old_name:
                self.shirt_index = i
                return len(self.shirt_cache)
        self.shirt_index = 0
        return len(self.shirt_cache)

    def get_clothing_list(self):
        return [os.path.basename(p) for p in self.shirt_paths]

    def get_current_shirt_name(self):
        return os.path.basename(self.shirt_paths[self.shirt_index]) if self.shirt_paths else ""

    def _build_person_mask(self, seg_mask):
        """Build smooth person mask from segmentation."""
        person = (seg_mask > PERSON_TH).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, kernel, iterations=2)
        person = cv2.morphologyEx(person, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.GaussianBlur(person.astype(np.float32) / 255.0, (0, 0), 5)

    def _load_size_models(self):
        """Load trained ML models for shirt and pant size prediction."""
        model_dir = os.path.dirname(os.path.abspath(__file__))
        
        shirt_path = os.path.join(model_dir, "shirt_size_model.pkl")
        pant_path = os.path.join(model_dir, "pant_size_model.pkl")
        
        if os.path.exists(shirt_path):
            try:
                with open(shirt_path, 'rb') as f:
                    self.shirt_size_model = pickle.load(f)
                print(f"[OK] Shirt size model loaded (v{self.shirt_size_model.get('version', '?')})")
            except Exception as e:
                print(f"[WARN] Failed to load shirt model: {e}")
                self.shirt_size_model = None
        else:
            print("[WARN] shirt_size_model.pkl not found - using rule-based fallback")
        
        if os.path.exists(pant_path):
            try:
                with open(pant_path, 'rb') as f:
                    self.pant_size_model = pickle.load(f)
                print(f"[OK] Pant size model loaded (v{self.pant_size_model.get('version', '?')})")
            except Exception as e:
                print(f"[WARN] Failed to load pant model: {e}")
                self.pant_size_model = None
        else:
            print("[WARN] pant_size_model.pkl not found - using rule-based fallback")

    def _predict_size(self, shoulder_w, torso_h, hip_w, leg_h, is_pants=False):
        """
        Predict clothing size using ML model with MediaPipe measurements.
        
        Uses runtime-compatible ratio features (scale-invariant) that match
        the training features exactly. Falls back to rule-based thresholds
        if ML models are not loaded.
        
        Args:
            shoulder_w: shoulder width in pixels (left-right shoulder distance)
            torso_h:    torso height in pixels (shoulder to hip distance)
            hip_w:      hip width in pixels (left-right hip distance)
            leg_h:      leg height in pixels (hip to knee distance)
            is_pants:   True for pant sizing, False for shirt sizing
        
        Returns:
            Size string: "S", "M", "L", or "XL"
        """
        # Try ML prediction first
        try:
            if is_pants and self.pant_size_model is not None:
                # Pant features: same order as training
                features = np.array([[
                    hip_w / (leg_h + 1e-6),           # hip_leg_ratio
                    hip_w / (shoulder_w + 1e-6),      # hip_shoulder_ratio
                    leg_h / (torso_h + 1e-6),         # leg_torso_ratio
                    hip_w / (torso_h + 1e-6),         # hip_torso_ratio
                    shoulder_w / (leg_h + 1e-6),      # shoulder_leg_ratio
                    leg_h / (torso_h + leg_h + 1e-6)  # body_lower_ratio
                ]])
                pred_idx = self.pant_size_model['model'].predict(features)[0]
                pred_size = self.pant_size_model['label_encoder'].inverse_transform([pred_idx])[0]
                print(f"[ML] Pant prediction: {pred_size} (features: hip_leg={features[0][0]:.3f})")
                return pred_size
                
            elif not is_pants and self.shirt_size_model is not None:
                # Shirt features: same order as training
                features = np.array([[
                    shoulder_w / (torso_h + 1e-6),       # shoulder_torso_ratio
                    shoulder_w / (hip_w + 1e-6),         # shoulder_hip_ratio
                    torso_h / (leg_h + 1e-6),            # torso_leg_ratio
                    shoulder_w / (leg_h + 1e-6),         # shoulder_leg_ratio
                    hip_w / (torso_h + 1e-6),            # hip_torso_ratio
                    torso_h / (torso_h + leg_h + 1e-6)   # body_upper_ratio
                ]])
                pred_idx = self.shirt_size_model['model'].predict(features)[0]
                pred_size = self.shirt_size_model['label_encoder'].inverse_transform([pred_idx])[0]
                print(f"[ML] Shirt prediction: {pred_size} (features: sh_torso={features[0][0]:.3f})")
                return pred_size
                
        except Exception as e:
            print(f"[WARN] ML prediction failed, using fallback: {e}")
        
        # ─── Rule-based fallback ───
        return self._predict_size_fallback(shoulder_w, torso_h, hip_w, leg_h, is_pants)

    def _predict_size_fallback(self, shoulder_w, torso_h, hip_w, leg_h, is_pants=False):
        """Rule-based size prediction fallback (original hardcoded thresholds)."""
        if is_pants:
            ratio = hip_w / (leg_h + 1e-6)
            ratio = float(np.clip(ratio, 0.15, 0.50))
            if ratio < 0.18:
                return "28"
            elif ratio < 0.22:
                return "30"
            elif ratio < 0.27:
                return "32"
            elif ratio < 0.32:
                return "34"
            elif ratio < 0.38:
                return "36"
            return "38"
        else:
            ratio = shoulder_w / (torso_h + 1e-6)
            ratio = float(np.clip(ratio, 0.55, 0.85))
            if ratio < 0.60:
                return "S"
            elif ratio < 0.67:
                return "M"
            elif ratio < 0.75:
                return "L"
            return "XL"
        

    def process_frame(self, frame_bgr):
        """Process frame and overlay clothing."""
        # --- Gender detection phase (runs BEFORE size detection) ---
        self._update_gender_countdown(frame_bgr)
        gender_countdown = self._get_gender_countdown_remaining()
        
        # If female is blocked, return early with the message
        if self.female_blocked:
            frame = frame_bgr.copy()
            h, w = frame.shape[:2]
            # Dark overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            
            # Formal message
            msg1 = "We apologize for the inconvenience."
            msg2 = "Our virtual try-on is currently available"
            msg3 = "for men's clothing only."
            msg4 = "Women's collection coming soon!"
            msg5 = "Press [R] to reset for another user."
            
            cv2.putText(frame, msg1, (w//2 - 280, h//2 - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100, 180, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, msg2, (w//2 - 280, h//2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, msg3, (w//2 - 280, h//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, msg4, (w//2 - 280, h//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, msg5, (w//2 - 220, h//2 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1, cv2.LINE_AA)
            
            return frame, {
                "user_size_shirt": "--",
                "user_size_pant": "--",
                "item_type": "Shirt",
                "filename": self.get_current_shirt_name(),
                "countdown": 0,
                "gender_countdown": 0,
                "detected_gender": self.detected_gender,
                "state": "FEMALE_BLOCKED",
                "size_locked": False,
                "distance_valid": False,
                "distance_warning": "",
                "distance_warning_color": "white",
            }
        
        # If still in gender countdown, show countdown overlay but still render camera
        if self.gender_countdown_active:
            frame = frame_bgr.copy()
            h, w = frame.shape[:2]
            remaining = self._get_gender_countdown_remaining()
            cv2.putText(frame, "GENDER DETECTION", (w//2 - 200, h//2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "Please face the camera...", (w//2 - 200, h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str(remaining), (w//2 - 20, h//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4, cv2.LINE_AA)
            
            # Draw gender detection results live if available
            if self.gender_detector is not None and self.gender_detector.model_loaded:
                results = self.gender_detector.detect_gender(frame_bgr)
                frame = self.gender_detector.draw_results(frame, results)
            
            return frame, {
                "user_size_shirt": "--",
                "user_size_pant": "--",
                "item_type": "Shirt",
                "filename": self.get_current_shirt_name(),
                "countdown": 0,
                "gender_countdown": remaining,
                "detected_gender": None,
                "state": "GENDER_COUNTDOWN",
                "size_locked": False,
                "distance_valid": False,
                "distance_warning": "",
                "distance_warning_color": "white",
            }
        
        # --- Normal size detection flow (only reached if Male confirmed) ---
        # Update countdown state
        countdown = self._update_countdown()
        
        frame = frame_bgr.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        seg_res = self.seg.process(rgb)
        person_mask = self._build_person_mask(seg_res.segmentation_mask)
        pose_res = self.pose.process(rgb)

        # Frame state initialization
        self.distance_warning = ""
        self.distance_warning_color = "white"
        self.is_distance_valid = False
        
        # Display sizes: show locked sizes if available, otherwise "--"
        user_size_shirt = self.locked_shirt_size if self.is_size_locked else "--"
        user_size_pant = self.locked_pant_size if self.is_size_locked else "--"
        
        shirt_rgba = self.shirt_cache[self.shirt_index].copy()
        filename = self.get_current_shirt_name().lower()
        is_pants = "pant" in filename or "short" in filename

        # ===== AUTO-RESET CHECK (30 minutes after locking) =====
        if self.is_size_locked and self.lock_start_time is not None:
            elapsed_time = time.time() - self.lock_start_time
            if elapsed_time > self.auto_reset_timeout:
                print("[WARN] 30 minutes passed. Auto-resetting for next user...")
                self.reset_for_next_user()
                countdown = self._update_countdown()

        # Process pose landmarks
        if pose_res.pose_landmarks:
            h, w = frame.shape[:2]
            lm = pose_res.pose_landmarks.landmark

            # Get and smooth key landmarks
            ls, rs = lm[mp.PoseLandmark.LEFT_SHOULDER], lm[mp.PoseLandmark.RIGHT_SHOULDER]
            lh, rh = lm[mp.PoseLandmark.LEFT_HIP], lm[mp.PoseLandmark.RIGHT_HIP]

            lsx, lsy = self.smoother.update("ls", int(ls.x * w), int(ls.y * h))
            rsx, rsy = self.smoother.update("rs", int(rs.x * w), int(rs.y * h))
            lhx, lhy = self.smoother.update("lh", int(lh.x * w), int(lh.y * h))
            rhx, rhy = self.smoother.update("rh", int(rh.x * w), int(rh.y * h))

            shoulder_w = int(math.hypot(lsx - rsx, lsy - rsy))
            
            # ===== DISTANCE VALIDATION (ALWAYS CHECK) =====
            is_dist_valid, distance_msg, msg_color = self._validate_distance(shoulder_w)
            self.is_distance_valid = is_dist_valid
            
            # Show warning ONLY if size is not locked yet
            if not self.is_size_locked:
                self.distance_warning = distance_msg
                self.distance_warning_color = msg_color
            else:
                # After locking, no warnings
                self.distance_warning = ""
                self.distance_warning_color = "white"
            
            print(f"[DIST] shoulder={shoulder_w}px | valid={is_dist_valid} | size_locked={self.is_size_locked}", flush=True)

            # ===== DETECT BOTH SHIRT AND PANT SIZE AT 6 FEET =====
            # Only detect if:
            # 1. Countdown is over (detection_ready = True)
            # 2. Distance is exactly 6 feet (is_distance_valid = True)
            # 3. Size hasn't been locked yet (is_size_locked = False)
            
            if self.detection_ready and is_dist_valid and not self.is_size_locked:
                # Get measurements for both shirt and pants
                lk, rk = lm[mp.PoseLandmark.LEFT_KNEE], lm[mp.PoseLandmark.RIGHT_KNEE]
                lky, rky = int(lk.y * h), int(rk.y * h)
                
                hip_w = int(math.hypot(lhx - rhx, lhy - rhy))
                leg_h = int(((lky + rky) // 2) - ((lhy + rhy) // 2))
                torso_h = int(((lhy + rhy) // 2) - ((lsy + rsy) // 2))
                
                # Detect SHIRT size
                if shoulder_w > 10 and torso_h > 10:
                    shirt_predicted = self._predict_size(shoulder_w, torso_h, hip_w, leg_h, is_pants=False)
                    self.locked_shirt_size = shirt_predicted
                    ratio_shirt = shoulder_w / torso_h
                    print(f"[OK] SHIRT SIZE DETECTED: {shirt_predicted} (shoulder_w={shoulder_w}, torso_h={torso_h}, ratio={ratio_shirt:.3f})", flush=True)
                
                # Detect PANT size
                if hip_w > 10 and leg_h > 10:
                    pant_predicted = self._predict_size(shoulder_w, torso_h, hip_w, leg_h, is_pants=True)
                    self.locked_pant_size = pant_predicted
                    ratio_pant = hip_w / leg_h
                    print(f"[OK] PANT SIZE DETECTED: {pant_predicted} (hip_w={hip_w}, leg_h={leg_h}, ratio={ratio_pant:.3f})", flush=True)
                
                # Lock both sizes together
                if self.locked_shirt_size is not None and self.locked_pant_size is not None:
                    self.is_size_locked = True
                    self.size_locked = True
                    self.lock_start_time = time.time()
                    self.display_state = "READY"
                    user_size_shirt = self.locked_shirt_size
                    user_size_pant = self.locked_pant_size
                    print(f"[OK] BOTH SIZES LOCKED! Shirt: {self.locked_shirt_size}, Pant: {self.locked_pant_size}", flush=True)
            
            # ===== OVERLAY CLOTHING (ONLY IF SIZE IS LOCKED AND DISTANCE VALID) =====
            # Render clothing only when size is locked AND user is at exactly 6 feet
            if self.is_size_locked and self.is_distance_valid:
                if is_pants:
                    lk, rk = lm[mp.PoseLandmark.LEFT_KNEE], lm[mp.PoseLandmark.RIGHT_KNEE]
                    lky, rky = int(lk.y * h), int(rk.y * h)
                    hip_w = int(math.hypot(lhx - rhx, lhy - rhy))
                    leg_h = int(((lky + rky) // 2) - ((lhy + rhy) // 2))

                    if hip_w > 10 and leg_h > 10:
                        preset = SIZE_PRESETS.get(self.locked_pant_size, SIZE_PRESETS["32"])
                        target_w = int(hip_w * preset["width_scale"])
                        target_h = int(leg_h * preset["height_scale"])
                        resized = cv2.resize(shirt_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

                        cx, cy = (lhx + rhx) // 2, (lhy + rhy) // 2
                        x = cx - target_w // 2
                        y = cy - int(target_h * preset["y_lift"])

                        resized = adapt_lighting(resized, frame, x, y, LIGHT_ADAPT_STRENGTH)
                        resized = apply_body_shading(resized, frame, x, y, SHADOW_STRENGTH, HIGHLIGHT_STRENGTH)
                        frame = blend_overlay(frame, resized, x, y)
                else:
                    torso_h = int(((lhy + rhy) // 2) - ((lsy + rsy) // 2))

                    if shoulder_w > 10 and torso_h > 10:
                        preset = SIZE_PRESETS.get(self.locked_shirt_size, SIZE_PRESETS["M"])
                        target_w = int(shoulder_w * preset["width_scale"])
                        target_h = int(torso_h * preset["height_scale"])
                        resized = cv2.resize(shirt_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

                        cx, cy = (lsx + rsx) // 2, (lsy + rsy) // 2
                        x = cx - target_w // 2
                        y = cy - int(target_h * preset["y_lift"])

                        # Neck cutout
                        neck_radius = shoulder_w * 0.15
                        neck_y = cy - int(torso_h * 0.1)
                        resized = create_neck_cutout(resized, cx, neck_y, neck_radius, x, y)

                        resized = adapt_lighting(resized, frame, x, y, LIGHT_ADAPT_STRENGTH)
                        resized = apply_body_shading(resized, frame, x, y, SHADOW_STRENGTH, HIGHLIGHT_STRENGTH)
                        frame = blend_overlay(frame, resized, x, y)

        return frame, {
            "user_size_shirt": user_size_shirt,
            "user_size_pant": user_size_pant,
            "item_type": "Pant/Short" if is_pants else "Shirt",
            "filename": self.get_current_shirt_name(),
            "countdown": countdown,
            "gender_countdown": 0,
            "detected_gender": self.detected_gender,
            "state": self.display_state,
            "size_locked": self.is_size_locked,
            "distance_valid": self.is_distance_valid,
            "distance_warning": self.distance_warning,
            "distance_warning_color": self.distance_warning_color,
        }
