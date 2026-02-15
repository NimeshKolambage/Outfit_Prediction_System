import cv2
import numpy as np
import os, math, glob, sys
import mediapipe_compat as mp

# -------------------------
# Config
# -------------------------
# Camera index can be overridden with command-line argument: python app.py <camera_index>
CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SHIRT_DIR = "shirts"     # shirts folder
PERSON_TH = 0.55         # segmentation threshold

# Realism tuning
EDGE_SOFTEN_SIGMA = 1.2  # alpha edge blur (0.8-2.0)
LIGHT_MIN = 0.70         # lighting factor clamp
LIGHT_MAX = 1.20
SHADOW_STRENGTH = 0.22   # 0.0..0.4

# Size prediction tuning (M/L)
RATIO_THRESHOLD = 0.72   # tune: 0.68 - 0.78
STABLE_FRAMES = 8        # frames to confirm size change (avoid flicker)

# Scale presets per size
SIZE_PRESETS = {
    "M": {"width_scale": 1.70, "height_scale": 1.50, "y_lift": 0.18},
    "L": {"width_scale": 1.90, "height_scale": 1.70, "y_lift": 0.16},
}

# -------------------------
# Load shirts
# -------------------------
if not os.path.isdir(SHIRT_DIR):
    raise FileNotFoundError(f"[ERROR] '{SHIRT_DIR}' folder eka nathi. app.py ekata same folder eke shirts/ folder ekak hadanna.")

shirt_paths = sorted(glob.glob(os.path.join(SHIRT_DIR, "*.png")))
if len(shirt_paths) == 0:
    raise FileNotFoundError("[ERROR] shirts folder eke .png files nathi. shirts/shirt1.png wage transparent PNG danna.")

def load_shirt(idx: int):
    path = shirt_paths[idx]
    
    # Validate file size (max 10MB)
    file_size = os.path.getsize(path)
    max_size_mb = 10
    if file_size > max_size_mb * 1024 * 1024:
        actual_size_mb = file_size / (1024 * 1024)
        raise ValueError(f"[ERROR] Shirt file too large (max {max_size_mb}MB, got {actual_size_mb:.1f}MB): {path}")
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"[ERROR] Load wenne na: {path}")

    # Validate dimensions (reasonable size limits)
    h, w = img.shape[:2]
    if w < 50 or h < 50:
        raise ValueError(f"[ERROR] Image too small ({w}x{h}): {path}")
    if w > 4000 or h > 4000:
        raise ValueError(f"[ERROR] Image too large ({w}x{h}): {path}")

    # ensure RGBA
    if img.ndim == 3 and img.shape[2] == 3:
        print(f"[WARNING] {os.path.basename(path)} alpha channel nathi. RGBA convert karanawa.")
        a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, a], axis=2)
    elif img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"[ERROR] Unsupported image format: {path}")

    return img

shirt_index = 0

# Preload all shirt images at startup for better performance
print("[INFO] Preloading shirt images...")
valid_shirts = []
shirt_cache = []
for i, path in enumerate(shirt_paths):
    try:
        img = load_shirt(i)
        valid_shirts.append(path)
        shirt_cache.append(img)
        print(f"  [{len(valid_shirts)}/{len(shirt_paths)}] Loaded: {os.path.basename(path)}")
    except Exception as e:
        print(f"  [ERROR] Failed to load {os.path.basename(path)}: {e}")

# Update shirt_paths to only include valid shirts
shirt_paths = valid_shirts

if len(shirt_cache) == 0:
    raise RuntimeError("[ERROR] No valid shirt images could be loaded")

print(f"[OK] Loaded {len(shirt_cache)} shirt(s)")
shirt_rgba = shirt_cache[shirt_index]

# -------------------------
# MediaPipe
# -------------------------
mp_pose = mp.PoseLandmark
pose = mp.Pose(static_image_mode=False, model_complexity=1)

seg = mp.SelfieSegmentation(model_selection=1)

# -------------------------
# Helpers
# -------------------------
def rotate_rgba(img_rgba, angle_deg):
    h, w = img_rgba.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        img_rgba, M, (new_w, new_h),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

def soften_alpha_edges(rgba, sigma=1.2):
    out = rgba.copy()
    a = out[:, :, 3]
    # Use (0, 0) so OpenCV derives the Gaussian kernel size from sigma
    a_blur = cv2.GaussianBlur(a, (0, 0), sigma)
    out[:, :, 3] = a_blur
    return out

def match_lighting(shirt_rgba, frame_bgr, x, y, light_min=0.70, light_max=1.20):
    H, W = frame_bgr.shape[:2]
    oh, ow = shirt_rgba.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x1 >= x2 or y1 >= y2:
        return shirt_rgba

    roi = frame_bgr[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    over = shirt_rgba[(y1 - y):(y2 - y), (x1 - x):(x2 - x)]
    alpha = over[:, :, 3].astype(np.float32) / 255.0
    mask = (alpha > 0.2).astype(np.uint8) * 255

    if cv2.countNonZero(mask) < 50:
        return shirt_rgba

    body_mean = cv2.mean(roi_gray, mask=mask)[0]
    shirt_gray = cv2.cvtColor(over[:, :, :3], cv2.COLOR_BGR2GRAY)
    shirt_mean = cv2.mean(shirt_gray, mask=mask)[0] + 1e-6

    factor = body_mean / shirt_mean
    factor = float(np.clip(factor, light_min, light_max))

    adjusted = shirt_rgba.copy()
    rgb = adjusted[:, :, :3].astype(np.float32)
    rgb = np.clip(rgb * factor, 0, 255).astype(np.uint8)
    adjusted[:, :, :3] = rgb
    return adjusted

def apply_shadow_from_body(shirt_rgba, frame_bgr, x, y, strength=0.22):
    if strength <= 0:
        return shirt_rgba

    H, W = frame_bgr.shape[:2]
    oh, ow = shirt_rgba.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x1 >= x2 or y1 >= y2:
        return shirt_rgba

    roi = frame_bgr[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    shadow = 1.0 - roi_gray
    shadow = cv2.GaussianBlur(shadow, (0, 0), 7)

    out = shirt_rgba.copy()
    over = out[(y1 - y):(y2 - y), (x1 - x):(x2 - x)]
    alpha = over[:, :, 3].astype(np.float32) / 255.0
    mask = (alpha > 0.2).astype(np.float32)

    mult = 1.0 - (strength * shadow)
    mult = np.clip(mult, 0.6, 1.0)

    rgb = over[:, :, :3].astype(np.float32)
    rgb = rgb * (mult[..., None] * mask[..., None] + (1 - mask[..., None]))
    over[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)

    out[(y1 - y):(y2 - y), (x1 - x):(x2 - x)] = over
    return out

def overlay_rgba(frame_bgr, overlay_rgba, x, y, front_mask=None):
    H, W = frame_bgr.shape[:2]
    oh, ow = overlay_rgba.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x1 >= x2 or y1 >= y2:
        return frame_bgr

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    roi = frame_bgr[y1:y2, x1:x2]
    over = overlay_rgba[oy1:oy2, ox1:ox2]

    over_rgb = over[:, :, :3].astype(np.float32)
    alpha = (over[:, :, 3].astype(np.float32) / 255.0)

    if front_mask is not None:
        fm = front_mask[y1:y2, x1:x2].astype(bool)
        alpha = np.where(fm, 0.0, alpha)

    roi_f = roi.astype(np.float32)
    alpha3 = np.dstack([alpha, alpha, alpha])
    blended = alpha3 * over_rgb + (1 - alpha3) * roi_f
    frame_bgr[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame_bgr

def build_front_mask(seg_mask):
    person = (seg_mask > PERSON_TH).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    dil = cv2.dilate(person, k, iterations=1)
    ero = cv2.erode(person, k, iterations=1)
    edge = cv2.absdiff(dil, ero)

    front = (edge > 0).astype(np.uint8)
    front = cv2.dilate(front, k, iterations=1)
    return front

# Size stabilization class to encapsulate state
class SizeStabilizer:
    """Encapsulates state for stabilizing size predictions over multiple frames"""
    def __init__(self, initial_size="M", need_frames=8):
        self.stable_size = initial_size
        self.stable_count = 0
        self.need_frames = need_frames
    
    def update(self, current_size):
        """Update with new size detection and return stabilized size"""
        if current_size != self.stable_size:
            self.stable_count += 1
            if self.stable_count >= self.need_frames:
                self.stable_size = current_size
                self.stable_count = 0
        else:
            self.stable_count = 0
        return self.stable_size

# -------------------------
# Camera
# -------------------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("[ERROR] Camera open wenne na. permission/other app camera use karanawada balanna.")

# Initialize size stabilizer
size_stabilizer = SizeStabilizer(initial_size="M", need_frames=STABLE_FRAMES)

print("[OK] Running... Controls: [A] prev shirt | [D] next shirt | [Q]/[ESC] exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    seg_res = seg.process(rgb)
    seg_mask = seg_res.segmentation_mask
    front_mask = build_front_mask(seg_mask)

    pose_res = pose.process(rgb)

    user_size = size_stabilizer.stable_size  # default
    
    # Detect if current item is pants/shorts or shirt by filename
    current_filename = os.path.basename(shirt_paths[shirt_index]).lower()
    is_pant_or_short = 'pant' in current_filename or 'short' in current_filename

    if pose_res.pose_landmarks:
        h, w = frame.shape[:2]
        lm = pose_res.pose_landmarks.landmark

        # Get all landmarks
        ls = lm[mp_pose.LEFT_SHOULDER]
        rs = lm[mp_pose.RIGHT_SHOULDER]
        lh = lm[mp_pose.LEFT_HIP]
        rh = lm[mp_pose.RIGHT_HIP]
        lk = lm[mp_pose.LEFT_KNEE]
        rk = lm[mp_pose.RIGHT_KNEE]

        lsx, lsy = int(ls.x * w), int(ls.y * h)
        rsx, rsy = int(rs.x * w), int(rs.y * h)
        lhx, lhy = int(lh.x * w), int(lh.y * h)
        rhx, rhy = int(rh.x * w), int(rh.y * h)
        lky = int(lk.y * h)
        rky = int(rk.y * h)

        if is_pant_or_short:
            # ===== PANTS/SHORTS: Align to hips =====
            hip_w = int(math.hypot(lhx - rhx, lhy - rhy))
            leg_h = int(((lky + rky) // 2) - ((lhy + rhy) // 2))

            if hip_w > 30 and leg_h > 30:
                # Predict user size (M/L) ratio-based on hip/leg ratio
                ratio = hip_w / (leg_h + 1e-6)
                predicted = "M" if ratio < RATIO_THRESHOLD else "L"

                user_size = size_stabilizer.update(predicted)

                preset = SIZE_PRESETS[user_size]
                width_scale = preset["width_scale"]
                height_scale = preset["height_scale"]
                y_lift = preset["y_lift"]

                # angle from hips
                angle_rad = math.atan2(lhy - rhy, lhx - rhx)
                angle_deg = -math.degrees(angle_rad)

                target_w = int(hip_w * width_scale)
                target_h = int(leg_h * height_scale)

                clothing_resized = cv2.resize(shirt_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
                clothing_rot = rotate_rgba(clothing_resized, angle_deg)

                cx = (lhx + rhx) // 2
                cy = (lhy + rhy) // 2

                x = cx - clothing_rot.shape[1] // 2
                y = cy - int(clothing_rot.shape[0] * y_lift)

                # Realism upgrades
                clothing_rot = soften_alpha_edges(clothing_rot, sigma=EDGE_SOFTEN_SIGMA)
                clothing_rot = match_lighting(clothing_rot, frame, x, y, light_min=LIGHT_MIN, light_max=LIGHT_MAX)
                clothing_rot = apply_shadow_from_body(clothing_rot, frame, x, y, strength=SHADOW_STRENGTH)

                frame = overlay_rgba(frame, clothing_rot, x, y, front_mask=front_mask)

        else:
            # ===== SHIRTS: Align to shoulders =====
            shoulder_w = int(math.hypot(lsx - rsx, lsy - rsy))
            torso_h = int(((lhy + rhy) // 2) - ((lsy + rsy) // 2))

            if shoulder_w > 30 and torso_h > 30:
                # Predict user size (M/L) ratio-based
                ratio = shoulder_w / (torso_h + 1e-6)
                predicted = "M" if ratio < RATIO_THRESHOLD else "L"

                user_size = size_stabilizer.update(predicted)

                preset = SIZE_PRESETS[user_size]
                width_scale = preset["width_scale"]
                height_scale = preset["height_scale"]
                y_lift = preset["y_lift"]

                # angle from shoulders
                angle_rad = math.atan2(lsy - rsy, lsx - rsx)
                angle_deg = -math.degrees(angle_rad)

                target_w = int(shoulder_w * width_scale)
                target_h = int(torso_h * height_scale)

                shirt_resized = cv2.resize(shirt_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
                shirt_rot = rotate_rgba(shirt_resized, angle_deg)

                cx = (lsx + rsx) // 2
                cy = (lsy + rsy) // 2

                x = cx - shirt_rot.shape[1] // 2
                y = cy - int(shirt_rot.shape[0] * y_lift)

                # Realism upgrades
                shirt_rot = soften_alpha_edges(shirt_rot, sigma=EDGE_SOFTEN_SIGMA)
                shirt_rot = match_lighting(shirt_rot, frame, x, y, light_min=LIGHT_MIN, light_max=LIGHT_MAX)
                shirt_rot = apply_shadow_from_body(shirt_rot, frame, x, y, strength=SHADOW_STRENGTH)

                frame = overlay_rgba(frame, shirt_rot, x, y, front_mask=front_mask)

    # UI text
    item_type = "Pant/Short" if is_pant_or_short else "Shirt"
    label = f"{item_type}: {os.path.basename(shirt_paths[shirt_index])}  (A/D change)"
    cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Predicted Size: {user_size}", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("Virtual Try-On (Auto Size M/L)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('Q'), 27]:
        break
    elif key in [ord('d'), ord('D')]:
        shirt_index = (shirt_index + 1) % len(shirt_paths)
        shirt_rgba = shirt_cache[shirt_index]
    elif key in [ord('a'), ord('A')]:
        shirt_index = (shirt_index - 1) % len(shirt_paths)
        shirt_rgba = shirt_cache[shirt_index]

cap.release()
cv2.destroyAllWindows()
