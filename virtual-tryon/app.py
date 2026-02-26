import cv2
import sys
from tryon_engine import TryOnEngine

# -------------------------
# Camera
# -------------------------
CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("[ERROR] Cannot open camera. Check permissions or if another app is using the camera.")

engine = TryOnEngine(shirt_dir="shirts")

print("[OK] Running... Controls: [A] prev shirt | [D] next shirt | [R] reset for next user | [Q]/[ESC] exit")
print("[INFO] Stand at 7.5 feet from camera for accurate sizing")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame, info = engine.process_frame(frame)

    h, w = frame.shape[:2]
    
    # Show distance instruction
    cv2.putText(frame, "Stand at 7.5 feet from camera", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
    
    # Show countdown state
    state = info['state']
    if state == "COUNTDOWN":
        countdown = info['countdown']
        cv2.putText(frame, f"Countdown: {countdown} seconds", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
        # Show distance feedback during countdown
        if info['distance_warning']:
            warning_color = (0, 0, 255) if "TOO" in info['distance_warning'] else (0, 255, 0)
            cv2.putText(frame, info['distance_warning'], (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, warning_color, 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Position yourself", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    elif state == "DETECTING":
        cv2.putText(frame, "DETECTING SIZE...", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        # Show distance warning - MUST be valid to detect
        if info['distance_warning']:
            warning_color = (0, 0, 255) if "TOO" in info['distance_warning'] else (0, 255, 0)
            cv2.putText(frame, info['distance_warning'], (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, warning_color, 3, cv2.LINE_AA)
    elif info['size_locked'] and "OK" in info.get('distance_warning', ''):
        # ONLY show size if currently at correct distance (not just locked)
        cv2.putText(frame, f"YOUR SIZE: {info['user_size']}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Item: {info['item_type']} - {info['filename']}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, info['distance_warning'], (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    elif info['size_locked']:
        # Size was locked but now distance is wrong - show warning
        cv2.putText(frame, "Move back to 7 feet!", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        if info['distance_warning']:
            warning_color = (0, 0, 255) if "TOO" in info['distance_warning'] else (0, 255, 0)
            cv2.putText(frame, info['distance_warning'], (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, warning_color, 3, cv2.LINE_AA)
    
    # Show controls
    cv2.putText(frame, "[A/D] Change item | [R] Next user | [Q] Exit", (15, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Virtual Try-On (Size Detection)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('Q'), 27]:
        break
    elif key in [ord('d'), ord('D')]:
        engine.next_shirt()
    elif key in [ord('a'), ord('A')]:
        engine.prev_shirt()
    elif key in [ord('r'), ord('R')]:
        engine.reset_for_next_user()

cap.release()
cv2.destroyAllWindows()
