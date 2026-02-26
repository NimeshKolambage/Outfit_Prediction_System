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
    
    # Show countdown state
    state = info['state']
    if state == "COUNTDOWN":
        countdown = info['countdown']
        cv2.putText(frame, f"Countdown: {countdown} seconds", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
        # Show distance feedback during countdown
        if info['distance_warning']:
            warning_color = (0, 0, 255) if "TOO" in info['distance_warning'] else (0, 255, 0)
            cv2.putText(frame, info['distance_warning'], (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, warning_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "Stand at 6 feet", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    elif state == "DETECTING":
        cv2.putText(frame, "DETECTING SIZE...", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        # Show distance warning
        if info['distance_warning']:
            warning_color = (0, 0, 255) if "TOO" in info['distance_warning'] else (0, 255, 0)
            cv2.putText(frame, info['distance_warning'], (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, warning_color, 2, cv2.LINE_AA)
    
    elif state == "READY":
        # Size locked - show sizes continuously and NO warnings
        shirt_size = info.get('user_size_shirt', '--')
        pant_size = info.get('user_size_pant', '--')
        
        cv2.putText(frame, "SIZE LOCKED ✓", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Shirt: {shirt_size}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Pants: {pant_size}", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Item: {info['filename']}", (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
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
