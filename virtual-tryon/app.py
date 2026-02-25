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

print("[OK] Running... Controls: [A] prev shirt | [D] next shirt | [Q]/[ESC] exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame, info = engine.process_frame(frame)

    # UI text
    label = f"{info['item_type']}: {info['filename']}  (A/D change)"
    cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Predicted Size: {info['user_size']}",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Virtual Try-On (Auto Size M/L)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('Q'), 27]:
        break
    elif key in [ord('d'), ord('D')]:
        engine.next_shirt()
    elif key in [ord('a'), ord('A')]:
        engine.prev_shirt()

cap.release()
cv2.destroyAllWindows()
