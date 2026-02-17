"""
Test script for body_measurements.py module.
Run this to test measurement and virtual try-on functionality with webcam.

Key tests:
  1. MediaPipe Pose loading
  2. Real-time measurement detection
  3. Size prediction
  4. Measurement visualization
  5. Virtual try-on overlay (if shirts available)

Press 'q' to exit, 's' to toggle sleeve mode, 'n/p' to change shirts
"""

import cv2
import sys
import os
from body_measurements import BodyMeasurements, VirtualTryOn, HAS_MEDIAPIPE


def test_body_measurements_webcam():
    """Test BodyMeasurements with live webcam feed."""
    
    print("\n" + "="*60)
    print("BODY MEASUREMENTS TEST - Live Webcam")
    print("="*60)
    print(f"MediaPipe available: {HAS_MEDIAPIPE}")
    
    if not HAS_MEDIAPIPE:
        print("[ERROR] MediaPipe not available. Cannot run test.")
        print("Install with: pip install mediapipe")
        return False
    
    # Initialize measurement engine
    bm = BodyMeasurements()
    if bm.pose is None:
        print("[ERROR] Failed to initialize BodyMeasurements")
        return False
    
    # Initialize virtual try-on
    vto = VirtualTryOn()
    
    # Test settings
    gender = "Male"
    sleeve_mode = "long"
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return False
    
    print("\n[TEST] Starting webcam capture...")
    print("Controls:")
    print("  'q'   - Quit")
    print("  'g'   - Toggle gender (Male/Female)")
    print("  's'   - Toggle sleeve mode (long/short)")
    print("  'n'   - Next shirt")
    print("  'p'   - Previous shirt")
    print("  'v'   - Toggle virtual try-on")
    print()
    
    enable_vto = True
    frame_count = 0
    detected_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        frame_count += 1
        
        # Flip for selfie view
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Get measurements
        measurements = bm.measure(frame, gender=gender, sleeve_mode=sleeve_mode)
        
        if measurements.get("detected"):
            detected_count += 1
            size = bm.predict_size(measurements, gender=gender)
            
            # Draw measurements
            frame = bm.draw_measurements(
                frame, measurements, 
                gender=gender, 
                size=size,
                sleeve_mode=sleeve_mode
            )
            
            # Draw virtual try-on if enabled
            if enable_vto:
                frame = vto.apply(frame, measurements, size=size)
        else:
            # Guide text
            cv2.putText(frame, "No pose detected - stand in frame", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        
        # Draw HUD info
        cv2.putText(frame, f"Frames: {frame_count} | Detected: {detected_count}", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Mode: {sleeve_mode} | Gender: {gender} | VTO: {'ON' if enable_vto else 'OFF'}", 
                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display
        cv2.imshow("Body Measurements Test", frame)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[OK] Exiting test...")
            break
        elif key == ord('g'):
            gender = "Female" if gender == "Male" else "Male"
            print(f"[OK] Gender changed to: {gender}")
        elif key == ord('s'):
            sleeve_mode = "short" if sleeve_mode == "long" else "long"
            print(f"[OK] Sleeve mode changed to: {sleeve_mode}")
        elif key == ord('n'):
            vto.next_shirt()
            if vto.shirt_names:
                print(f"[OK] Next shirt: {vto.shirt_names[vto.current_index]}")
        elif key == ord('p'):
            vto.prev_shirt()
            if vto.shirt_names:
                print(f"[OK] Previous shirt: {vto.shirt_names[vto.current_index]}")
        elif key == ord('v'):
            enable_vto = not enable_vto
            print(f"[OK] Virtual try-on: {'ON' if enable_vto else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total frames: {frame_count}")
    print(f"Poses detected: {detected_count}")
    if frame_count > 0:
        detect_rate = (detected_count / frame_count) * 100
        print(f"Detection rate: {detect_rate:.1f}%")
    print(f"Shirts loaded: {len(vto.shirt_names)}")
    if vto.shirt_names:
        print(f"Available shirts: {', '.join(vto.shirt_names)}")
    print("[OK] Test completed successfully!")
    print("="*60 + "\n")
    
    return True


def quick_test_measurements():
    """Quick single-frame test with a static image or webcam capture."""
    print("\n" + "="*60)
    print("QUICK TEST - Single Frame Measurement")
    print("="*60)
    
    if not HAS_MEDIAPIPE:
        print("[ERROR] MediaPipe not available")
        return False
    
    bm = BodyMeasurements()
    if bm.pose is None:
        return False
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return False
    
    print("[OK] Capturing frame...")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[ERROR] Failed to capture frame")
        return False
    
    frame = cv2.flip(frame, 1)
    
    # Test both genders
    for gender in ["Male", "Female"]:
        for sleeve_mode in ["long", "short"]:
            print(f"\n  Testing: {gender}, {sleeve_mode} sleeve")
            measurements = bm.measure(frame, gender=gender, sleeve_mode=sleeve_mode)
            
            if measurements["detected"]:
                size = bm.predict_size(measurements, gender=gender)
                print(f"    [DETECTED]")
                print(f"    - Shoulder: {measurements['shoulder_cm']} cm")
                print(f"    - Chest: {measurements['chest_cm']} cm")
                print(f"    - Arm: {measurements['arm_cm']} cm")
                print(f"    - Predicted size: {size}")
                print(f"    - Pixel->CM scale: {measurements['pixel_to_cm']}")
            else:
                print(f"    [NO POSE DETECTED]")
    
    print("\n[OK] Quick test completed!")
    print("="*60 + "\n")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BODY MEASUREMENTS MODULE - TEST SUITE")
    print("="*60)
    
    # Check dependencies
    print("\n[CHECK] Dependencies...")
    deps_ok = True
    try:
        import cv2
        print(f"  [OK] OpenCV: {cv2.__version__}")
    except:
        print("  [NO] OpenCV not found")
        deps_ok = False
    
    try:
        import numpy
        print(f"  [OK] NumPy: {numpy.__version__}")
    except:
        print("  [NO] NumPy not found")
        deps_ok = False
    
    try:
        import mediapipe
        print(f"  [OK] MediaPipe: {mediapipe.__version__}")
    except:
        print("  [NO] MediaPipe not found - install with: pip install mediapipe")
        deps_ok = False
    
    if not deps_ok:
        print("\n[ERROR] Missing dependencies. Install with:")
        print("  pip install opencv-python mediapipe numpy")
        sys.exit(1)
    
    # Check shirts directory
    print("\n[CHECK] Resources...")
    shirt_dir = os.path.join(os.path.dirname(__file__), "shirts")
    if os.path.isdir(shirt_dir):
        shirt_count = len([f for f in os.listdir(shirt_dir) if f.endswith('.png')])
        print(f"  [OK] Shirts directory exists ({shirt_count} shirts)")
    else:
        print(f"  [NOTE] Shirts directory not found: {shirt_dir}")
        print("    Virtual try-on will not work - create '/shirts' folder with PNG files")
    
    # Run tests
    print("\n[SELECT] Test mode:")
    print("  1. Live webcam (interactive)")
    print("  2. Quick test (single frame)")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        success = test_body_measurements_webcam()
    elif choice == "2":
        success = quick_test_measurements()
    else:
        print("[ERROR] Invalid choice")
        sys.exit(1)
    
    sys.exit(0 if success else 1)
