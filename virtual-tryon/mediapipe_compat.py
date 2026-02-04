"""
MediaPipe compatibility layer for pose detection and segmentation
Fallback to stub implementation when MediaPipe solutions is not available
"""

import numpy as np

# Try to import MediaPipe solutions, fall back to stub if not available
try:
    import mediapipe as mp
    mp_pose_module = mp.solutions.pose
    mp_seg_module = mp.solutions.selfie_segmentation
    HAS_MEDIAPIPE_SOLUTIONS = True
except (ImportError, AttributeError):
    HAS_MEDIAPIPE_SOLUTIONS = False

class Landmark:
    """Lightweight landmark holder"""
    def __init__(self, x=0, y=0, z=0, visibility=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class Landmarks:
    """Landmarks container"""
    def __init__(self, landmarks_list=None):
        self.landmark = []
        if landmarks_list:
            for lm in landmarks_list:
                self.landmark.append(Landmark(lm.x, lm.y, lm.z, lm.visibility))
        else:
            self.landmark = [Landmark() for _ in range(33)]

class PoseLandmarkEnum:
    """Pose landmark indices"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

class PoseResults:
    def __init__(self, pose_landmarks=None):
        self.pose_landmarks = None
        if pose_landmarks is not None:
            self.pose_landmarks = Landmarks(pose_landmarks)

class SegmentationResults:
    def __init__(self, segmentation_mask):
        self.segmentation_mask = segmentation_mask

class Pose:
    """Pose estimator - use MediaPipe if available, otherwise simple OpenCV detection"""
    def __init__(self, static_image_mode=False, model_complexity=1):
        self.use_mediapipe = HAS_MEDIAPIPE_SOLUTIONS
        
        if self.use_mediapipe:
            try:
                self.pose = mp_pose_module.Pose(
                    static_image_mode=static_image_mode,
                    model_complexity=model_complexity,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[OK] Using MediaPipe Pose detector")
            except Exception as e:
                print(f"[ERROR] Failed to initialize MediaPipe Pose: {e}")
                print("[HELP] To fix MediaPipe issues, try:")
                print("       1. pip uninstall mediapipe")
                print("       2. pip install mediapipe")
                print("[WARNING] Falling back to OpenCV-based detection (shoulders/hips only)")
                self.use_mediapipe = False
                self.pose = None
        else:
            if not HAS_MEDIAPIPE_SOLUTIONS:
                print("[WARNING] MediaPipe solutions not available. Install with: pip install mediapipe")
            print("[WARNING] Using OpenCV-based fallback pose detection (shoulders/hips only)")
            self.pose = None
    
    def process(self, image_rgb):
        """Process image and return pose landmarks"""
        if self.use_mediapipe and self.pose is not None:
            try:
                results = self.pose.process(image_rgb)
                if results.pose_landmarks is not None:
                    return PoseResults(results.pose_landmarks.landmark)
            except Exception as e:
                print(f"[ERROR] Pose processing failed: {e}")
        
        # Return empty result - OpenCV will be handled by app.py
        return PoseResults(None)

class SelfieSegmentation:
    """Segmentation - use MediaPipe if available, otherwise return full mask"""
    def __init__(self, model_selection=1):
        self.use_mediapipe = HAS_MEDIAPIPE_SOLUTIONS
        
        if self.use_mediapipe:
            try:
                self.segmentation = mp_seg_module.SelfieSegmentation(model_selection=model_selection)
                print("[OK] Using MediaPipe Segmentation")
            except Exception as e:
                print(f"[ERROR] Failed to initialize MediaPipe Segmentation: {e}")
                print("[HELP] To fix MediaPipe issues, try:")
                print("       1. pip uninstall mediapipe")
                print("       2. pip install mediapipe")
                print("[WARNING] Falling back to full foreground mask")
                self.use_mediapipe = False
                self.segmentation = None
        else:
            if not HAS_MEDIAPIPE_SOLUTIONS:
                print("[WARNING] MediaPipe solutions not available. Install with: pip install mediapipe")
            print("[WARNING] Using full foreground mask as fallback")
            self.segmentation = None
    
    def process(self, image_rgb):
        """Process image and return segmentation mask"""
        if self.use_mediapipe and self.segmentation is not None:
            try:
                results = self.segmentation.process(image_rgb)
                if results.segmentation_mask is not None:
                    return SegmentationResults(results.segmentation_mask)
            except Exception as e:
                print(f"[ERROR] Segmentation processing failed: {e}")
        
        # Return full foreground mask (everyone is foreground)
        height, width = image_rgb.shape[:2]
        return SegmentationResults(np.ones((height, width), dtype=np.float32))

# Make PoseLandmark available at module level
PoseLandmark = PoseLandmarkEnum
