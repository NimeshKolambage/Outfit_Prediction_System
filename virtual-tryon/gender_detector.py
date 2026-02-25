"""
Gender Detection Module - Updated for sklearn model
Loads pretrained sklearn pipeline and detects Male/Female from face images
"""

import cv2
import numpy as np
import pickle
import os

IMG_SIZE = 64

class GenderDetector:
    """Gender detection using trained sklearn model"""
    
    def __init__(self, model_path="gender_model.pkl"):
        self.model_path = model_path
        self.img_size = IMG_SIZE
        self.model_loaded = False
        self.model = None
        
        # Face detection using Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load model if exists
        self.load_model()
    
    def load_model(self):
        """Load the trained gender model"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Check model type
                if model_data.get('type') == 'sklearn_pipeline':
                    self.model = model_data['model']
                    self.img_size = model_data.get('img_size', 64)
                    self.model_loaded = True
                    print(f"[OK] Gender model (sklearn) loaded from {self.model_path}")
                else:
                    # Old format - neural network
                    print("[WARNING] Old model format detected. Please retrain the model.")
                    self.model_loaded = False
                    
            except Exception as e:
                print(f"[ERROR] Failed to load gender model: {e}")
                self.model_loaded = False
        else:
            print(f"[WARNING] Gender model not found at {self.model_path}")
            print("[INFO] Run train_gender_model.py first to train the model")
            self.model_loaded = False
    
    def extract_features(self, img):
        """Extract features from BGR image - must match training features exactly"""
        if len(img.shape) == 2:
            # Grayscale image, convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Ensure correct size
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        feature_list = []
        
        # 1. Grayscale histogram (16 bins)
        hist_gray = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        hist_gray = hist_gray / (hist_gray.sum() + 1e-6)
        feature_list.extend(hist_gray)
        
        # 2. Color histogram (RGB - 8 bins each)
        for c in range(3):
            hist_c = cv2.calcHist([img], [c], None, [8], [0, 256]).flatten()
            hist_c = hist_c / (hist_c.sum() + 1e-6)
            feature_list.extend(hist_c)
        
        # 3. HOG-like features (gradient orientations)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)
        
        # Gradient magnitude histogram (8 bins)
        hist_mag = np.histogram(magnitude, bins=8, range=(0, magnitude.max() + 1e-6))[0]
        hist_mag = hist_mag / (hist_mag.sum() + 1e-6)
        feature_list.extend(hist_mag)
        
        # Gradient orientation histogram (8 bins)
        hist_ori = np.histogram(orientation, bins=8, range=(-np.pi, np.pi))[0]
        hist_ori = hist_ori / (hist_ori.sum() + 1e-6)
        feature_list.extend(hist_ori)
        
        # 4. Local Binary Pattern-like features
        center = gray[1:-1, 1:-1].astype(np.float32)
        neighbors = [
            gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:],
            gray[1:-1, :-2], gray[1:-1, 2:],
            gray[2:, :-2], gray[2:, 1:-1], gray[2:, 2:]
        ]
        lbp_code = np.zeros_like(center, dtype=np.uint8)
        for idx, n in enumerate(neighbors):
            lbp_code += ((n >= center).astype(np.uint8) << idx)
        
        hist_lbp = np.histogram(lbp_code, bins=32, range=(0, 256))[0]
        hist_lbp = hist_lbp / (hist_lbp.sum() + 1e-6)
        feature_list.extend(hist_lbp)
        
        # 5. Face region statistics
        upper = gray[:self.img_size//3, :]
        feature_list.extend([upper.mean() / 255.0, upper.std() / 128.0])
        
        middle = gray[self.img_size//3:2*self.img_size//3, :]
        feature_list.extend([middle.mean() / 255.0, middle.std() / 128.0])
        
        lower = gray[2*self.img_size//3:, :]
        feature_list.extend([lower.mean() / 255.0, lower.std() / 128.0])
        
        # 6. Symmetry features
        left_half = gray[:, :self.img_size//2]
        right_half = cv2.flip(gray[:, self.img_size//2:], 1)
        min_w = min(left_half.shape[1], right_half.shape[1])
        symmetry_diff = np.abs(left_half[:, :min_w].astype(float) - right_half[:, :min_w].astype(float))
        feature_list.extend([symmetry_diff.mean() / 128.0, symmetry_diff.std() / 64.0])
        
        # 7. Texture features from Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        feature_list.extend([lap.mean() / 128.0, lap.std() / 64.0, lap.var() / 4096.0])
        
        # 8. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (self.img_size * self.img_size)
        feature_list.append(edge_density)
        
        # 9. Downsampled image (8x8 = 64 features)
        small = cv2.resize(gray, (8, 8)).flatten() / 255.0
        feature_list.extend(small)
        
        return np.array(feature_list)
    
    def predict(self, features):
        """Run sklearn model prediction"""
        X = features.reshape(1, -1)
        
        # Get prediction and probability
        pred_class = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        confidence = proba[pred_class]
        
        return pred_class, confidence
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )
        return faces
    
    def detect_gender(self, frame):
        """
        Detect gender from frame
        Returns: list of (face_rect, gender, confidence)
        """
        results = []
        
        if not self.model_loaded:
            return results
        
        try:
            faces = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                # Extract face region (BGR)
                face_roi = frame[y:y+h, x:x+w]
                
                # Resize to model input size
                face_resized = cv2.resize(face_roi, (self.img_size, self.img_size))
                
                # Extract features and predict
                features = self.extract_features(face_resized)
                pred_class, confidence = self.predict(features)
                
                # Gender label
                gender = "Male" if pred_class == 0 else "Female"
                
                results.append({
                    'rect': (x, y, w, h),
                    'gender': gender,
                    'confidence': confidence
                })
        
        except Exception as e:
            print(f"[ERROR] Gender detection failed: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def draw_results(self, frame, results):
        """Draw gender label on frame (no face box)"""
        if results:
            gender = results[0]['gender']
            label = f"Gender - {gender}"
            
            # Choose color based on gender
            if gender == "Male":
                color = (255, 150, 50)  # Blue for male
            else:
                color = (180, 50, 255)  # Pink for female
            
            # Draw simple text at bottom left
            cv2.putText(frame, label, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame


# Test function
def test_gender_detector():
    """Test gender detector with webcam"""
    detector = GenderDetector()
    
    if not detector.model_loaded:
        print("[ERROR] Cannot test - model not loaded")
        return
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return
    
    print("[INFO] Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect gender
        results = detector.detect_gender(frame)
        
        # Draw results
        frame = detector.draw_results(frame, results)
        
        # Show frame
        cv2.imshow("Gender Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_gender_detector()
