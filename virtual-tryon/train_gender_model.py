"""
Gender Detection Model Training Script - Improved Version
Uses UTKFace dataset with scikit-learn RandomForest for better accuracy
UTKFace filename format: [age]_[gender]_[race]_[date&time].jpg.chip.jpg
Gender: 0 = Male, 1 = Female
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import glob

# Configuration
DATASET_PATH = "../datasets/UTKFace"
MODEL_PATH = "gender_model.pkl"
IMG_SIZE = 64  # Resize images to 64x64

def load_dataset():
    """Load UTKFace dataset and extract gender labels from filenames"""
    print("[INFO] Loading UTKFace dataset...")
    
    images = []
    labels = []
    
    # Get all image files
    image_files = glob.glob(os.path.join(DATASET_PATH, "*.jpg"))
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {DATASET_PATH}")
    
    print(f"[INFO] Found {len(image_files)} images")
    
    for i, img_path in enumerate(image_files):
        try:
            # Parse filename: [age]_[gender]_[race]_[date].jpg.chip.jpg
            filename = os.path.basename(img_path)
            parts = filename.split('_')
            
            if len(parts) < 3:
                continue
            
            # Extract gender (0 = Male, 1 = Female)
            gender = int(parts[1])
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Resize to standard size
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            images.append(img)
            labels.append(gender)
            
            if (i + 1) % 2000 == 0:
                print(f"[INFO] Processed {i + 1}/{len(image_files)} images...")
                
        except Exception as e:
            continue
    
    print(f"[INFO] Successfully loaded {len(images)} images")
    return np.array(images), np.array(labels)

def extract_features(images):
    """Extract robust features from images"""
    print("[INFO] Extracting features...")
    features = []
    
    for i, img in enumerate(images):
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
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
        upper = gray[:IMG_SIZE//3, :]
        feature_list.extend([upper.mean() / 255.0, upper.std() / 128.0])
        
        middle = gray[IMG_SIZE//3:2*IMG_SIZE//3, :]
        feature_list.extend([middle.mean() / 255.0, middle.std() / 128.0])
        
        lower = gray[2*IMG_SIZE//3:, :]
        feature_list.extend([lower.mean() / 255.0, lower.std() / 128.0])
        
        # 6. Symmetry features
        left_half = gray[:, :IMG_SIZE//2]
        right_half = cv2.flip(gray[:, IMG_SIZE//2:], 1)
        min_w = min(left_half.shape[1], right_half.shape[1])
        symmetry_diff = np.abs(left_half[:, :min_w].astype(float) - right_half[:, :min_w].astype(float))
        feature_list.extend([symmetry_diff.mean() / 128.0, symmetry_diff.std() / 64.0])
        
        # 7. Texture features from Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        feature_list.extend([lap.mean() / 128.0, lap.std() / 64.0, lap.var() / 4096.0])
        
        # 8. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (IMG_SIZE * IMG_SIZE)
        feature_list.append(edge_density)
        
        # 9. Downsampled image (8x8 = 64 features)
        small = cv2.resize(gray, (8, 8)).flatten() / 255.0
        feature_list.extend(small)
        
        features.append(feature_list)
        
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(images)} images...")
    
    return np.array(features)

def train_model(X_train, y_train, X_val, y_val):
    """Train Random Forest classifier"""
    print("\n[INFO] Training Random Forest classifier...")
    
    # Create pipeline with scaler and classifier
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # Combine train and validation for final training
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    print("[INFO] Fitting model on combined training data...")
    model.fit(X_combined, y_combined)
    
    # Evaluate on validation set
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    
    print(f"\n[INFO] Training Accuracy: {train_acc:.4f}")
    print(f"[INFO] Validation Accuracy: {val_acc:.4f}")
    
    return model

def main():
    print("=" * 50)
    print("Gender Detection Model Training (Improved)")
    print("=" * 50)
    
    # Load dataset
    images, labels = load_dataset()
    
    # Print class distribution
    male_count = np.sum(labels == 0)
    female_count = np.sum(labels == 1)
    print(f"\n[INFO] Class Distribution:")
    print(f"  Male (0): {male_count}")
    print(f"  Female (1): {female_count}")
    
    # Extract features
    features = extract_features(images)
    print(f"[INFO] Feature vector size: {features.shape[1]}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print(f"\n[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Validation samples: {len(X_val)}")
    print(f"[INFO] Test samples: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Test accuracy
    test_acc = model.score(X_test, y_test)
    print(f"\n[INFO] Test Accuracy: {test_acc:.4f}")
    
    # Detailed evaluation
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Male', 'Female']))
    
    # Save model
    model_data = {
        'model': model,
        'img_size': IMG_SIZE,
        'type': 'sklearn_pipeline'
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n[INFO] Model saved to {MODEL_PATH}")
    print("=" * 50)

if __name__ == "__main__":
    main()
