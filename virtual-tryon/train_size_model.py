"""
Size Prediction Model Training Script (v3 - Runtime Compatible)
================================================================
Trains ML classifiers for shirt and pant size prediction using the
Personalized Clothing & Body Measurements dataset.

KEY DESIGN:
  The models are trained on RATIO features that are directly computable
  from MediaPipe pose landmarks at runtime. All features are ratios of
  visible body dimensions (shoulder_w, torso_h, hip_w, leg_h), making
  them scale-invariant — the same ratio whether measured in cm or pixels.

  Dataset measurements are converted to "visible equivalents" using
  standard anthropometric proportions before computing ratios.

Labels:
  The dataset's "RecommendedSize" column does NOT correlate with body 
  measurements (verified: r < 0.06). We derive proper labels from 
  standard clothing size charts (chest/shoulder for shirts, waist/hip 
  for pants).

Dataset: https://www.kaggle.com/datasets/zara2099/personalized-clothing-and-body-measurements-data
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "clothing_size", "personalized_clothing_dataset.csv"
)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # virtual-tryon/
SHIRT_MODEL_PATH = os.path.join(OUTPUT_DIR, "shirt_size_model.pkl")
PANT_MODEL_PATH = os.path.join(OUTPUT_DIR, "pant_size_model.pkl")

# Size classes used in this project
SHIRT_SIZE_ORDER = ["S", "M", "L", "XL"]
PANT_SIZE_ORDER = ["28", "30", "32", "34", "36", "38"]


# ──────────────────────────────────────────────
# Standard Clothing Size Charts (cm)
# Based on international men's sizing standards
# ──────────────────────────────────────────────

def assign_shirt_size(row):
    """
    Assign shirt size based on shoulder width — the primary VISIBLE
    measurement that a camera can detect via MediaPipe landmarks.
    
    Standard bi-deltoid breadth ranges (men):
      S  :  < 41 cm       (~25th percentile in dataset)
      M  :  41 - 48 cm    (~50th percentile)
      L  :  48 - 54 cm    (~75th percentile)
      XL :  54+ cm
    
    Shoulder width is the most reliable predictor available from
    a single front-facing camera. In real-world data it correlates
    strongly with chest circumference (r≈0.7-0.8).
    """
    shoulder = row['ShoulderWidth_cm']
    
    if shoulder < 41:
        return "S"
    elif shoulder < 48:
        return "M"
    elif shoulder < 54:
        return "L"
    return "XL"


def assign_pant_size(row):
    """
    Assign pant size as numeric waist size (inches) based on hip
    circumference — the primary VISIBLE lower-body measurement
    detectable via MediaPipe hip landmarks.
    
    Hip-to-waist mapping (standard men's sizing):
      Hip < 93 cm   → Waist 28
      Hip 93-104 cm → Waist 30
      Hip 104-116 cm→ Waist 32
      Hip 116-127 cm→ Waist 34
      Hip 127-133 cm→ Waist 36
      Hip 133+ cm   → Waist 38
    """
    hip = row['Hip_cm']
    
    if hip < 93:
        return "28"
    elif hip < 104:
        return "30"
    elif hip < 116:
        return "32"
    elif hip < 127:
        return "34"
    elif hip < 133:
        return "36"
    return "38"


# ──────────────────────────────────────────────
# Feature Engineering (v3 - Runtime Compatible)
# ──────────────────────────────────────────────

# Anthropometric constants for converting dataset cm measurements
# to "visible equivalent" measurements (what a camera would see).
TORSO_FRACTION = 0.30       # shoulder-to-hip ≈ 30% of total height
HIP_CIRC_TO_WIDTH = 3.14    # hip circumference → visible width (π)
UPPER_LEG_FRACTION = 0.53   # hip-to-knee ≈ 53% of total leg length


def estimate_visible_measurements(df):
    """
    Convert dataset measurements (cm) to pixel-equivalent visible dimensions.
    
    At runtime, MediaPipe gives us 4 pixel measurements:
      - shoulder_w: distance between left/right shoulder landmarks
      - torso_h:    shoulder_y → hip_y (visual trunk height)
      - hip_w:      distance between left/right hip landmarks
      - leg_h:      hip_y → knee_y (upper leg visual height)
    
    We estimate equivalent values from the dataset so that RATIOS
    computed here match the RATIOS from MediaPipe pixel data.
    """
    vis = pd.DataFrame()
    vis['shoulder_w'] = df['ShoulderWidth_cm']                      # direct match
    vis['torso_h'] = df['Height_cm'] * TORSO_FRACTION               # shoulder-to-hip distance
    vis['hip_w'] = df['Hip_cm'] / HIP_CIRC_TO_WIDTH                 # circumference → width
    vis['leg_h'] = df['LegLength_cm'] * UPPER_LEG_FRACTION          # total leg → hip-to-knee
    return vis


def engineer_shirt_features(df):
    """
    Shirt features: ratios of visible body dimensions.
    All features are computable from MediaPipe landmarks at runtime.
    
    1. shoulder_torso_ratio  - primary: wider shoulders relative to torso → larger size
    2. shoulder_hip_ratio    - V-shape indicator (shoulder vs hip width)
    3. torso_leg_ratio       - body proportion (short torso vs long legs)
    4. shoulder_leg_ratio    - shoulder width vs leg height
    5. hip_torso_ratio       - hip width vs torso height
    6. body_upper_ratio      - fraction of upper body in total body
    """
    vis = estimate_visible_measurements(df)
    features = pd.DataFrame()
    
    features['shoulder_torso_ratio'] = vis['shoulder_w'] / (vis['torso_h'] + 1e-6)
    features['shoulder_hip_ratio']   = vis['shoulder_w'] / (vis['hip_w'] + 1e-6)
    features['torso_leg_ratio']      = vis['torso_h'] / (vis['leg_h'] + 1e-6)
    features['shoulder_leg_ratio']   = vis['shoulder_w'] / (vis['leg_h'] + 1e-6)
    features['hip_torso_ratio']      = vis['hip_w'] / (vis['torso_h'] + 1e-6)
    features['body_upper_ratio']     = vis['torso_h'] / (vis['torso_h'] + vis['leg_h'] + 1e-6)
    
    return features


def engineer_pant_features(df):
    """
    Pant features: ratios of visible body dimensions.
    All features are computable from MediaPipe landmarks at runtime.
    
    1. hip_leg_ratio         - primary: wider hips relative to legs → larger size
    2. hip_shoulder_ratio    - hip vs shoulder width
    3. leg_torso_ratio       - leg height vs torso height
    4. hip_torso_ratio       - hip width vs torso height
    5. shoulder_leg_ratio    - shoulder width vs leg height
    6. body_lower_ratio      - fraction of lower body in total body
    """
    vis = estimate_visible_measurements(df)
    features = pd.DataFrame()
    
    features['hip_leg_ratio']        = vis['hip_w'] / (vis['leg_h'] + 1e-6)
    features['hip_shoulder_ratio']   = vis['hip_w'] / (vis['shoulder_w'] + 1e-6)
    features['leg_torso_ratio']      = vis['leg_h'] / (vis['torso_h'] + 1e-6)
    features['hip_torso_ratio']      = vis['hip_w'] / (vis['torso_h'] + 1e-6)
    features['shoulder_leg_ratio']   = vis['shoulder_w'] / (vis['leg_h'] + 1e-6)
    features['body_lower_ratio']     = vis['leg_h'] / (vis['torso_h'] + vis['leg_h'] + 1e-6)
    
    return features


# ──────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────

def train_and_evaluate(X, y, model_type="shirt"):
    """Train model with cross-validation and return best pipeline."""
    
    le = LabelEncoder()
    if model_type == "pant":
        le.classes_ = np.array(PANT_SIZE_ORDER)
    else:
        le.classes_ = np.array(SHIRT_SIZE_ORDER)
    y_enc = le.transform(y)
    
    # Split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # Try multiple classifiers, pick best
    models = {
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                       class_weight='balanced', probability=True, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_split=4,
                class_weight='balanced', random_state=42, n_jobs=-1))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42))
        ]),
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    print(f"\n  Comparing classifiers:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        mean_acc = scores.mean()
        print(f"    {name:22s}: CV accuracy = {mean_acc:.4f} (+/- {scores.std():.4f})")
        
        if mean_acc > best_score:
            best_score = mean_acc
            best_model = pipeline
            best_name = name
    
    print(f"\n  Best model: {best_name} ({best_score:.4f})")
    
    # Train best model on training data and evaluate on test
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"\n  Classification Report:")
    print(report)
    
    # Retrain on ALL data for deployment
    best_model.fit(X, y_enc)
    
    return best_model, le, test_acc, best_name


def save_model(pipeline, label_encoder, feature_names, model_path, model_info):
    """Save model with metadata for runtime loading."""
    model_data = {
        'model': pipeline,
        'label_encoder': label_encoder,
        'feature_names': list(feature_names),
        'size_classes': list(label_encoder.classes_),
        'model_info': model_info,
        'type': 'size_prediction_v3_runtime_compatible',
        'version': '3.0'
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  Saved: {model_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Size Prediction Model Training (v3 - Runtime Compatible)")
    print("  Ratio features computable from MediaPipe landmarks")
    print("=" * 60)
    
    # Load dataset
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        sys.exit(1)
    
    df = pd.read_csv(DATASET_PATH)
    print(f"[OK] Loaded: {len(df)} samples")
    
    # ─── Assign proper size labels ───
    print("\n[INFO] Assigning sizes from standard clothing size charts...")
    df['shirt_size'] = df.apply(assign_shirt_size, axis=1)
    df['pant_size'] = df.apply(assign_pant_size, axis=1)
    
    print("\nShirt size distribution (chart-based):")
    print(df['shirt_size'].value_counts().reindex(SHIRT_SIZE_ORDER).to_string())
    
    print("\nPant size distribution (chart-based):")
    print(df['pant_size'].value_counts().reindex(PANT_SIZE_ORDER).to_string())
    
    # Verify correlation with new labels
    shirt_num_map = {"S": 0, "M": 1, "L": 2, "XL": 3}
    pant_num_map = {"28": 0, "30": 1, "32": 2, "34": 3, "36": 4, "38": 5}
    df['shirt_num'] = df['shirt_size'].map(shirt_num_map)
    df['pant_num'] = df['pant_size'].map(pant_num_map)
    
    print("\n[CHECK] Correlation with chart-based labels:")
    print(f"  Chest     vs shirt_size: r = {df['Chest_cm'].corr(df['shirt_num']):+.4f}")
    print(f"  Shoulder  vs shirt_size: r = {df['ShoulderWidth_cm'].corr(df['shirt_num']):+.4f}")
    print(f"  Waist     vs pant_size:  r = {df['Waist_cm'].corr(df['pant_num']):+.4f}")
    print(f"  Hip       vs pant_size:  r = {df['Hip_cm'].corr(df['pant_num']):+.4f}")
    
    # ─── SHIRT MODEL ────────────────────
    print("\n" + "=" * 60)
    print("  SHIRT SIZE MODEL")
    print("=" * 60)
    
    shirt_features = engineer_shirt_features(df)
    shirt_labels = df['shirt_size']
    
    shirt_model, shirt_le, shirt_acc, shirt_name = train_and_evaluate(
        shirt_features.values, shirt_labels, model_type="shirt"
    )
    save_model(shirt_model, shirt_le, shirt_features.columns, SHIRT_MODEL_PATH, shirt_name)
    
    # ─── PANT MODEL ─────────────────────
    print("\n" + "=" * 60)
    print("  PANT SIZE MODEL (numeric waist sizes)")
    print("=" * 60)
    
    pant_features = engineer_pant_features(df)
    pant_labels = df['pant_size']
    
    pant_model, pant_le, pant_acc, pant_name = train_and_evaluate(
        pant_features.values, pant_labels, model_type="pant"
    )
    save_model(pant_model, pant_le, pant_features.columns, PANT_MODEL_PATH, pant_name)
    
    # ─── SUMMARY ────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Shirt: {shirt_name} - {shirt_acc:.1%} accuracy")
    print(f"  Pant:  {pant_name} - {pant_acc:.1%} accuracy")
    print(f"  Shirt classes: {SHIRT_SIZE_ORDER}")
    print(f"  Pant classes:  {PANT_SIZE_ORDER}")
    print(f"  Models saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
