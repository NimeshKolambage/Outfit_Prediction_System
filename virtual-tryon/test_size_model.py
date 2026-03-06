"""
Quick test to verify ML size models load and predict at runtime.
Run: python test_size_model.py
"""
import pickle
import numpy as np

def main():
    # Load both models
    with open("shirt_size_model.pkl", "rb") as f:
        shirt = pickle.load(f)
    with open("pant_size_model.pkl", "rb") as f:
        pant = pickle.load(f)

    print("=" * 65)
    print("  ML SIZE MODEL VERIFICATION")
    print("=" * 65)

    print("\n--- SHIRT MODEL ---")
    print(f"  Version : {shirt['version']}")
    print(f"  Type    : {shirt['type']}")
    print(f"  Algo    : {shirt['model_info']}")
    print(f"  Features: {shirt['feature_names']}")
    print(f"  Classes : {shirt['size_classes']}")

    print("\n--- PANT MODEL ---")
    print(f"  Version : {pant['version']}")
    print(f"  Type    : {pant['type']}")
    print(f"  Algo    : {pant['model_info']}")
    print(f"  Features: {pant['feature_names']}")
    print(f"  Classes : {pant['size_classes']}")

    # Simulate MediaPipe pixel measurements (as if from a webcam at ~6 feet)
    print("\n--- TEST PREDICTIONS (simulated MediaPipe pixel values) ---")
    print(f"  {'Body Type':<16} | {'shoulder':>8} {'torso':>6} {'hip':>5} {'leg':>5} | {'Shirt':>5}  {'Pant':>5}")
    print("  " + "-" * 62)

    test_cases = [
        ("Small person",    100, 180,  80, 200),
        ("Medium person",   140, 170, 100, 190),
        ("Large person",    170, 160, 130, 180),
        ("XL person",       200, 150, 160, 170),
        ("Typical webcam",  150, 165, 110, 185),
    ]

    for label, sw, th, hw, lh in test_cases:
        # Shirt features (same order as training)
        sf = np.array([[
            sw / (th + 1e-6),       # shoulder_torso_ratio
            sw / (hw + 1e-6),       # shoulder_hip_ratio
            th / (lh + 1e-6),       # torso_leg_ratio
            sw / (lh + 1e-6),       # shoulder_leg_ratio
            hw / (th + 1e-6),       # hip_torso_ratio
            th / (th + lh + 1e-6),  # body_upper_ratio
        ]])
        shirt_idx = shirt["model"].predict(sf)[0]
        shirt_size = shirt["label_encoder"].inverse_transform([shirt_idx])[0]

        # Pant features (same order as training)
        pf = np.array([[
            hw / (lh + 1e-6),       # hip_leg_ratio
            hw / (sw + 1e-6),       # hip_shoulder_ratio
            lh / (th + 1e-6),       # leg_torso_ratio
            hw / (th + 1e-6),       # hip_torso_ratio
            sw / (lh + 1e-6),       # shoulder_leg_ratio
            lh / (th + lh + 1e-6),  # body_lower_ratio
        ]])
        pant_idx = pant["model"].predict(pf)[0]
        pant_size = pant["label_encoder"].inverse_transform([pant_idx])[0]

        print(f"  {label:<16} | {sw:>6}px {th:>4}px {hw:>3}px {lh:>3}px | {shirt_size:>5}  {pant_size:>5}")

    print("\n[OK] ML models loaded and predicting successfully!")
    print("     When the server runs, look for '[ML]' log lines to confirm live usage.")
    print("=" * 65)


if __name__ == "__main__":
    main()
