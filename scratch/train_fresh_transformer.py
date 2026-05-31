import os
import glob
from primerforge.ml_scorer import MLScorer

# Clear old models to prevent loading legacy weights
print("Clearing old models from models/ directory...")
for f in glob.glob("models/primerforge_lightgbm*"):
    try:
        os.remove(f)
        print(f"  Removed: {f}")
    except Exception as e:
        print(f"  Failed to remove {f}: {e}")

# Instantiate MLScorer and trigger training
scorer = MLScorer()
print("\nStarting train_ultra_ensemble (which triggers DNA Transformer pre-training)...")
# Run with standard size
scorer.train_ultra_ensemble(target_size=5000, n_samples=2000)

print("\nModel training complete. Reloading models to verify serialization...")
scorer.load()
print(f"Loaded {len(scorer.models)} boosters successfully.")
print(f"Platt Calibration coefficients: platt_a = {scorer.platt_a:.4f}, platt_b = {scorer.platt_b:.4f}")
print("DNA Transformer weights loaded:", hasattr(scorer, "transformer") and scorer.transformer is not None)
