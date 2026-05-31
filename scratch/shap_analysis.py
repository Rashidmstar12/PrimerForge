"""SHAP Feature Importance Analysis for SantaLucia and Deep Learning Ensemble."""

import os
import random
import numpy as np
import shap

from primerforge.biophysics import BiophysicsEngine
from primerforge.ml_scorer import MLScorer

# Define the exact 40-dimensional feature names from ml_scorer.py
FEATURE_NAMES = [
    # 1. Thermodynamics Features (8)
    "Forward Tm",
    "Reverse Tm",
    "Tm Difference",
    "Forward Hairpin dG",
    "Reverse Hairpin dG",
    "Forward Homodimer dG",
    "Reverse Homodimer dG",
    "Cross Dimer dG (Heterodimer)",
    
    # 2. Sequence Composition Features (12)
    "Forward GC%",
    "Reverse GC%",
    "Forward Length",
    "Reverse Length",
    "Forward 3' GC Clamp Count",
    "Reverse 3' GC Clamp Count",
    "Forward Homopolymer Run",
    "Reverse Homopolymer Run",
    "Forward 3' Dinuc GC (0/1)",
    "Reverse 3' Dinuc GC (0/1)",
    "Forward 3' Dinuc AA (0/1)",
    "Forward 3' Dinuc TT (0/1)",
    
    # 3. 3' Stability Features (4)
    "Reverse 3' Dinuc AA (0/1)",
    "Reverse 3' Dinuc TT (0/1)",
    "Forward 3' NN Stability dG",
    "Reverse 3' NN Stability dG",
    
    # 4. Target Secondary Structure Features (4)
    "Target Nussinov MFE dG",
    "Target Amplicon GC%",
    "Target Amplicon Length",
    "Target Folding Fraction",
    
    # 5. Pangenome/Variant Features (4)
    "Forward Graph Off-Targets Count",
    "Reverse Graph Off-Targets Count",
    "Forward Minor Variant Proximity",
    "Reverse Minor Variant Proximity",
    
    # 6. Chemical/Enzymatic Features (4)
    "Monovalent Salt Conc (mM)",
    "Divalent Salt Conc (mM)",
    "dNTP Conc (mM)",
    "Polymerase Type (Encoded)",
    
    # 7. BioGNN Features (2)
    "BioGNN Fold Success Probability",
    "BioGNN Structural Confidence",
    
    # 8. DNA Transformer Features (2)
    "DNA Transformer CLS Success",
    "DNA Transformer Attention Confidence"
]

def main() -> None:
    print("================================================================================")
    print("                PRIMERFORGE GLOBAL SHAP FEATURE IMPORTANCE REPORT               ")
    print("================================================================================")
    
    # 1. Initialize MLScorer and retrieve ensembled boosters
    print("Loading MLScorer and GBDT ensembled model boosters...")
    scorer = MLScorer()
    
    if not scorer.models:
        print("Error: No pre-trained GBDT boosters loaded.")
        return
        
    booster = scorer.models[0]
    print(f"Booster loaded successfully. Objective: {booster.params.get('objective', 'regression')}")
    
    # 2. Design realistic candidate primer pairs to extract features from
    print("\nDesigning candidate primer pairs from a high-complexity biological locus...")
    biophys = BiophysicsEngine()
    
    # Realistic 200bp sequence
    target_seq = (
        "CAAATAAGGCGCTCAGTCCGCTGCAAGTTTGTCATGGATGACCTTGGCCAGGGGCGACGCTAACCGTGCTAACGTTGCAG"
        "CTTGTCGTACTTGCTAGCTGACTGACGATCGATCGTTCGCTAGCTAGCTAGCGCATCGATCGATCGATCGATCGATCGAC"
        "TGACTGACTGACTGACTGACTGACTACGTACGTACGT"
    )
    
    candidates = biophys.generate_candidates(target_seq, num_return=100)
    print(f"Generated N={len(candidates)} biophysical candidate primer pairs.")
    
    # 3. Extract 40-dimensional feature matrix
    print("Extracting ensembled 40-dimensional feature matrices (thermodynamics + GNN + GFA + Transformer)...")
    X = []
    for pair in candidates:
        # Mock some specificity and variant proximity coordinates
        spec_data = {
            "f_off_targets": float(random.choice([0, 0, 1])),
            "r_off_targets": float(random.choice([0, 0, 0])),
            "f_var_dist": float(random.choice([20.0, 20.0, 12.0])),
            "r_var_dist": float(random.choice([20.0, 20.0, 20.0])),
            "f_var_maf": 0.0,
            "r_var_maf": 0.0,
        }
        X.append(scorer.extract_features(pair, spec_data))
        
    X = np.array(X, dtype=np.float32)
    print(f"Compiled feature matrix of shape: {X.shape}")
    
    # 4. Compute Tree SHAP Values
    print("\nComputing game-theoretic Tree SHAP value attributions...")
    try:
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X)
        
        # Handle different output structures of SHAP values
        if isinstance(shap_values, list):
            # Binary classification list often has [neg_class_shap, pos_class_shap]
            shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values
            
        print(f"SHAP calculations completed successfully. Shape: {shap_vals.shape}")
    except Exception as e:
        print(f"SHAP TreeExplainer calculation failed: {e}")
        return
        
    # 5. Aggregate and Rank Global Feature Importance
    # Global feature importance = mean absolute SHAP value across all samples
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    
    ranked_indices = np.argsort(mean_abs_shap)[::-1]
    
    print("\n================================================================================")
    print("           RANKED FEATURE IMPORTANCE (MEAN ABSOLUTE SHAP CONTRIBUTION)")
    print("================================================================================")
    print(f"  {'Rank':<5} | {'Biophysical Feature Description':<40} | {'Mean SHAP Value':<15}")
    print("  " + "-" * 66)
    
    for rank_idx, f_idx in enumerate(ranked_indices[:15], 1):
        name = FEATURE_NAMES[f_idx]
        val = mean_abs_shap[f_idx]
        print(f"  {rank_idx:<5} | {name:<40} | {val:.6f}")
        
    print("================================================================================")
    print("  Note: High Mean SHAP indicates features that drive the ensembled model's")
    print("        predictions of empirical PCR amplification success the most.")
    print("================================================================================\n")

if __name__ == "__main__":
    main()
