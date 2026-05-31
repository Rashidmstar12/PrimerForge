import os
import pandas as pd
import numpy as np
from primerforge.biophysics import BiophysicsEngine, PrimerPair, PrimerSequence
from primerforge.ml_scorer import MLScorer

def main():
    # 1. Load data/lab_validation_primers.csv
    csv_path = "data/lab_validation_primers.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)

    # 2. Skip rows with N in sequences
    df = df[~df["forward_seq"].str.upper().str.contains("N") & ~df["reverse_seq"].str.upper().str.contains("N")].copy()

    engine = BiophysicsEngine()
    # Initialize MLScorer with auto_train=False to load the trained model
    scorer = MLScorer(auto_train=False)

    results = []

    for idx, row in df.iterrows():
        gene = row["gene"]
        f_seq = row["forward_seq"]
        r_seq = row["reverse_seq"]
        target_size = int(row["amplicon_size"])
        cell_line = row["cell_line"]
        validated_success = int(row["validated_success"])
        source = row["source"]

        # Compute biophysical features
        f_thermo = engine.calculate_thermo_features(f_seq)
        r_thermo = engine.calculate_thermo_features(r_seq)
        cross_dg = engine.calculate_heterodimer_dg(f_seq, r_seq)

        f_gc = sum(1 for b in f_seq.upper() if b in "GC") / len(f_seq) * 100.0
        r_gc = sum(1 for b in r_seq.upper() if b in "GC") / len(r_seq) * 100.0

        f_primer = PrimerSequence(
            sequence=f_seq,
            start=0,
            length=len(f_seq),
            tm=f_thermo["tm"],
            gc_percent=f_gc,
            hairpin_dg=f_thermo["hairpin_dg"],
            homodimer_dg=f_thermo["homodimer_dg"],
            penalty=0.0
        )

        r_primer = PrimerSequence(
            sequence=r_seq,
            start=0,
            length=len(r_seq),
            tm=r_thermo["tm"],
            gc_percent=r_gc,
            hairpin_dg=r_thermo["hairpin_dg"],
            homodimer_dg=r_thermo["homodimer_dg"],
            penalty=0.0
        )

        pair = PrimerPair(
            forward=f_primer,
            reverse=r_primer,
            product_size=target_size,
            cross_dimer_dg=cross_dg,
            penalty=0.0
        )

        spec_data = {
            "f_off_targets": 0,
            "r_off_targets": 0,
            "f_var_dist": 20.0,
            "r_var_dist": 20.0,
            "f_var_maf": 0.0,
            "r_var_maf": 0.0,
            "salt_monovalent_mm": 50.0,
            "salt_divalent_mm": 1.5,
            "dntp_conc_mm": 0.2,
            "polymerase_encoded": 0.0,
            "additive_dmso": 0.0,
            "mg_conc_mm": 1.5,
            "specificity_encoded": 1.0,
        }

        # Run prediction
        prob = scorer.predict_success(pair, spec_data)

        # Classification threshold (using standard 0.50 cutoff for predicted probability)
        predicted_success = 1 if prob >= 0.50 else 0
        correct = 1 if predicted_success == validated_success else 0

        results.append({
            "gene": gene,
            "forward_seq": f_seq,
            "reverse_seq": r_seq,
            "amplicon_size": target_size,
            "cell_line": cell_line,
            "validated_success": validated_success,
            "source": source,
            "predicted_prob": round(prob, 4),
            "predicted_success": predicted_success,
            "correct": correct
        })

    res_df = pd.DataFrame(results)

    # Compute metrics
    total = len(res_df)
    correct_count = res_df["correct"].sum()
    accuracy = correct_count / total

    positives = res_df[res_df["validated_success"] == 1]
    negatives = res_df[res_df["validated_success"] == 0]

    tp = sum((positives["predicted_success"] == 1))
    fn = sum((positives["predicted_success"] == 0))
    tn = sum((negatives["predicted_success"] == 0))
    fp = sum((negatives["predicted_success"] == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    print("\n" + "="*80)
    print("                     PRIMERFORGE RETROSPECTIVE VALIDATION")
    print("="*80)
    print(f"Accuracy:    {accuracy:.4f} ({correct_count}/{total})")
    print(f"Sensitivity: {sensitivity:.4f} (TP: {tp}, FN: {fn})")
    print(f"Specificity: {specificity:.4f} (TN: {tn}, FP: {fp})")
    print(f"PPV (Prec.): {ppv:.4f}")
    print(f"NPV:         {npv:.4f}")
    print("-"*80)

    # Print clean table
    print(res_df[["gene", "validated_success", "predicted_prob", "predicted_success", "correct"]].to_string(index=False))
    print("="*80 + "\n")

    # Save to data/retrospective_validation_results.csv
    results_path = "data/retrospective_validation_results.csv"
    res_df.to_csv(results_path, index=False)
    print(f"Saved validation details to {results_path}")

if __name__ == "__main__":
    main()
