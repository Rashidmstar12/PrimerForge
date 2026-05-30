"""Command-line utility for regularized transfer learning on custom wet-lab primer outcomes."""

import os
import sys
import click
import numpy as np
import pandas as pd

from primerforge.ml_scorer import MLScorer


def format_markdown_comparison(results: dict) -> str:
    """Formats a beautiful Markdown table comparing before/after metrics without external dependencies."""
    headers = ["Metric", "Before Fine-Tuning", "After Fine-Tuning", "Absolute Improvement"]
    separator = ["---", "---", "---", "---"]
    
    brier_diff = results["Brier_Before"] - results["Brier_After"]
    ece_diff = results["ECE_Before"] - results["ECE_After"]
    
    rows = [
        [
            "Brier Score (MSE)", 
            f"{results['Brier_Before']:.4f}", 
            f"{results['Brier_After']:.4f}", 
            f"{brier_diff:+.4f} (Error Reduction)" if brier_diff >= 0 else f"{brier_diff:+.4f} (Regression)"
        ],
        [
            "Expected Calibration Error (ECE)", 
            f"{results['ECE_Before']:.4f}", 
            f"{results['ECE_After']:.4f}", 
            f"{ece_diff:+.4f} (Calibration Gain)" if ece_diff >= 0 else f"{ece_diff:+.4f} (Regression)"
        ]
    ]
    
    table = []
    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(separator) + " |")
    for r in rows:
        table.append("| " + " | ".join(r) + " |")
    return "\n".join(table)


@click.command()
@click.option(
    "--csv",
    "-c",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to the user's custom lab PCR outcomes CSV file.",
)
@click.option(
    "--out-dir",
    "-o",
    type=str,
    default="models/fine_tuned",
    help="Directory directory to export the fine-tuned boosters and calibration parameters.",
)
def main(csv: str, out_dir: str) -> None:
    """Ingests custom qPCR/endpoint PCR data, performs regularized fine-tuning, and exports the model."""
    click.echo("================================================================================")
    click.echo("             PRIMERFORGE USER WET-LAB ENSEMBLE FINE-TUNING UTILITY")
    click.echo("================================================================================")
    
    click.echo(f"Ingesting lab outcomes dataset from: {csv}...")
    try:
        df = pd.read_csv(csv)
    except Exception as e:
        click.echo(f"Error reading CSV: {e}")
        sys.exit(1)

    # 1. Normalize Column Headers case-insensitively
    cols = {c.lower(): c for c in df.columns}
    
    # Resolve sequence headers
    f_header = None
    for key in ["forward_seq", "forward", "fwd", "primer_f", "fwd_seq", "sequence_f"]:
        if key in cols:
            f_header = cols[key]
            break
            
    r_header = None
    for key in ["reverse_seq", "reverse", "rev", "primer_r", "rev_seq", "sequence_r"]:
        if key in cols:
            r_header = cols[key]
            break

    if not f_header or not r_header:
        click.echo("Error: Could not resolve forward and reverse sequence columns in your CSV.")
        click.echo("Please make sure your CSV contains columns like 'forward_seq' and 'reverse_seq'.")
        sys.exit(1)
        
    df = df.rename(columns={f_header: "forward_seq", r_header: "reverse_seq"})

    # Resolve outcome targets
    success_header = None
    for key in ["success", "success_idx", "outcome", "class", "label", "result"]:
        if key in cols:
            success_header = cols[key]
            break
            
    ct_header = None
    for key in ["ct", "ct_value", "cycle_threshold", "threshold_cycle"]:
        if key in cols:
            ct_header = cols[key]
            break

    efficiency_header = None
    for key in ["efficiency", "amp_efficiency", "pcr_efficiency"]:
        if key in cols:
            efficiency_header = cols[key]
            break

    # Outcome Mapping Logic
    if success_header:
        # Standard label normalization
        df["success"] = df[success_header].apply(
            lambda x: 0.95 if str(x).lower() in ["pass", "positive", "1", "1.0", "true"] 
            else (0.05 if str(x).lower() in ["fail", "negative", "0", "0.0", "false"] 
            else float(x))
        )
    elif ct_header:
        # Ct-based success mapping: y = max(0.01, min(0.99, (35.0 - Ct) / 15.0))
        click.echo("Success outcome column missing. Automatically mapping Ct values to success index...")
        df["success"] = df[ct_header].apply(lambda x: float(max(0.01, min(0.99, (35.0 - float(x)) / 15.0))) if pd.notna(x) else 0.05)
    elif efficiency_header:
        # Efficiency-based success mapping
        click.echo("Success outcome column missing. Automatically mapping PCR amplification efficiency...")
        df["success"] = df[efficiency_header].apply(lambda x: float(np.clip(float(x), 0.0, 1.0) if pd.notna(x) else 0.05))
    else:
        # Impute positive default success rate if missing
        click.echo("No outcomes detected. Assuming all provided sequences designed in your lab are successful (success = 0.95)...")
        df["success"] = 0.95

    # 2. Check Dataset Size Constraint
    if len(df) < 5:
        click.echo("Error: A minimum of N=5 primer pairs is required for transfer learning validation.")
        sys.exit(1)

    click.echo(f"Ingested N={len(df)} standardized PCR outcomes. Initializing MLScorer...")
    scorer = MLScorer()

    click.echo("Performing biophysically regularized GBDT refitting + EWC MLP sequence fine-tuning...")
    try:
        results = scorer.fine_tune_on_user_data(df, out_dir)
    except Exception as e:
        click.echo(f"Fine-tuning failed during execution: {e}")
        sys.exit(1)

    click.echo("\n" + "=" * 80)
    click.echo("               PRIMERFORGE TRANS-LEARNING BENCHMARK COMPARISON REPORT")
    click.echo("=" * 80)
    click.echo(format_markdown_comparison(results))
    click.echo("=" * 80)
    
    click.echo(f"\nFine-tuned models successfully exported for permanent use to: {out_dir}")
    click.echo("To design assays with this custom model, run the CLI with --model-dir options:")
    click.echo(f"  poetry run primerforge design -t locus.fasta --model-dir {out_dir}\n")
    click.echo("================================================================================")


if __name__ == "__main__":
    main()
