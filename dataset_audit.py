#!/usr/bin/env python3
"""
dataset_audit.py

Analyzes the unified Master Training Database v2, populating missing biophysical
thermo features if needed, and generates a comprehensive, publication-ready
statistics report.
"""

import os
import re
import sys
import time
import datetime
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Import pipeline helpers
from data_collection_checklist import (
    validate_primer_entry,
    calculate_santa_lucia_tm
)
from primerforge.data_curation import DataCurationPipeline

# Master columns definition
MASTER_COLUMNS = [
    'primer_id', 'gene_name', 'organism', 'source_db', 'paper_doi',
    'sequence_fwd', 'sequence_rev', 'label', 'label_confidence', 'label_source',
    'label_notes', 'cell_line_tested', 'annealing_tm_used_C', 'amplicon_size_bp',
    'qpcr_efficiency', 'qpcr_r2', 'gel_band_confirmed', 'date_collected',
    'f_tm', 'r_tm', 'tm_diff', 'f_gc', 'r_gc', 'f_hairpin_dg', 'r_hairpin_dg',
    'cross_dimer_dg', 'f_len', 'r_len', 'f_clamp_gc', 'r_clamp_gc',
    'f_poly_run', 'r_poly_run', 'target_gc', 'target_len',
    'is_synthetic_negative', 'is_prospective_validation', 'is_held_out_test'
]

# ---------------------------------------------------------------------------
# Populate Biophysical Features if Missing
# ---------------------------------------------------------------------------
def populate_missing_features(df: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """Calculates biophysical features for any row with NaN values in f_tm."""
    missing_mask = df['f_tm'].isna()
    missing_count = missing_mask.sum()
    if missing_count > 0:
        print(f"Detecting {missing_count} rows with missing biophysical features.", flush=True)
        print("Populating biophysical features via BiophysicsEngine...", flush=True)
        curator = DataCurationPipeline()
        records = df.to_dict('records')
        
        for idx, r in enumerate(records):
            if pd.isna(r['f_tm']):
                fwd = str(r['sequence_fwd']).upper().strip()
                rev = str(r['sequence_rev']).upper().strip()
                try:
                    features = curator._compute_biophysical_features(fwd, rev)
                    for k in ['f_tm', 'r_tm', 'tm_diff', 'f_gc', 'r_gc', 'f_hairpin_dg', 'r_hairpin_dg',
                              'cross_dimer_dg', 'f_len', 'r_len', 'f_clamp_gc', 'r_clamp_gc',
                              'f_poly_run', 'r_poly_run', 'target_gc', 'target_len']:
                        r[k] = features.get(k, np.nan)
                except Exception as e:
                    pass
            if (idx + 1) % 500 == 0 or idx + 1 == len(records):
                print(f"  Processed {idx + 1}/{len(records)} rows...", flush=True)
                
        df = pd.DataFrame(records)
        df.to_csv(db_path, index=False)
        print("Saved updated master training database.", flush=True)
    return df

# ---------------------------------------------------------------------------
# Pure-Python CD-HIT Clustering Preview (Optimized via K-mer Index & Pruning)
# ---------------------------------------------------------------------------
def run_cd_hit_clustering(df: pd.DataFrame, threshold: float = 0.80, k: int = 4) -> List[Dict[str, Any]]:
    """Greedy sequence clustering at 80% similarity using dynamic k-mer pruning."""
    records = df.to_dict('records')
    n = len(records)
    
    print("Pre-calculating base counts and k-mers...", flush=True)
    for r in records:
        seq = (str(r['sequence_fwd']) + str(r['sequence_rev'])).upper()
        r['combined_seq'] = seq
        r['len_combined'] = len(seq)
        
        c_a = seq.count('A')
        c_c = seq.count('C')
        c_g = seq.count('G')
        c_t = seq.count('T')
        c_other = len(seq) - (c_a + c_c + c_g + c_t)
        r['counts_tuple'] = (c_a, c_c, c_g, c_t, c_other)
        r['kmers'] = set(seq[i:i+k] for i in range(len(seq) - k + 1))

    # Sort indices by length descending
    sorted_indices = sorted(range(n), key=lambda idx: records[idx]['len_combined'], reverse=True)
    
    clusters = []
    kmer_to_cluster_ids = {}
    
    print("Running CD-HIT style clustering preview (k-mer index)...", flush=True)
    
    for count, idx in enumerate(sorted_indices):
        r = records[idx]
        seq = r['combined_seq']
        length = r['len_combined']
        c_tuple = r['counts_tuple']
        r_kmers = r['kmers']
        
        # Retrieve candidate cluster IDs sharing at least one k-mer
        candidate_cluster_ids = set()
        for kmer in r_kmers:
            if kmer in kmer_to_cluster_ids:
                candidate_cluster_ids.update(kmer_to_cluster_ids[kmer])
                
        found = False
        matcher = SequenceMatcher(None, seq, "")
        
        for c_id in sorted(candidate_cluster_ids):
            c = clusters[c_id]
            rep_r = records[c['rep_idx']]
            rep_len = rep_r['len_combined']
            
            # 1. Prune if length difference > 20%
            if abs(length - rep_len) > (1.0 - threshold) * rep_len:
                continue
                
            # 2. Prune using dynamic minimum shared k-mers
            L_min = min(length, rep_len)
            E_max = int((1.0 - threshold) * L_min)
            min_shared = L_min - k + 1 - E_max * k
            
            if min_shared > 0:
                shared_count = len(r_kmers.intersection(rep_r['kmers']))
                if shared_count < min_shared:
                    continue
                    
            # 3. Prune using base count L1 distance (diff)
            rep_c = rep_r['counts_tuple']
            diff = (
                abs(c_tuple[0] - rep_c[0]) +
                abs(c_tuple[1] - rep_c[1]) +
                abs(c_tuple[2] - rep_c[2]) +
                abs(c_tuple[3] - rep_c[3]) +
                abs(c_tuple[4] - rep_c[4])
            )
            
            if diff > (1.0 - threshold) * (length + rep_len):
                continue
                
            # 4. SequenceMatcher similarity
            matcher.set_seq2(rep_r['combined_seq'])
            if matcher.ratio() >= threshold:
                c['members'].append(idx)
                found = True
                break
                
        if not found:
            c_id = len(clusters)
            clusters.append({'rep_idx': idx, 'members': [idx]})
            for kmer in r_kmers:
                if kmer not in kmer_to_cluster_ids:
                    kmer_to_cluster_ids[kmer] = []
                kmer_to_cluster_ids[kmer].append(c_id)
                
        if (count + 1) % 1000 == 0 or count + 1 == n:
            print(f"  Clustered {count + 1}/{n} sequences...", flush=True)
            
    return clusters

# ---------------------------------------------------------------------------
# Main Audit Routine
# ---------------------------------------------------------------------------
def main():
    db_path = "data/master_training_db_v2.csv"
    if not os.path.exists(db_path):
        print(f"Error: Master training database not found at: {db_path}", file=sys.stderr)
        sys.exit(1)
        
    df = pd.read_csv(db_path)
    
    # Ensure biophysical columns are complete
    df = populate_missing_features(df, db_path)
    
    print("\nAuditing dataset statistics...", flush=True)
    
    # ---------------------------------------------------------------------------
    # Set Matplotlib Styles (Okabe-Ito Color Palette)
    # ---------------------------------------------------------------------------
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    COLOR_BLUE = '#0072B2'
    COLOR_SKY = '#56B4E9'
    COLOR_GREEN = '#009E73'
    COLOR_ORANGE = '#E69F00'
    
    os.makedirs('plots', exist_ok=True)
    
    # ---------------------------------------------------------------------------
    # Section 1: Overview Calculations
    # ---------------------------------------------------------------------------
    total_pairs = len(df)
    
    # Normalize source names
    source_counts = df['source_db'].astype(str).str.lower().value_counts()
    
    labels = df['label'].value_counts()
    label_1_count = labels.get(1, 0)
    label_1_pct = label_1_count / total_pairs * 100.0
    label_0_count = labels.get(0, 0)
    label_0_pct = label_0_count / total_pairs * 100.0
    
    confidences = df['label_confidence'].astype(str).str.lower().value_counts()
    
    synth_neg_count = df['is_synthetic_negative'].sum()
    synth_neg_pct = synth_neg_count / total_pairs * 100.0
    
    # ---------------------------------------------------------------------------
    # Section 2: Sequence Statistics & Histograms
    # ---------------------------------------------------------------------------
    f_len = df['f_len']
    r_len = df['r_len']
    
    gc_vals = pd.concat([df['f_gc'], df['r_gc']])
    tm_vals = pd.concat([df['f_tm'], df['r_tm']])
    
    amp_sizes = df['amplicon_size_bp'].dropna()
    
    # Helper to style and save plots
    def style_and_save_plot(fig, ax, title, xlabel, ylabel, filename):
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12, color='#333333')
        ax.set_xlabel(xlabel, fontsize=9, fontweight='bold', color='#333333')
        ax.set_ylabel(ylabel, fontsize=9, fontweight='bold', color='#333333')
        ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cccccc')
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#cccccc')
            ax.spines[spine].set_linewidth(0.5)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, facecolor='white')
        plt.close()

    # Plot GC content
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.hist(gc_vals, bins=25, color=COLOR_BLUE, edgecolor='#1e293b', linewidth=0.5, alpha=0.85)
    style_and_save_plot(fig, ax, 'GC Content Distribution', 'GC (%)', 'Count', 'plots/gc_distribution.png')
    
    # Plot Tm
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.hist(tm_vals, bins=25, color=COLOR_GREEN, edgecolor='#1e293b', linewidth=0.5, alpha=0.85)
    style_and_save_plot(fig, ax, 'Melting Temperature (Tm) Distribution', 'Tm (°C)', 'Count', 'plots/tm_distribution.png')
    
    # Plot Amplicon size
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.hist(amp_sizes, bins=25, color=COLOR_ORANGE, edgecolor='#1e293b', linewidth=0.5, alpha=0.85)
    style_and_save_plot(fig, ax, 'Amplicon Size Distribution', 'Size (bp)', 'Count', 'plots/amplicon_size_distribution.png')
    
    # ---------------------------------------------------------------------------
    # Section 3: Label Quality Assessment
    # ---------------------------------------------------------------------------
    # Mean efficiency for label=1
    label_1_eff = df[(df['label'] == 1) & (df['qpcr_efficiency'].notna())]['qpcr_efficiency']
    mean_eff = label_1_eff.mean()
    std_eff = label_1_eff.std()
    
    # Cross-tabulation
    crosstab_df = pd.crosstab(df['source_db'], df['label_confidence'])
    
    # Near-duplicates
    near_dups_count = df['near_duplicate_flag'].sum()
    
    # Completeness of biophysical features
    biophys_cols = [
        'f_tm', 'r_tm', 'tm_diff', 'f_gc', 'r_gc', 'f_hairpin_dg', 'r_hairpin_dg',
        'cross_dimer_dg', 'f_len', 'r_len', 'f_clamp_gc', 'r_clamp_gc',
        'f_poly_run', 'r_poly_run', 'target_gc', 'target_len'
    ]
    completeness = {}
    for col in biophys_cols:
        completeness[col] = df[col].notna().mean() * 100.0
        
    # ---------------------------------------------------------------------------
    # Section 4: Diversity
    # ---------------------------------------------------------------------------
    unique_genes = df['gene_name'].nunique()
    org_counts = df['organism'].astype(str).str.lower().value_counts()
    
    # Plot Gene Diversity Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    gene_counts = df['gene_name'].value_counts().head(20)
    gene_counts.plot(kind='bar', color=COLOR_BLUE, edgecolor='#1e293b', linewidth=0.5, alpha=0.85, ax=ax)
    plt.xticks(rotation=45, ha='right')
    style_and_save_plot(fig, ax, 'Top 20 Genes by Primer Pair Count', 'Gene', 'Count', 'plots/gene_diversity.png')
    
    # ---------------------------------------------------------------------------
    # Section 5: Readiness Checklist
    # ---------------------------------------------------------------------------
    check_1_size = total_pairs >= 800
    check_2_neg = label_0_count >= 200
    check_3_neg_ratio = (label_0_count / total_pairs) <= 0.40
    
    # Percent complete (all 16 features non-NaN)
    all_complete_pct = df[biophys_cols].notna().all(axis=1).mean() * 100.0
    check_4_biophys = all_complete_pct >= 90.0
    
    # Duplicates check
    seq_pairs = df['sequence_fwd'].astype(str).str.upper() + "_" + df['sequence_rev'].astype(str).str.upper()
    check_5_dups = not seq_pairs.duplicated().any()
    
    check_6_org = df['organism'].nunique() >= 3
    
    check_7_prov = ((df['paper_doi'].notna() & (df['paper_doi'] != "")) | 
                    (df['source_db'].notna() & (df['source_db'] != ""))).all()
                    
    checklist = {
        "Minimum 800 total pairs (for Bioinformatics submission minimum)": check_1_size,
        "At least 200 label=0 entries (sufficient negatives)": check_2_neg,
        "label=0 entries ≤ 40% of total (not over-represented)": check_3_neg_ratio,
        "All biophysical features computed for ≥ 90% of rows": check_4_biophys,
        "No exact duplicate sequences": check_5_dups,
        "At least 3 different organisms represented": check_6_org,
        "All rows have paper_doi or source_db populated": check_7_prov
    }
    
    # ---------------------------------------------------------------------------
    # Section 6: CD-HIT Clustering
    # ---------------------------------------------------------------------------
    clusters = run_cd_hit_clustering(df, 0.80)
    num_clusters = len(clusters)
    cluster_sizes = [len(c['members']) for c in clusters]
    avg_cluster_size = np.mean(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)
    singletons_count = sum(1 for size in cluster_sizes if size == 1)
    
    leakage_warnings = []
    for c in clusters:
        if len(c['members']) > 15:
            rep_r = df.iloc[c['rep_idx']]
            leakage_warnings.append(
                f"Cluster represented by ID {rep_r['primer_id']} (Gene: {rep_r['gene_name']}) contains {len(c['members'])} similar entries."
            )
            
    # ---------------------------------------------------------------------------
    # Generate Markdown Report File
    # ---------------------------------------------------------------------------
    report_md = []
    report_md.append("# PrimerForge Training Dataset Audit Report\n")
    report_md.append(f"**Date Generated**: {datetime.date.today().isoformat()}\n")
    report_md.append("This document contains the publication-ready data validation and quality control audit for `data/master_training_db_v2.csv`.\n")
    report_md.append("---")
    
    # Section 1 - Overview Table
    report_md.append("## Section 1 — Dataset Overview\n")
    report_md.append("| Parameter | Category | Count | Percentage |")
    report_md.append("| :--- | :--- | :---: | :---: |")
    report_md.append(f"| **Total Primer Pairs** | All | {total_pairs:,} | 100.00% |")
    report_md.append("| **Source DB** | rtprimerdb | {} | {:.2f}% |".format(source_counts.get('rtprimerdb', 0), source_counts.get('rtprimerdb', 0) / total_pairs * 100.0))
    report_md.append("| | artic | {} | {:.2f}% |".format(source_counts.get('artic', 0), source_counts.get('artic', 0) / total_pairs * 100.0))
    report_md.append("| | primerbank | {} | {:.2f}% |".format(source_counts.get('primerbank', 0), source_counts.get('primerbank', 0) / total_pairs * 100.0))
    report_md.append("| | pmc | {} | {:.2f}% |".format(source_counts.get('pmc', 0), source_counts.get('pmc', 0) / total_pairs * 100.0))
    report_md.append("| | synthetic | {} | {:.2f}% |".format(source_counts.get('synthetic', 0), source_counts.get('synthetic', 0) / total_pairs * 100.0))
    report_md.append(f"| **Label** | label=1 (Functional) | {label_1_count} | {label_1_pct:.2f}% |")
    report_md.append(f"| | label=0 (Non-Functional) | {label_0_count} | {label_0_pct:.2f}% |")
    report_md.append("| **Confidence** | High | {} | {:.2f}% |".format(confidences.get('high', 0), confidences.get('high', 0) / total_pairs * 100.0))
    report_md.append("| | Medium | {} | {:.2f}% |".format(confidences.get('medium', 0), confidences.get('medium', 0) / total_pairs * 100.0))
    report_md.append("| | Low | {} | {:.2f}% |".format(confidences.get('low', 0), confidences.get('low', 0) / total_pairs * 100.0))
    report_md.append(f"| **Negative Type** | Synthetic Negatives | {synth_neg_count} | {synth_neg_pct:.2f}% |\n")
    
    # Section 2
    report_md.append("## Section 2 — Sequence Statistics\n")
    report_md.append("| Metric | Mean ± Std | Min | Max |")
    report_md.append("| :--- | :---: | :---: | :---: |")
    report_md.append(f"| **Forward Length (nt)** | {f_len.mean():.2f} ± {f_len.std():.2f} | {f_len.min()} | {f_len.max()} |")
    report_md.append(f"| **Reverse Length (nt)** | {r_len.mean():.2f} ± {r_len.std():.2f} | {r_len.min()} | {r_len.max()} |")
    report_md.append(f"| **GC Content (%)** | {gc_vals.mean():.2f}% ± {gc_vals.std():.2f}% | {gc_vals.min():.1f}% | {gc_vals.max():.1f}% |")
    report_md.append(f"| **Melting Temperature (°C)** | {tm_vals.mean():.2f}°C ± {tm_vals.std():.2f}°C | {tm_vals.min():.1f}°C | {tm_vals.max():.1f}°C |")
    if len(amp_sizes) > 0:
        report_md.append(f"| **Amplicon Size (bp)** | {amp_sizes.mean():.2f} ± {amp_sizes.std():.2f} | {amp_sizes.min()} - {amp_sizes.max()} |")
    else:
        report_md.append("| **Amplicon Size (bp)** | N/A | N/A |")
    report_md.append("\n*Generated plots saved to:* `plots/gc_distribution.png`, `plots/tm_distribution.png`, and `plots/amplicon_size_distribution.png`.\n")
    
    # Section 3
    report_md.append("## Section 3 — Label Quality Assessment\n")
    if not pd.isna(mean_eff):
        report_md.append(f"- **qPCR Amplification Efficiency (label=1)**: {mean_eff:.3f} ± {std_eff:.3f} (N={len(label_1_eff)})")
    else:
        report_md.append("- **qPCR Amplification Efficiency (label=1)**: N/A (no efficiency data loaded)")
    report_md.append(f"- **Flagged Near-Duplicate Pairs**: {near_dups_count}")
    report_md.append("\n### Source Database × Label Confidence Crosstabulation")
    report_md.append(crosstab_df.to_markdown())
    report_md.append("\n### Biophysical Features Column Completeness")
    report_md.append("| Column Name | Non-NaN Completeness (%) |")
    report_md.append("| :--- | :---: |")
    for col, pct in completeness.items():
        report_md.append(f"| `{col}` | {pct:.1f}% |")
    report_md.append("")
    
    # Section 4
    report_md.append("## Section 4 — Organism & Gene Diversity\n")
    report_md.append(f"- **Unique Genes Represented**: {unique_genes}")
    report_md.append("- **Organism Breakdown**:")
    for org, count in org_counts.items():
        report_md.append(f"  - *{org}*: {count} pairs ({count / total_pairs * 100.0:.2f}%)")
    report_md.append("\n*Top 20 genes plot saved to:* `plots/gene_diversity.png`.\n")
    
    # Section 5
    report_md.append("## Section 5 — Dataset Readiness Checklist\n")
    all_checklist_passed = True
    for desc, passed in checklist.items():
        box = "[x]" if passed else "[ ]"
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_checklist_passed = False
        report_md.append(f"- {box} **{status}**: {desc}")
    report_md.append("")
    
    # Section 6
    report_md.append("## Section 6 — CD-HIT Clustering Preview (80% Threshold)\n")
    report_md.append(f"- **Total Clusters**: {num_clusters}")
    report_md.append(f"- **Average Cluster Size**: {avg_cluster_size:.2f} entries")
    report_md.append(f"- **Largest Cluster Size**: {max_cluster_size} entries")
    report_md.append(f"- **Singletons Count**: {singletons_count} clusters")
    
    if leakage_warnings:
        report_md.append("\n### :warning: Homology Leakage Warnings")
        for warning in leakage_warnings:
            report_md.append(f"- {warning}")
    else:
        report_md.append("\n*No homology leakage risk zones found (all clusters contain <= 15 entries).*")
    report_md.append("")
    
    # Final Verdict
    verdict_issues = []
    for desc, passed in checklist.items():
        if not passed:
            verdict_issues.append(desc)
            
    if all_checklist_passed:
        verdict = "DATASET READY FOR TRAINING"
    else:
        verdict = f"DATASET NEEDS ATTENTION: {'; '.join(verdict_issues)}"
        
    report_md.append("---")
    report_md.append(f"## Final Verdict\n**{verdict}**\n")
    
    # Write file with explicit UTF-8 encoding
    report_content = "\n".join(report_md)
    with open("data/dataset_audit_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    # Print the report to stdout replacing non-ASCII characters for terminal safety
    stdout_report = report_content\
        .replace("±", "+/-")\
        .replace("°", " deg ")\
        .replace("≤", "<=")\
        .replace("≥", ">=")\
        .replace("×", "x")
    print(stdout_report, flush=True)
    
    # Print the one-line final verdict as the last line
    stdout_verdict = verdict.replace("≤", "<=").replace("≥", ">=")
    print(stdout_verdict, flush=True)

if __name__ == "__main__":
    main()
