#!/usr/bin/env python3
"""
generate_negatives.py

Generates realistic computational negative controls (label=0) for the
PrimerForge training dataset by introducing thermodynamic failure modes.
"""

import os
import re
import sys
import random
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Import pipeline helpers
from data_collection_checklist import (
    validate_primer_entry,
    calculate_santa_lucia_tm,
    get_reverse_complement
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

FINAL_COLUMNS = MASTER_COLUMNS + ['near_duplicate_flag', 'near_duplicate_partner_id']

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def build_synth_row(
    p_id: str,
    gene_name: str,
    fwd_seq: str,
    rev_seq: str,
    notes: str,
    amp_size: Any
) -> Dict[str, Any]:
    """Assembles a standard row structure for a synthetic negative."""
    row = {
        # Identity
        'primer_id': p_id,
        'gene_name': gene_name,
        'organism': 'human',
        'source_db': 'synthetic',
        'paper_doi': 'N/A',
        # Sequences
        'sequence_fwd': fwd_seq,
        'sequence_rev': rev_seq,
        # Labels
        'label': 0,
        'label_confidence': 'medium',
        'label_source': 'synthetic',
        'label_notes': notes,
        # Experimental Metadata
        'cell_line_tested': "",
        'annealing_tm_used_C': np.nan,
        'amplicon_size_bp': int(amp_size) if pd.notna(amp_size) else np.nan,
        'qpcr_efficiency': np.nan,
        'qpcr_r2': np.nan,
        'gel_band_confirmed': False,
        'date_collected': datetime.date.today().isoformat(),
        # Flags
        'is_synthetic_negative': True,
        'is_prospective_validation': False,
        'is_held_out_test': False,
        'near_duplicate_flag': False,
        'near_duplicate_partner_id': ""
    }

    # Recompute biophysical features using the platform's curator module
    try:
        curator = DataCurationPipeline()
        features = curator._compute_biophysical_features(fwd_seq, rev_seq)
        for k in ['f_tm', 'r_tm', 'tm_diff', 'f_gc', 'r_gc', 'f_hairpin_dg', 'r_hairpin_dg',
                  'cross_dimer_dg', 'f_len', 'r_len', 'f_clamp_gc', 'r_clamp_gc',
                  'f_poly_run', 'r_poly_run', 'target_gc', 'target_len']:
            row[k] = features.get(k, np.nan)
    except Exception:
        # Fallback to NaN
        for k in ['f_tm', 'r_tm', 'tm_diff', 'f_gc', 'r_gc', 'f_hairpin_dg', 'r_hairpin_dg',
                  'cross_dimer_dg', 'f_len', 'r_len', 'f_clamp_gc', 'r_clamp_gc',
                  'f_poly_run', 'r_poly_run', 'target_gc', 'target_len']:
            row[k] = np.nan

    return row

def validate_synth(row: Dict[str, Any], existing_ids: List[str], existing_seqs: set) -> bool:
    """Runs Checklist validation allowing specific Tier 2 checks to fail."""
    # Prevent exact sequence duplication
    fwd = str(row['sequence_fwd']).upper().strip()
    rev = str(row['sequence_rev']).upper().strip()
    seq_key = f"{fwd}_{rev}"
    if seq_key in existing_seqs:
        return False

    is_valid, failed_checks = validate_primer_entry(row, existing_ids)
    
    # Exclude biophysical and Tm checks intentionally violated
    allowed_failures = {"CHECK_5_EXTREME_GC", "CHECK_6_EXTREME_TM", "CHECK_8_HOMOPOLYMER_RUN"}
    real_failures = [c for c in failed_checks if c not in allowed_failures]
    
    return len(real_failures) == 0

# ---------------------------------------------------------------------------
# Mutators for Negative Failure Modes
# ---------------------------------------------------------------------------
def generate_type_a(df_pos: pd.DataFrame, target_count: int, existing_ids: List[str], existing_seqs: set) -> List[Dict[str, Any]]:
    """Type A: High Tm mismatch (> 8°C). Mutates the reverse primer if needed to establish mismatch."""
    negatives = []
    pos_records = df_pos.to_dict('records')
    
    for r in pos_records:
        r['f_tm_calc'] = calculate_santa_lucia_tm(r['sequence_fwd'])
        r['r_tm_calc'] = calculate_santa_lucia_tm(r['sequence_rev'])
        
    attempts = 0
    while len(negatives) < target_count and attempts < 20000:
        attempts += 1
        r1 = random.choice(pos_records)
        r2 = random.choice(pos_records)
        
        if r1['gene_name'] == r2['gene_name']:
            continue
            
        tm_fwd = r1['f_tm_calc']
        tm_rev = r2['r_tm_calc']
        r_seq = r2['sequence_rev']
        
        diff = abs(tm_fwd - tm_rev)
        if diff <= 8.0:
            seq_list = list(r_seq)
            gc_indices = [idx for idx, base in enumerate(r_seq) if base in "GC"]
            if len(gc_indices) >= 4:
                mutate_indices = random.sample(gc_indices, min(len(gc_indices), 6))
                for idx in mutate_indices:
                    seq_list[idx] = random.choice(["A", "T"])
                r_seq_mutated = "".join(seq_list)
                tm_rev_mutated = calculate_santa_lucia_tm(r_seq_mutated)
                if abs(tm_fwd - tm_rev_mutated) > 8.0:
                    r_seq = r_seq_mutated
                    tm_rev = tm_rev_mutated
                else:
                    continue
            else:
                continue
                
        if abs(tm_fwd - tm_rev) > 8.0:
            new_id = f"SYNTH_TMMIS_{len(negatives) + 1:03d}"
            row = build_synth_row(
                new_id, 
                f"{r1['gene_name']}_{r2['gene_name']}", 
                r1['sequence_fwd'], 
                r_seq, 
                "synthetic_neg_tm_mismatch", 
                r1.get('amplicon_size_bp')
            )
            if validate_synth(row, existing_ids, existing_seqs):
                existing_ids.append(new_id)
                existing_seqs.add(f"{r1['sequence_fwd'].upper().strip()}_{r_seq.upper().strip()}")
                negatives.append(row)
                
    return negatives

def generate_type_b(df_pos: pd.DataFrame, target_count: int, existing_ids: List[str], existing_seqs: set) -> List[Dict[str, Any]]:
    """Type B: 3' End complementarity / dimer formation."""
    negatives = []
    pos_records = df_pos.to_dict('records')
    
    attempts = 0
    while len(negatives) < target_count and attempts < 20000:
        attempts += 1
        r = random.choice(pos_records)
        f_seq = r['sequence_fwd']
        r_seq = r['sequence_rev']
        
        f_3_end = f_seq[-6:]
        rev_comp = get_reverse_complement(f_3_end)
        new_r_seq = r_seq[:-6] + rev_comp
        
        new_id = f"SYNTH_DIMER_{len(negatives) + 1:03d}"
        row = build_synth_row(
            new_id, 
            r['gene_name'], 
            f_seq, 
            new_r_seq, 
            "synthetic_neg_3prime_dimer", 
            r.get('amplicon_size_bp')
        )
        if validate_synth(row, existing_ids, existing_seqs):
            existing_ids.append(new_id)
            existing_seqs.add(f"{f_seq.upper().strip()}_{new_r_seq.upper().strip()}")
            negatives.append(row)
            
    return negatives

def generate_type_c(df_pos: pd.DataFrame, target_count: int, existing_ids: List[str], existing_seqs: set) -> List[Dict[str, Any]]:
    """Type C: GC clamp failure (end mutated to AAA or TTT)."""
    negatives = []
    pos_records = df_pos.to_dict('records')
    
    attempts = 0
    while len(negatives) < target_count and attempts < 20000:
        attempts += 1
        r = random.choice(pos_records)
        f_seq = r['sequence_fwd']
        r_seq = r['sequence_rev']
        
        new_f_seq = f_seq[:-3] + random.choice(["AAA", "TTT"])
        new_r_seq = r_seq[:-3] + random.choice(["AAA", "TTT"])
        
        new_id = f"SYNTH_CLAMP_{len(negatives) + 1:03d}"
        row = build_synth_row(
            new_id, 
            r['gene_name'], 
            new_f_seq, 
            new_r_seq, 
            "synthetic_neg_gc_clamp_failure", 
            r.get('amplicon_size_bp')
        )
        if validate_synth(row, existing_ids, existing_seqs):
            existing_ids.append(new_id)
            existing_seqs.add(f"{new_f_seq.upper().strip()}_{new_r_seq.upper().strip()}")
            negatives.append(row)
            
    return negatives

def generate_type_d(df_pos: pd.DataFrame, target_count: int, existing_ids: List[str], existing_seqs: set) -> List[Dict[str, Any]]:
    """Type D: Poly-run insertion."""
    negatives = []
    pos_records = df_pos.to_dict('records')
    
    attempts = 0
    while len(negatives) < target_count and attempts < 20000:
        attempts += 1
        r = random.choice(pos_records)
        f_seq = r['sequence_fwd']
        r_seq = r['sequence_rev']
        
        if random.choice([True, False]):
            mid = len(f_seq) // 2
            new_f_seq = f_seq[:mid] + random.choice(["AAAAA", "GGGGG"]) + f_seq[mid:]
            new_r_seq = r_seq
        else:
            mid = len(r_seq) // 2
            new_f_seq = f_seq
            new_r_seq = r_seq[:mid] + random.choice(["AAAAA", "GGGGG"]) + r_seq[mid:]
            
        new_id = f"SYNTH_POLY_{len(negatives) + 1:03d}"
        row = build_synth_row(
            new_id, 
            r['gene_name'], 
            new_f_seq, 
            new_r_seq, 
            "synthetic_neg_poly_run", 
            r.get('amplicon_size_bp')
        )
        if validate_synth(row, existing_ids, existing_seqs):
            existing_ids.append(new_id)
            existing_seqs.add(f"{new_f_seq.upper().strip()}_{new_r_seq.upper().strip()}")
            negatives.append(row)
            
    return negatives

def generate_type_e(df_pos: pd.DataFrame, target_count: int, existing_ids: List[str], existing_seqs: set) -> List[Dict[str, Any]]:
    """Type E: Low GC (< 30%)."""
    negatives = []
    pos_records = df_pos.to_dict('records')
    
    def mutate_gc_to_at(seq: str) -> str:
        seq_list = list(seq)
        gc_indices = [idx for idx, base in enumerate(seq) if base in "GC"]
        if not gc_indices:
            return seq
        num_mutate = int(len(gc_indices) * 0.40)
        if num_mutate == 0:
            num_mutate = 1
        mutate_indices = random.sample(gc_indices, num_mutate)
        for idx in mutate_indices:
            seq_list[idx] = random.choice(["A", "T"])
        return "".join(seq_list)
        
    attempts = 0
    while len(negatives) < target_count and attempts < 20000:
        attempts += 1
        r = random.choice(pos_records)
        f_seq = r['sequence_fwd']
        r_seq = r['sequence_rev']
        
        new_f_seq = mutate_gc_to_at(f_seq)
        new_r_seq = mutate_gc_to_at(r_seq)
        
        new_id = f"SYNTH_LOWGC_{len(negatives) + 1:03d}"
        row = build_synth_row(
            new_id, 
            r['gene_name'], 
            new_f_seq, 
            new_r_seq, 
            "synthetic_neg_low_gc", 
            r.get('amplicon_size_bp')
        )
        if validate_synth(row, existing_ids, existing_seqs):
            existing_ids.append(new_id)
            existing_seqs.add(f"{new_f_seq.upper().strip()}_{new_r_seq.upper().strip()}")
            negatives.append(row)
            
    return negatives

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    db_path = "data/master_training_db_v2.csv"
    if not os.path.exists(db_path):
        print(f"Error: Master training database not found at: {db_path}", file=sys.stderr)
        sys.exit(1)
        
    df_db = pd.read_csv(db_path)
    
    # Remove any existing synthetic records to keep execution idempotent
    df_db = df_db[df_db['source_db'].str.lower() != 'synthetic'].copy()
    total_merged_size = len(df_db)
    print(f"Loaded Clean Master Database. Size: {total_merged_size} rows.")
    
    # Filter to label=1 (positives) only
    df_pos = df_db[df_db['label'] == 1]
    print(f"Found N={len(df_pos)} functional primer pairs (label=1).")
    
    if len(df_pos) == 0:
        print("Error: No functional primers to generate negatives from. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    # Check 35% cap
    target_per_type = 90
    total_target = target_per_type * 5
    max_negatives_allowed = int(0.35 * total_merged_size)
    
    if total_target > max_negatives_allowed:
        target_per_type = max_negatives_allowed // 5
        total_target = target_per_type * 5
        print(f"Warning: Target negatives ({total_target}) exceeds 35% cap ({max_negatives_allowed}).")
        print(f"Scaling down to {target_per_type} pairs per type.")
        
    if target_per_type == 0:
        print("Error: Target negative count scaled to 0. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Initialize tracking structures to prevent ID and sequence duplication
    existing_ids = list(df_db['primer_id'].unique())
    
    # Initialize set of all sequence keys in the database to prevent duplicate sequences
    db_seq_keys = (df_db['sequence_fwd'].astype(str).str.upper().str.strip() + "_" + 
                   df_db['sequence_rev'].astype(str).str.upper().str.strip())
    existing_seqs = set(db_seq_keys.unique())
    
    all_negatives = []
    random.seed(42)
    
    print(f"Generating synthetic negatives (target={target_per_type} of each type)...")
    
    # Type A
    print("Generating Type A (High Tm mismatch)...")
    neg_a = generate_type_a(df_pos, target_per_type, existing_ids, existing_seqs)
    all_negatives.extend(neg_a)
    print(f"  - Generated {len(neg_a)} Type A negatives.")
    
    # Type B
    print("Generating Type B (3' Self-complementarity)...")
    neg_b = generate_type_b(df_pos, target_per_type, existing_ids, existing_seqs)
    all_negatives.extend(neg_b)
    print(f"  - Generated {len(neg_b)} Type B negatives.")
    
    # Type C
    print("Generating Type C (GC clamp failure)...")
    neg_c = generate_type_c(df_pos, target_per_type, existing_ids, existing_seqs)
    all_negatives.extend(neg_c)
    print(f"  - Generated {len(neg_c)} Type C negatives.")
    
    # Type D
    print("Generating Type D (Poly-run failure)...")
    neg_d = generate_type_d(df_pos, target_per_type, existing_ids, existing_seqs)
    all_negatives.extend(neg_d)
    print(f"  - Generated {len(neg_d)} Type D negatives.")
    
    # Type E
    print("Generating Type E (Extremely low GC)...")
    neg_e = generate_type_e(df_pos, target_per_type, existing_ids, existing_seqs)
    all_negatives.extend(neg_e)
    print(f"  - Generated {len(neg_e)} Type E negatives.")
    
    print(f"Total synthetic negatives generated: {len(all_negatives)}")
    
    if not all_negatives:
        print("Warning: No negatives were generated. Database remains unchanged.", file=sys.stderr)
        sys.exit(0)
        
    df_neg = pd.DataFrame(all_negatives)
    
    # Reindex columns to match FINAL_COLUMNS
    df_neg = df_neg.reindex(columns=FINAL_COLUMNS)
    
    # Append to database
    df_final = pd.concat([df_db, df_neg], ignore_index=True)
    df_final.to_csv(db_path, index=False)
    print(f"\nSuccessfully appended synthetic negatives and saved to: {db_path}")
    
    # Print per-type summary and final distribution
    print("\n" + "="*50)
    print("SYNTHETIC NEGATIVE GENERATION SUMMARY")
    print("="*50)
    print("1. Synthetic Negatives Appended by Type:")
    print(f"   - Type A (Tm mismatch):           {len(neg_a)}")
    print(f"   - Type B (3' complementarity):     {len(neg_b)}")
    print(f"   - Type C (GC clamp failure):      {len(neg_c)}")
    print(f"   - Type D (Poly-run insertion):     {len(neg_d)}")
    print(f"   - Type E (Extremely low GC):      {len(neg_e)}")
    print(f"   - Total Appended:                 {len(all_negatives)}")
    
    labels_final = df_final['label'].value_counts()
    print("\n2. Final Database Label Distribution:")
    print(f"   - Label 1 (Functional):     {labels_final.get(1, 0)}")
    print(f"   - Label 0 (Non-Functional): {labels_final.get(0, 0)}")
    print(f"   - Total Database Rows:      {len(df_final)}")
    print(f"   - Negatives Proportion:     {labels_final.get(0, 0)/len(df_final)*100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
