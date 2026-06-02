#!/usr/bin/env python3
"""
merge_and_deduplicate.py

Combines cleaned primer datasets, resolves exact duplicates/label conflicts,
applies non-human post-filters, flags near-duplicates with length-pruned
sequence matcher ratios, and generates final Master Database v2.
"""

import os
import re
import sys
import datetime
from difflib import SequenceMatcher
from collections import Counter
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# ---------------------------------------------------------------------------
# Master Columns Definition from master_schema.md
# ---------------------------------------------------------------------------
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

# Add near-duplicate flagging columns
FINAL_COLUMNS = MASTER_COLUMNS + ['near_duplicate_flag', 'near_duplicate_partner_id']

# ---------------------------------------------------------------------------
# Near-Duplicate Similarity Pre-Filter
# ---------------------------------------------------------------------------
def check_similarity_prefilter(
    counts1: Dict[str, int],
    counts2: Dict[str, int],
    len1: int,
    len2: int
) -> bool:
    """
    Checks if two sequences can possibly have a SequenceMatcher ratio > 0.90.
    1. Length difference must be <= 10% of the maximum length.
    2. Character frequency difference must be <= 20% of the maximum length.
    """
    max_len = max(len1, len2)
    
    # 1. Length constraint: diff <= 10%
    if abs(len1 - len2) > 0.10 * max_len:
        return False
        
    # 2. Count constraint: mismatch count diff <= 20%
    diff = 0
    all_bases = set(counts1.keys()).union(counts2.keys())
    for base in all_bases:
        diff += abs(counts1.get(base, 0) - counts2.get(base, 0))
        
    if diff > 0.20 * max_len:
        return False
        
    return True

# ---------------------------------------------------------------------------
# Main Pipeline execution
# ---------------------------------------------------------------------------
def main():
    input_files = {
        'rtprimerdb': 'data/processed/rtprimerdb_clean.csv',
        'artic': 'data/processed/artic_clean.csv',
        'primerbank': 'data/processed/primerbank_clean.csv'
    }

    loaded_dfs = []
    input_stats = {db: 0 for db in input_files}

    # Load clean data files
    for db, path in input_files.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                row_count = len(df)
                input_stats[db] = row_count
                print(f"Loaded {row_count} entries from: {path}")
                if row_count > 0:
                    loaded_dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {path} due to error: {e}", file=sys.stderr)
        else:
            print(f"Warning: Processed file not found, skipping: {path}", file=sys.stderr)

    if not loaded_dfs:
        print("Error: No input data files loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Merge DataFrames
    df_merged = pd.concat(loaded_dfs, ignore_index=True)
    print(f"Merged raw database has N={len(df_merged)} rows.")

    # Reindex columns to match master schema
    df_merged = df_merged.reindex(columns=MASTER_COLUMNS)

    # Normalize sequence columns to uppercase, stripped for matching
    df_merged['sequence_fwd'] = df_merged['sequence_fwd'].astype(str).str.upper().str.strip()
    df_merged['sequence_rev'] = df_merged['sequence_rev'].astype(str).str.upper().str.strip()

    # Generate sequence pair grouping key
    df_merged['seq_key'] = df_merged['sequence_fwd'] + "_" + df_merged['sequence_rev']

    groups = df_merged.groupby('seq_key')

    kept_rows = []
    removed_rows = []
    
    conflicts_resolved_count = 0
    dup_removed_stats = {db: 0 for db in input_files}

    confidence_map = {'high': 3, 'medium': 2, 'low': 1}
    source_map = {'rtprimerdb': 3, 'artic': 2, 'primerbank': 1}

    print("Deduplicating exact sequence matches and resolving label conflicts...")

    for seq_key, group in groups:
        if len(group) == 1:
            kept_rows.append(group.iloc[0].to_dict())
            continue

        # Check for label conflicts
        labels = group['label'].unique()
        has_conflict = (1 in labels) and (0 in labels)

        if has_conflict:
            conflicts_resolved_count += 1
            label_1_rows = group[group['label'] == 1]
            label_0_rows = group[group['label'] == 0]

            # Discard label=0 rows
            for _, r in label_0_rows.iterrows():
                row_dict = r.to_dict()
                row_dict['dedup_reason'] = "label_conflict_discarded_0"
                removed_rows.append(row_dict)
                db_src = str(row_dict.get('source_db', '')).lower()
                if db_src in dup_removed_stats:
                    dup_removed_stats[db_src] += 1

            # Update remaining label=1 rows to low confidence and append label conflict notes
            label_1_rows = label_1_rows.copy()
            label_1_rows['label_confidence'] = "low"
            notes_append = "conflict resolved: label=1 from higher-trust source"
            label_1_rows['label_notes'] = label_1_rows['label_notes'].apply(
                lambda x: f"{x}; {notes_append}" if pd.notna(x) and str(x).strip() else notes_append
            )
            candidates = label_1_rows
        else:
            candidates = group

        # Sort remaining candidates by confidence, then source priority
        def get_row_priority(r):
            conf = str(r.get('label_confidence', '')).strip().lower()
            conf_score = confidence_map.get(conf, 0)
            
            src = str(r.get('source_db', '')).strip().lower()
            src_score = source_map.get(src, 0)
            
            return (conf_score, src_score)

        candidate_list = [r.to_dict() for _, r in candidates.iterrows()]
        candidate_list.sort(key=get_row_priority, reverse=True)

        # Keep the best candidate
        best_row = candidate_list[0]
        kept_rows.append(best_row)

        # Discard the rest as duplicates
        for other_row in candidate_list[1:]:
            other_row['dedup_reason'] = "exact_duplicate_discarded"
            removed_rows.append(other_row)
            db_src = str(other_row.get('source_db', '')).lower()
            if db_src in dup_removed_stats:
                dup_removed_stats[db_src] += 1

    # Save exact deduplication log
    os.makedirs("data/processed", exist_ok=True)
    if removed_rows:
        df_dedup_log = pd.DataFrame(removed_rows)
        df_dedup_log.to_csv("data/processed/dedup_log.csv", index=False)
        print(f"Logged N={len(removed_rows)} exact duplicates/conflicts to: 'data/processed/dedup_log.csv'")
    else:
        # Create empty log with columns if no duplicates found
        pd.DataFrame(columns=MASTER_COLUMNS + ['dedup_reason']).to_csv("data/processed/dedup_log.csv", index=False)

    df_dedup = pd.DataFrame(kept_rows)

    # ---------------------------------------------------------------------------
    # Human-Only Post-Filtering
    # ---------------------------------------------------------------------------
    nonhuman_keywords = ['scerevisiae', 'drosophila', 'avian', 'yeast', 'zebrafish', 'xenopus', 'celegans']
    print(f"Applying strict human-only filter. Rejecting gene matches containing: {nonhuman_keywords}")

    nonhuman_mask = df_dedup['gene_name'].astype(str).str.lower().apply(
        lambda val: any(kw in val for kw in nonhuman_keywords)
    )

    df_nonhuman = df_dedup[nonhuman_mask]
    df_master = df_dedup[~nonhuman_mask].copy()

    # Save non-human removed entries
    df_nonhuman.to_csv("data/processed/nonhuman_removed.csv", index=False)
    print(f"Logged N={len(df_nonhuman)} non-human records to: 'data/processed/nonhuman_removed.csv'")

    nonhuman_removed_count = len(df_nonhuman)

    # ---------------------------------------------------------------------------
    # Near-Duplicate Detection (Pruned length check)
    # ---------------------------------------------------------------------------
    print("Performing near-duplicate sequence detection (SequenceMatcher > 0.90)...")
    df_master['near_duplicate_flag'] = False
    df_master['near_duplicate_partner_id'] = ""

    records = df_master.to_dict('records')
    n = len(records)

    # Pre-calculate sequence lengths and character frequencies to optimize execution
    for r in records:
        r['len_fwd'] = len(r['sequence_fwd'])
        r['len_rev'] = len(r['sequence_rev'])
        r['counts_fwd'] = Counter(r['sequence_fwd'])
        r['counts_rev'] = Counter(r['sequence_rev'])

    # Sort indices by length for forward sequences
    sorted_by_fwd = sorted(range(n), key=lambda idx: records[idx]['len_fwd'])

    for i_idx, i in enumerate(sorted_by_fwd):
        r_i = records[i]
        len_i = r_i['len_fwd']
        seq_i = r_i['sequence_fwd']
        counts_i = r_i['counts_fwd']

        for j_idx in range(i_idx + 1, n):
            j = sorted_by_fwd[j_idx]
            r_j = records[j]
            len_j = r_j['len_fwd']

            # Stop scanning if length difference is strictly > 10%
            if len_j - len_i > 0.10 * len_i:
                break

            # Fast character overlap pre-filter
            if check_similarity_prefilter(counts_i, r_j['counts_fwd'], len_i, len_j):
                ratio = SequenceMatcher(None, seq_i, r_j['sequence_fwd']).ratio()
                if ratio > 0.90:
                    r_i['near_duplicate_flag'] = True
                    r_j['near_duplicate_flag'] = True
                    if not r_i['near_duplicate_partner_id']:
                        r_i['near_duplicate_partner_id'] = r_j['primer_id']
                    if not r_j['near_duplicate_partner_id']:
                        r_j['near_duplicate_partner_id'] = r_i['primer_id']

    # Sort indices by length for reverse sequences
    sorted_by_rev = sorted(range(n), key=lambda idx: records[idx]['len_rev'])

    for i_idx, i in enumerate(sorted_by_rev):
        r_i = records[i]
        len_i = r_i['len_rev']
        seq_i = r_i['sequence_rev']
        counts_i = r_i['counts_rev']

        for j_idx in range(i_idx + 1, n):
            j = sorted_by_rev[j_idx]
            r_j = records[j]
            len_j = r_j['len_rev']

            # Stop scanning if length difference is strictly > 10%
            if len_j - len_i > 0.10 * len_i:
                break

            # Fast character overlap pre-filter
            if check_similarity_prefilter(counts_i, r_j['counts_rev'], len_i, len_j):
                ratio = SequenceMatcher(None, seq_i, r_j['sequence_rev']).ratio()
                if ratio > 0.90:
                    r_i['near_duplicate_flag'] = True
                    r_j['near_duplicate_flag'] = True
                    if not r_i['near_duplicate_partner_id']:
                        r_i['near_duplicate_partner_id'] = r_j['primer_id']
                    if not r_j['near_duplicate_partner_id']:
                        r_j['near_duplicate_partner_id'] = r_i['primer_id']

    # Re-build final DataFrame from updated records
    df_final = pd.DataFrame(records)
    
    # Save flagged entries to near_duplicates_flagged.csv for manual review
    df_flagged = df_final[df_final['near_duplicate_flag'] == True]
    
    # Save flagged entries
    flagged_columns = ['primer_id', 'gene_name', 'organism', 'source_db', 'sequence_fwd', 'sequence_rev', 'near_duplicate_partner_id']
    df_flagged_out = df_flagged.reindex(columns=flagged_columns)
    df_flagged_out.to_csv("data/processed/near_duplicates_flagged.csv", index=False)
    print(f"Saved N={len(df_flagged)} near-duplicate flagged records to: 'data/processed/near_duplicates_flagged.csv'")

    near_duplicates_flagged_count = len(df_flagged)

    # Save final master training database matching columns
    df_output = df_final.reindex(columns=FINAL_COLUMNS)
    df_output.to_csv("data/master_training_db_v2.csv", index=False)
    print(f"\nFinal merged Master Database saved successfully (N={len(df_output)} rows) to: 'data/master_training_db_v2.csv'")

    # ---------------------------------------------------------------------------
    # Final QC Report
    # ---------------------------------------------------------------------------
    print("\n" + "="*50)
    print("MASTER TRAINING DATABASE MERGE & QC REPORT")
    print("="*50)
    
    print("1. Input Rows Loaded per Source Database:")
    for db, count in input_stats.items():
        print(f"   - {db}: {count} rows")
        
    print("\n2. Exact Duplicates/Conflicts Discarded by Source:")
    for db, count in dup_removed_stats.items():
        print(f"   - {db}: {count} rows")
        
    print(f"\n3. Label Conflicts Resolved: {conflicts_resolved_count}")
    print(f"4. Non-Human Entries Filtered:  {nonhuman_removed_count}")
    print(f"5. Near-Duplicates Flagged:     {near_duplicates_flagged_count}")
    print(f"6. Final Unified Row Count:     {len(df_output)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
