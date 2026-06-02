#!/usr/bin/env python3
"""
collect_artic_schemes.py

Bioinformatics pipeline script to download, parse, and validate ARTIC Network
primer schemes from GitHub, mapping them to the PrimerForge master training schema.
"""

import os
import re
import sys
import time
import datetime
import requests
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Import validation helpers from our standalone module
from data_collection_checklist import (
    validate_primer_entry
)

# ---------------------------------------------------------------------------
# Download with Retries and Timeout
# ---------------------------------------------------------------------------
def download_bed_file(url: str, retries: int = 3, timeout: int = 10) -> str:
    """Downloads the file with timeout and exponential backoff retry logic."""
    for attempt in range(1, retries + 2):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 404:
                print(f"URL returned HTTP 404 (Not Found): {url}")
                return ""
            else:
                print(f"Attempt {attempt} failed with status {response.status_code}: {url}")
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
        if attempt <= retries:
            time.sleep(1.5)
    return ""

# ---------------------------------------------------------------------------
# IUPAC 3' Degeneracy Checker
# ---------------------------------------------------------------------------
def check_degenerate_3prime(seq: str) -> bool:
    """Checks if any of the last 3 nucleotides in the sequence contain IUPAC codes."""
    iupac_codes = set("RYSWKMBDHVN")
    if len(seq) < 3:
        return any(b in iupac_codes for b in seq.upper())
    return any(b in iupac_codes for b in seq[-3:].upper())

# ---------------------------------------------------------------------------
# Schema Mapping Helper
# ---------------------------------------------------------------------------
def map_artic_pair(
    left: Dict[str, Any],
    right: Dict[str, Any],
    prefix: str,
    amp_num: int,
    is_alt: bool,
    existing_ids: List[str]
) -> Tuple[Dict[str, Any], bool, str]:
    """Maps a paired LEFT/RIGHT primer record to the master training schema."""
    # Normalize scheme naming
    pref_lower = prefix.lower()
    if 'ncov' in pref_lower or '2019' in pref_lower or 'sars' in pref_lower:
        scheme_name = 'NCOV'
        organism = 'sars-cov-2'
        doi = '10.1038/s41587-020-0581-y'
    elif 'mpox' in pref_lower or 'monkeypox' in pref_lower:
        scheme_name = 'MPOX'
        organism = 'monkeypox'
        doi = 'N/A'
    elif 'influenza' in pref_lower:
        scheme_name = 'INF'
        organism = 'influenza'
        doi = 'N/A'
    else:
        scheme_name = prefix.upper()
        organism = pref_lower
        doi = 'N/A'

    # Format scheme suffix for alternative variants to avoid duplicates and match pattern
    scheme_suffix = "ALT" if is_alt else ""
    base_id = f"ARTIC_{scheme_name}{scheme_suffix}"
    
    # Resolve potential ID duplication to ensure strict format matching
    candidate_id = f"{base_id}_{amp_num:03d}"
    idx = amp_num
    while candidate_id in existing_ids:
        idx += 100
        candidate_id = f"{base_id}_{idx:03d}"
    p_id = candidate_id

    left_seq = left['sequence'].upper().strip()
    right_seq = right['sequence'].upper().strip()

    # Check degenerate 3' bases
    has_degenerate_3prime = False
    if check_degenerate_3prime(left_seq) or check_degenerate_3prime(right_seq):
        has_degenerate_3prime = True
        return {}, False, "degenerate_3prime"

    # Compute amplicon size (chromEnd of RIGHT primer minus chromStart of LEFT primer)
    amp_size = int(right['chromEnd']) - int(left['chromStart'])

    mapped_row = {
        # Identity
        'primer_id': p_id,
        'gene_name': scheme_name,
        'organism': organism,
        'source_db': 'artic',
        'paper_doi': doi,
        # Sequences
        'sequence_fwd': left_seq,
        'sequence_rev': right_seq,
        # Labels
        'label': 1,
        'label_confidence': 'high',
        'label_source': 'artic',
        'label_notes': f"Amplicon {amp_num} (Alt: {is_alt})",
        # Experimental Metadata
        'cell_line_tested': "",
        'annealing_tm_used_C': np.nan,
        'amplicon_size_bp': amp_size,
        'qpcr_efficiency': np.nan,
        'qpcr_r2': np.nan,
        'gel_band_confirmed': True,
        'date_collected': datetime.date.today().isoformat(),
        # Biophysical Features (computed in a later step)
        'f_tm': np.nan, 'r_tm': np.nan, 'tm_diff': np.nan,
        'f_gc': np.nan, 'r_gc': np.nan, 'f_hairpin_dg': np.nan, 'r_hairpin_dg': np.nan,
        'cross_dimer_dg': np.nan, 'f_len': np.nan, 'r_len': np.nan,
        'f_clamp_gc': np.nan, 'r_clamp_gc': np.nan, 'f_poly_run': np.nan, 'r_poly_run': np.nan,
        'target_gc': np.nan, 'target_len': np.nan,
        # Flags
        'is_synthetic_negative': False,
        'is_prospective_validation': False,
        'is_held_out_test': False
    }

    # Run quality control validation checklist
    is_valid, validation_errors = validate_primer_entry(mapped_row, existing_ids)
    if not is_valid:
        return {}, False, f"qc_fail: {validation_errors}"

    return mapped_row, True, "passed"

def main():
    schemes = [
        {
            'name': 'SARS-CoV-2 V5.3.2',
            'url': 'https://raw.githubusercontent.com/artic-network/primer-schemes/master/nCoV-2019/V5.3.2/nCoV-2019.primer.bed',
            'fallback_url': 'https://raw.githubusercontent.com/artic-network/primer-schemes/master/nCoV-2019/V5.3.2/SARS-CoV-2.primer.bed'
        },
        {
            'name': 'Monkeypox V1',
            'url': 'https://raw.githubusercontent.com/artic-network/primer-schemes/master/mpox/V1/mpox.primer.bed'
        },
        {
            'name': 'Influenza A V1',
            'url': 'https://raw.githubusercontent.com/artic-network/primer-schemes/master/influenza/V1/influenza.primer.bed'
        }
    ]

    primary_records = []
    alt_records = []

    primary_existing_ids = []
    alt_existing_ids = []

    for scheme in schemes:
        print(f"\nProcessing scheme: {scheme['name']}...")
        url = scheme['url']
        print(f"Downloading from primary URL: {url}")
        bed_text = download_bed_file(url)

        if not bed_text and 'fallback_url' in scheme:
            fallback = scheme['fallback_url']
            print(f"Primary URL failed. Trying fallback URL: {fallback}")
            bed_text = download_bed_file(fallback)

        if not bed_text:
            print(f"Skipped scheme {scheme['name']} due to download failure or 404.")
            continue

        lines = bed_text.strip().split('\n')
        records = []
        has_sequence_col = True

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                continue
            parts = line.split('\t')
            if len(parts) < 7:
                has_sequence_col = False
                break
            records.append({
                'chrom': parts[0],
                'chromStart': int(parts[1]),
                'chromEnd': int(parts[2]),
                'name': parts[3],
                'score': parts[4],
                'strand': parts[5],
                'sequence': parts[6]
            })

        if not has_sequence_col:
            print(f"Warning: Skipped scheme {scheme['name']} - missing 7th sequence column.")
            continue

        print(f"Parsed N={len(records)} primer lines. Pairing LEFT and RIGHT primers...")

        # Group primers by (prefix, amp_num)
        amplicons = {}
        for rec in records:
            name = rec['name']
            # Pattern: prefix, amplicon number, optional alt tag, and LEFT/RIGHT direction
            match = re.match(r"^(.+?)_(\d+)(_(?:alt|alt\d*))?_(LEFT|RIGHT)$", name, re.IGNORECASE)
            if not match:
                continue
            prefix = match.group(1)
            amp_num = int(match.group(2))
            is_alt = bool(match.group(3))
            direction = match.group(4).upper()

            key = (prefix, amp_num)
            if key not in amplicons:
                amplicons[key] = {'primary_LEFT': None, 'primary_RIGHT': None, 'alt_LEFTs': [], 'alt_RIGHTs': []}

            if not is_alt:
                amplicons[key][f'primary_{direction}'] = rec
            else:
                amplicons[key][f'alt_{direction}s'].append(rec)

        # Pairing and stats counters
        scheme_stats = {
            'total_amplicons': len(amplicons),
            'kept': 0,
            'skipped_degenerate': 0,
            'skipped_qc': 0,
            'alt_variants_found': 0
        }

        for (prefix, amp_num), data in amplicons.items():
            p_left = data['primary_LEFT']
            p_right = data['primary_RIGHT']
            alt_lefts = data['alt_LEFTs']
            alt_rights = data['alt_RIGHTs']

            # Calculate alternative combinations found
            alt_found_count = 0
            if p_right:
                alt_found_count += len(alt_lefts)
            if p_left:
                alt_found_count += len(alt_rights)
            alt_found_count += len(alt_lefts) * len(alt_rights)
            scheme_stats['alt_variants_found'] += alt_found_count

            # 1. Primary pair pairing
            if p_left and p_right:
                mapped, success, reason = map_artic_pair(
                    p_left, p_right, prefix, amp_num, is_alt=False, existing_ids=primary_existing_ids
                )
                if success:
                    primary_existing_ids.append(mapped['primer_id'])
                    primary_records.append(mapped)
                    scheme_stats['kept'] += 1
                else:
                    if reason == "degenerate_3prime":
                        scheme_stats['skipped_degenerate'] += 1
                    else:
                        scheme_stats['skipped_qc'] += 1

            # 2. Alternative pair pairing
            # Pair alt_LEFT with primary_RIGHT
            if p_right:
                for a_left in alt_lefts:
                    mapped, success, reason = map_artic_pair(
                        a_left, p_right, prefix, amp_num, is_alt=True, existing_ids=alt_existing_ids
                    )
                    if success:
                        alt_existing_ids.append(mapped['primer_id'])
                        alt_records.append(mapped)
                    else:
                        if reason == "degenerate_3prime":
                            scheme_stats['skipped_degenerate'] += 1
                        else:
                            scheme_stats['skipped_qc'] += 1

            # Pair primary_LEFT with alt_RIGHT
            if p_left:
                for a_right in alt_rights:
                    mapped, success, reason = map_artic_pair(
                        p_left, a_right, prefix, amp_num, is_alt=True, existing_ids=alt_existing_ids
                    )
                    if success:
                        alt_existing_ids.append(mapped['primer_id'])
                        alt_records.append(mapped)
                    else:
                        if reason == "degenerate_3prime":
                            scheme_stats['skipped_degenerate'] += 1
                        else:
                            scheme_stats['skipped_qc'] += 1

            # Pair alt_LEFT with alt_RIGHT
            for a_left in alt_lefts:
                for a_right in alt_rights:
                    mapped, success, reason = map_artic_pair(
                        a_left, a_right, prefix, amp_num, is_alt=True, existing_ids=alt_existing_ids
                    )
                    if success:
                        alt_existing_ids.append(mapped['primer_id'])
                        alt_records.append(mapped)
                    else:
                        if reason == "degenerate_3prime":
                            scheme_stats['skipped_degenerate'] += 1
                        else:
                            scheme_stats['skipped_qc'] += 1

        # Print per-scheme summary as requested
        print(f"Scheme Summary for {scheme['name']}:")
        print(f"  - total amplicons: {scheme_stats['total_amplicons']}")
        print(f"  - kept: {scheme_stats['kept']}")
        print(f"  - skipped (degenerate): {scheme_stats['skipped_degenerate']}")
        print(f"  - alt variants found: {scheme_stats['alt_variants_found']}")

    # Write output clean CSV files
    os.makedirs("data/processed", exist_ok=True)

    if primary_records:
        primary_df = pd.DataFrame(primary_records)
        primary_df.to_csv("data/processed/artic_clean.csv", index=False)
        print(f"\nSaved N={len(primary_records)} primary records successfully to: 'data/processed/artic_clean.csv'")
    else:
        print("\nNo primary records met the validation standard.", file=sys.stderr)

    if alt_records:
        alt_df = pd.DataFrame(alt_records)
        alt_df.to_csv("data/processed/artic_alt_primers.csv", index=False)
        print(f"Saved N={len(alt_records)} alternative records successfully to: 'data/processed/artic_alt_primers.csv'")

if __name__ == "__main__":
    main()
