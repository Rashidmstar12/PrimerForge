#!/usr/bin/env python3
"""
collect_primerbank.py

Bioinformatics script to ingest, parse, and validate RT-PCR primer data from
the PrimerBank database, mapping them to the PrimerForge master training schema.
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
def download_file_with_retry(url: str, retries: int = 3, timeout: int = 10) -> str:
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
# Delimiter Auto-Detection
# ---------------------------------------------------------------------------
def detect_delimiter(file_content: str) -> str:
    """Detects whether the file content is tab, comma, or pipe-delimited."""
    lines = [line for line in file_content.split('\n') if line.strip()][:10]
    if not lines:
        return ','
    
    delimiters = [',', '\t', '|']
    counts = {d: 0 for d in delimiters}
    for line in lines:
        for d in delimiters:
            counts[d] += line.count(d)
            
    best_delim = max(counts, key=counts.get)
    if counts[best_delim] == 0:
        return ','
    return best_delim

# ---------------------------------------------------------------------------
# Fuzzy Column Matching
# ---------------------------------------------------------------------------
def fuzzy_match_columns(df_cols: List[str]) -> Dict[str, str]:
    """
    Maps df_cols to target columns using case-insensitive fuzzy matches.
    """
    mapping = {}
    
    patterns = {
        'sequence_fwd': [r'fwd', r'forward', r'f_primer', r'fwd_seq', r'f_seq'],
        'sequence_rev': [r'rev', r'reverse', r'r_primer', r'rev_seq', r'r_seq'],
        'amplicon_size_bp': [r'size', r'product_size', r'amplicon_size', r'length', r'len'],
        'gene_name': [r'gene', r'symbol', r'target_id', r'target'],
        'ncbi_accession': [r'accession', r'acc', r'refseq', r'ncbi_accession'],
        'description': [r'desc', r'description'],
        'species': [r'species', r'organism', r'tax']
    }
    
    for target, regexes in patterns.items():
        matched_col = None
        # Exact match first (case-insensitive)
        for col in df_cols:
            col_clean = col.lower().strip()
            if col_clean == target:
                matched_col = col
                break
        
        # Regex search fallback
        if not matched_col:
            for col in df_cols:
                col_clean = col.lower().strip()
                for pat in regexes:
                    if re.search(pat, col_clean):
                        matched_col = col
                        break
                if matched_col:
                    break
        
        if matched_col:
            mapping[target] = matched_col
            
    return mapping

# ---------------------------------------------------------------------------
# Organism Filtering
# ---------------------------------------------------------------------------
def is_human_row(row: Dict[str, Any], col_mapping: Dict[str, str]) -> bool:
    """Checks if the row represents human sequence data."""
    # 1. Check species column if present
    if 'species' in col_mapping:
        val = str(row.get(col_mapping['species'], '')).lower()
        if 'mouse' in val or 'mus' in val or 'm.musculus' in val:
            return False
        if 'human' in val or 'homo' in val or 'sapiens' in val:
            return True
            
    # 2. Check organism column if present
    if 'organism' in col_mapping:
        val = str(row.get(col_mapping['organism'], '')).lower()
        if 'mouse' in val or 'mus' in val:
            return False
        if 'human' in val or 'homo' in val or 'sapiens' in val:
            return True
            
    # 3. Check description column if present
    if 'description' in col_mapping:
        val = str(row.get(col_mapping['description'], '')).lower()
        if 'mouse' in val or 'mus' in val:
            return False
        if 'human' in val or 'homo' in val or 'sapiens' in val:
            return True
            
    # Default fallback: if no species-specific columns exist, keep the entry
    return True

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    raw_path = "data/raw/primerbank_download.txt"
    file_content = ""

    # Scenario A: Local File Check
    if os.path.exists(raw_path):
        print(f"Scenario A: Found local PrimerBank file at: {raw_path}")
        try:
            with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
        except Exception as e:
            print(f"Error reading local file: {e}")

    # Scenario B: Fallbacks Check
    if not file_content:
        url1 = "https://pga.mgh.harvard.edu/primerbank/images/primerbank_info.txt"
        print(f"Scenario B: Local file not found. Fetching primary endpoint: {url1}")
        file_content = download_file_with_retry(url1)

        if not file_content:
            url2 = "https://pga.mgh.harvard.edu/primerbank/download.html"
            print(f"Primary endpoint failed. Attempting to parse links from: {url2}")
            html_content = download_file_with_retry(url2)
            if html_content:
                # Find download links (.txt, .zip, .csv, .gz)
                links = re.findall(r'href=["\']([^"\']+\.(?:txt|zip|gz|csv))["\']', html_content, re.IGNORECASE)
                for link in links:
                    if not link.startswith('http'):
                        base_url = "https://pga.mgh.harvard.edu/primerbank/"
                        resolved_url = "https://pga.mgh.harvard.edu" + link if link.startswith('/') else base_url + link
                    else:
                        resolved_url = link
                    
                    print(f"Found link candidate: {resolved_url}. Fetching...")
                    file_content = download_file_with_retry(resolved_url)
                    if file_content:
                        break

    # Exit if all inputs failed
    if not file_content:
        print("\n" + "="*80)
        print("CRITICAL ERROR: Failed to automatically obtain PrimerBank data.")
        print("="*80)
        print("Please follow these steps to manually download the dataset:")
        print("1. Navigate to the PrimerBank webpage: http://pga.mgh.harvard.edu/primerbank/")
        print("2. Search or find the bulk primer download options.")
        print("3. Save the flat file (tab, comma, or pipe-delimited) locally to:")
        print(f"   {raw_path}")
        print("4. Re-run this script.")
        print("="*80 + "\n")
        sys.exit(1)

    # Delimiter and column detection
    delim = detect_delimiter(file_content)
    print(f"Auto-detected delimiter: '{repr(delim)}'")

    # Read data into DataFrame
    from io import StringIO
    df = pd.read_csv(StringIO(file_content), sep=delim)
    print(f"Parsed raw table with N={len(df)} rows. Columns: {list(df.columns)}")

    # Fuzzy match columns
    col_mapping = fuzzy_match_columns(list(df.columns))
    print(f"Detected Column Mapping: {col_mapping}")

    # Check for core columns
    required_keys = ['sequence_fwd', 'sequence_rev']
    missing_keys = [k for k in required_keys if k not in col_mapping]
    if missing_keys:
        print(f"Error: Could not locate core sequence columns. Missing: {missing_keys}", file=sys.stderr)
        sys.exit(1)

    primary_records = []
    existing_ids = []
    gene_counts = {}
    
    total_entries = len(df)
    kept_count = 0
    rejected_count = 0

    for idx_row, raw_row in df.iterrows():
        row_dict = raw_row.to_dict()

        # Species Filter
        if not is_human_row(row_dict, col_mapping):
            rejected_count += 1
            continue

        # Extract sequences and details
        fwd_seq = str(row_dict.get(col_mapping['sequence_fwd'], '')).upper().strip()
        rev_seq = str(row_dict.get(col_mapping['sequence_rev'], '')).upper().strip()

        # Extract optional size
        size = np.nan
        if 'amplicon_size_bp' in col_mapping:
            try:
                size_val = row_dict.get(col_mapping['amplicon_size_bp'])
                if pd.notna(size_val):
                    size = int(float(size_val))
            except Exception:
                pass

        # Extract optional gene
        gene_name = "UNKNOWN"
        if 'gene_name' in col_mapping:
            gene_name = str(row_dict.get(col_mapping['gene_name'], '')).strip()

        # Extract optional accession
        accession = ""
        if 'ncbi_accession' in col_mapping:
            accession = str(row_dict.get(col_mapping['ncbi_accession'], '')).strip()

        # Generate standard ID PBANK_GENESYMBOL_INDEX
        gene_upper = re.sub(r'[^A-Z0-9]', '', gene_name.upper())
        if not gene_upper:
            gene_upper = "UNKNOWN"
            
        gene_counts[gene_upper] = gene_counts.get(gene_upper, 0) + 1
        p_id = f"PBANK_{gene_upper}_{gene_counts[gene_upper]:03d}"

        # Map to schema row
        mapped_row = {
            # Identity
            'primer_id': p_id,
            'gene_name': gene_upper,
            'organism': 'human',
            'source_db': 'PrimerBank',
            'paper_doi': 'primerbank_db',
            # Sequences
            'sequence_fwd': fwd_seq,
            'sequence_rev': rev_seq,
            # Labels
            'label': 1,
            'label_confidence': 'medium',
            'label_source': 'primerbank',
            'label_notes': f"Accession: {accession}" if accession else "",
            # Experimental Metadata
            'cell_line_tested': "",
            'annealing_tm_used_C': np.nan,
            'amplicon_size_bp': size,
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
        if is_valid:
            existing_ids.append(p_id)
            primary_records.append(mapped_row)
            kept_count += 1
        else:
            rejected_count += 1

    # Save output
    os.makedirs("data/processed", exist_ok=True)
    if primary_records:
        clean_df = pd.DataFrame(primary_records)
        clean_df.to_csv("data/processed/primerbank_clean.csv", index=False)
        print(f"\nSaved N={len(primary_records)} validated PrimerBank records to: 'data/processed/primerbank_clean.csv'")
    else:
        print("\nNo entries passed QC validation standard.", file=sys.stderr)

    # Print summary metrics
    print("\n" + "="*40)
    print("PrimerBank Ingestion Summary")
    print("="*40)
    print(f"Total entries parsed:   {total_entries}")
    print(f"Kept (validated human): {kept_count}")
    print(f"Rejected (QC / species): {rejected_count}")

    # Print top genes breakdown
    if primary_records:
        top_genes = pd.Series([r['gene_name'] for r in primary_records]).value_counts().head(5)
        print("\nTop 5 Genes Breakdown:")
        for gene, count in top_genes.items():
            print(f"  - {gene}: {count} pairs")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
