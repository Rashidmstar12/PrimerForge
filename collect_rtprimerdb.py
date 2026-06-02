#!/usr/bin/env python3
"""
collect_rtprimerdb.py

Bioinformatics pipeline script to parse, clean, and validate
raw RTPrimerDB flat-file exports before master database ingestion.
"""

import os
import sys
import argparse
import re
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Import validation helpers from our standalone module
from data_collection_checklist import (
    validate_primer_entry,
    calculate_gc,
    calculate_santa_lucia_tm
)

def build_column_mapping(df_columns: List[str]) -> Dict[str, str]:
    """
    Normalizes column headers case-insensitively and maps common
    variations to the expected raw schema columns.
    """
    cols_lower = {c.lower().replace("_", "").replace(" ", ""): c for c in df_columns}
    
    mapping = {}
    
    # Expected key mapping variations
    expected_maps = {
        'id': ['id', 'assayid', 'primerid', 'recordid'],
        'assay_name': ['assayname', 'genename', 'genesymbol', 'referencegene', 'target', 'gene'],
        'sequence_fwd': ['forwardseq', 'forward', 'fwdseq', 'sequencefwd', 'primerfwd', 'fwd'],
        'sequence_rev': ['reverseseq', 'reverse', 'revseq', 'sequencerev', 'primerrev', 'rev'],
        'efficiency': ['efficiency', 'qpcrefficiency', 'amp_efficiency', 'eff', 'e'],
        'r_squared': ['rsquared', 'r2', 'r_squared', 'rsq', 'r2value'],
        'amplicon_size': ['ampliconsize', 'size', 'ampliconlength', 'productsize', 'bp'],
        'organism': ['organism', 'species', 'targetorganism', 'taxa'],
        'tissue': ['tissue', 'cellline', 'celllinetested', 'sampletype', 'source'],
        'ncbi_accession': ['ncbiaccession', 'accession', 'accessionid', 'genbank', 'refseq'],
        'pubmed_id': ['pubmedid', 'pmid', 'pubmed', 'reference']
    }
    
    for standard_name, variants in expected_maps.items():
        found = False
        for var in variants:
            # Check normalized variant name
            norm_var = var.lower().replace("_", "").replace(" ", "")
            if norm_var in cols_lower:
                mapping[cols_lower[norm_var]] = standard_name
                found = True
                break
        if not found:
            # Try matching prefixes or simple substring matches if still not found
            for col_raw in df_columns:
                norm_raw = col_raw.lower().replace("_", "").replace(" ", "")
                if any(v in norm_raw for v in variants):
                    mapping[col_raw] = standard_name
                    break
                    
    return mapping

def main():
    parser = argparse.ArgumentParser(
        description="Ingests, cleans, and validates raw RTPrimerDB flat-file exports."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw/rtprimerdb_export.txt",
        help="Path to the raw TSV flat-file export from RTPrimerDB (default: data/raw/rtprimerdb_export.txt)"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Runs validation checks without writing output files."
    )
    args = parser.parse_args()
    
    # 1. Handle missing raw input file cases
    if not os.path.exists(args.input):
        print(f"Error: Input raw file not found at: '{args.input}'", file=sys.stderr)
        print("To download the flat-file export, please visit the public RTPrimerDB portal at:", file=sys.stderr)
        print("  - https://rtprimerdb.org/ or http://medgen.ugent.be/rtprimerdb", file=sys.stderr)
        print("Download the TSV database export and place it in 'data/raw/rtprimerdb_export.txt'", file=sys.stderr)
        sys.exit(1)
        
    print(f"Reading RTPrimerDB raw dataset from: '{args.input}'...")
    
    # 2. Load file and handle encoding issues
    try:
        df_raw = pd.read_csv(args.input, sep='\t', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df_raw = pd.read_csv(args.input, sep='\t', encoding='latin-1')
        except Exception as e:
            print(f"Fatal error reading input file: {e}", file=sys.stderr)
            sys.exit(1)
            
    total_raw_rows = len(df_raw)
    print(f"Successfully loaded N={total_raw_rows} records from flat-file export.")
    
    # 3. Detect column schema mapping
    column_mapping = build_column_mapping(df_raw.columns)
    
    # Verify minimal expected columns
    minimal_cols = {'sequence_fwd', 'sequence_rev', 'organism'}
    mapped_standards = set(column_mapping.values())
    missing_minimals = minimal_cols - mapped_standards
    if missing_minimals:
        print("Error: Could not auto-detect minimal required columns in input schema.", file=sys.stderr)
        print(f"Missing headers: {list(missing_minimals)}", file=sys.stderr)
        print(f"Detected columns: {list(df_raw.columns)}", file=sys.stderr)
        print("Ensure input file matches raw RTPrimerDB export schema.", file=sys.stderr)
        sys.exit(1)
        
    # Rename columns to standardized internal names
    df = df_raw.rename(columns=column_mapping)
    
    # Keep track of records by their curation disposition
    dispositions = {
        'kept_label1': [],
        'kept_label0': [],
        'discarded_borderline': [],
        'rejected_invalid': []
    }
    
    # Eukaryotic / viral organism filtering lists
    whitelist = ['human', 'mouse', 'homo', 'mus', 'rat', 'viral', 'sars', 'influenza', 'rattus', 'virus']
    blacklist = ['e.coli', 'bacterial', 'bacillus', 'coli', 'bacteria', 'archaeal', 'archaea']
    
    # Dictionary to manage unique ID counts
    gene_counters = {}
    existing_ids = []
    
    print("Processing and validating records...")
    for idx, row in df.iterrows():
        # Get organism and verify eukaryotic boundary rules
        org_name = str(row.get('organism', '')).strip().lower()
        if not org_name or not any(w in org_name for w in whitelist) or any(b in org_name for b in blacklist):
            dispositions['rejected_invalid'].append({
                'row_index': idx,
                'reason': f"Non-eukaryotic organism: {row.get('organism', 'UNKNOWN')}"
            })
            continue
            
        # Parse sequence characters
        f_seq = str(row.get('sequence_fwd', '')).strip().upper()
        r_seq = str(row.get('sequence_rev', '')).strip().upper()
        
        # Apply EWC label standard logic
        raw_eff = row.get('efficiency')
        raw_r2 = row.get('r_squared')
        
        # Check efficiency and r_squared values to assign classification label
        try:
            eff = float(raw_eff) if pd.notna(raw_eff) else None
        except (ValueError, TypeError):
            eff = None
            
        try:
            r2 = float(raw_r2) if pd.notna(raw_r2) else None
        except (ValueError, TypeError):
            r2 = None
            
        if eff is None:
            # No efficiency reported: label=1, medium confidence (as it is included in database)
            label = 1
            confidence = "medium"
            disp = 'kept_label1'
        elif eff < 1.60:
            label = 0
            confidence = "high"
            disp = 'kept_label0'
        elif 1.60 <= eff <= 1.79:
            # Borderline efficiency -> DISCARD
            dispositions['discarded_borderline'].append({
                'row_index': idx,
                'reason': f"Borderline efficiency: E={eff}"
            })
            continue
        elif eff >= 1.80:
            if r2 is not None and r2 >= 0.98:
                label = 1
                confidence = "high"
                disp = 'kept_label1'
            else:
                label = 1
                confidence = "medium"
                disp = 'kept_label1'
        else:
            dispositions['rejected_invalid'].append({
                'row_index': idx,
                'reason': "Invalid efficiency format"
            })
            continue
            
        # Generate Unique ID (RTPDB_GENENAME_PADDEDINDEX)
        raw_gene = str(row.get('assay_name', row.get('gene_name', 'UNKNOWN'))).strip().upper()
        # Clean gene name to alphanumeric characters only
        gene_clean = re.sub(r'[^A-Z0-9]', '', raw_gene)
        if not gene_clean:
            gene_clean = "UNKNOWN"
            
        if gene_clean not in gene_counters:
            gene_counters[gene_clean] = 1
        else:
            gene_counters[gene_clean] += 1
            
        p_id = f"RTPDB_{gene_clean}_{gene_counters[gene_clean]:03d}"
        
        # Parse PMC reference / paper DOI
        pmid = str(row.get('pubmed_id', '')).strip()
        # Check if PMID contains decimal places
        if pmid.endswith('.0'):
            pmid = pmid[:-2]
            
        doi_val = "N/A"
        if pmid and pmid.upper() not in ['NAN', 'NONE', 'NULL', '']:
            doi_val = f"PMID:{pmid}"
            
        # Assemble dictionary row mapping to master_schema columns
        mapped_row = {
            # Identity
            'primer_id': p_id,
            'gene_name': gene_clean,
            'organism': org_name,
            'source_db': 'rtprimerdb',
            'paper_doi': doi_val,
            # Sequences
            'sequence_fwd': f_seq,
            'sequence_rev': r_seq,
            # Labels
            'label': label,
            'label_confidence': confidence,
            'label_source': 'rtprimerdb',
            'label_notes': f"Original ID: {row.get('id', 'N/A')}",
            # Experimental Metadata
            'cell_line_tested': str(row.get('tissue', '')) if pd.notna(row.get('tissue')) else "",
            'annealing_tm_used_C': np.nan,
            'amplicon_size_bp': int(row.get('amplicon_size')) if pd.notna(row.get('amplicon_size')) else 200, # default fallback if missing
            'qpcr_efficiency': eff if eff is not None else np.nan,
            'qpcr_r2': r2 if r2 is not None else np.nan,
            'gel_band_confirmed': True if label == 1 else False,
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
        
        # Programmatic quality validation checks
        is_valid, validation_errors = validate_primer_entry(mapped_row, existing_ids)
        if not is_valid:
            dispositions['rejected_invalid'].append({
                'row_index': idx,
                'reason': f"QC checks failed: {validation_errors}"
            })
            continue
            
        # Append record to the corresponding kept list
        existing_ids.append(p_id)
        dispositions[disp].append(mapped_row)
        
    # Combine kept rows
    kept_records = dispositions['kept_label1'] + dispositions['kept_label0']
    
    # 4. Ingestion stats output summary report
    print("\n" + "=" * 50)
    print("           DATASET INGESTION SUMMARY REPORT")
    print("=" * 50)
    print(f"Total Flat File Rows Read:     {total_raw_rows}")
    print(f"Kept Class 1 (Functional):     {len(dispositions['kept_label1'])}")
    print(f"Kept Class 0 (Non-Functional):  {len(dispositions['kept_label0'])}")
    print(f"Discarded (Borderline):        {len(dispositions['discarded_borderline'])}")
    print(f"Rejected (QC Fail / Organism):  {len(dispositions['rejected_invalid'])}")
    print("-" * 50)
    
    if kept_records:
        kept_df = pd.DataFrame(kept_records)
        
        # Label distribution
        label_counts = kept_df['label'].value_counts()
        print("Label Distribution in Kept Rows:")
        for lbl, count in label_counts.items():
            print(f"  - Label {lbl}: {count} ({count/len(kept_df)*100:.1f}%)")
            
        # Organism breakdown
        org_counts = kept_df['organism'].value_counts()
        print("\nTarget Organism Distribution:")
        for org, count in org_counts.items():
            print(f"  - {org}: {count}")
            
        # Compute GC and Tm stats from sequences directly
        gc_values = []
        tm_values = []
        for r in kept_records:
            gc_values.extend([calculate_gc(r['sequence_fwd']), calculate_gc(r['sequence_rev'])])
            tm_values.extend([calculate_santa_lucia_tm(r['sequence_fwd']), calculate_santa_lucia_tm(r['sequence_rev'])])
            
        print("\nBiophysical Profile in Kept Primers:")
        print(f"  - GC Content: Min={min(gc_values):.1f}%, Max={max(gc_values):.1f}%, Mean={np.mean(gc_values):.1f}%")
        print(f"  - Melting Tm: Min={min(tm_values):.1f}°C, Max={max(tm_values):.1f}°C, Mean={np.mean(tm_values):.1f}°C")
        
        # 5. Output clean file
        if args.dry_run:
            print("\nDry-run mode active. Skipped exporting output files.")
        else:
            out_file = "data/processed/rtprimerdb_clean.csv"
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            kept_df.to_csv(out_file, index=False)
            print(f"\nSaved clean database file successfully to: '{out_file}'")
    else:
        print("\nWarning: No records passed the database validation standards.", file=sys.stderr)
        
    print("=" * 50)

if __name__ == "__main__":
    main()
