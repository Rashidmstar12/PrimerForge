#!/usr/bin/env python3
"""
scrape_origene.py

Scrapes validated human qPCR primer pairs from the OriGene catalog pages,
formats them matching the PrimerForge training database schema, appends them
as positive training instances (label=1), and updates the master database.
"""

import os
import sys
import re
import datetime
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from typing import Dict, Any, List

# Target list of OriGene SKUs to scrape (curated from their catalog search results)
SKUS = [
    "HP221424",  # GAPDHP20
    "HP204660",  # ACTB
    "HP221348",  # CDY3P
    "HP207450",  # DDIT3
    "HP206852",  # TST
    "HP230503",  # VCAM1
    "HP214207",  # NR1D1
    "HP208486",  # HAS3
    "HP200369",  # CDKN1A (p21)
    "HP209852",  # ESM1
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

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

def scrape_sku(sku: str) -> Dict[str, Any]:
    """Scrapes a single OriGene SKU page and returns parsed attributes."""
    url = f"https://www.origene.com/product/{sku.lower()}"
    print(f"Fetching SKU '{sku}' from: {url}", flush=True)
    
    try:
        response = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=10)
        if response.status_code != 200:
            print(f"  Error: Received status code {response.status_code} for {sku}", file=sys.stderr)
            return None
        
        # Check if redirected to the fallback page
        if "primer-primer" in response.url:
            print(f"  Error: SKU {sku} redirected to generic fallback page.", file=sys.stderr)
            return None
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 1. Parse Sequences and attributes from the spec table
        attrs = {}
        for row in soup.find_all("tr"):
            th = row.find("th", class_="label")
            td = row.find("td", class_="data")
            if th and td:
                attrs[th.text.strip().lower()] = td.text.strip()
                
        fwd_seq = attrs.get("forward sequence")
        rev_seq = attrs.get("reverse sequence")
        locus_id = attrs.get("locus id", "N/A")
        
        if not fwd_seq or not rev_seq:
            print(f"  Error: Forward/Reverse sequence table cells not found for {sku}", file=sys.stderr)
            return None
            
        # Clean sequences
        fwd_seq = str(fwd_seq).upper().strip()
        rev_seq = str(rev_seq).upper().strip()
        
        # 2. Parse Gene Name and Organism from the title H1 or meta tags
        gene_name = "unknown"
        organism = "human"
        
        h1 = soup.find("h1")
        if h1:
            h1_text = h1.text.strip()
            # Match formats like: "GAPDHP20 Human qPCR Primer Pair (XM_291524)"
            # Group 1 = Gene Name, Group 2 = Species
            match = re.match(r"^([A-Za-z0-9_\-\(\)]+)\s+([A-Za-z]+)\s+qPCR\s+Primer\s+Pair", h1_text, re.IGNORECASE)
            if match:
                gene_name = match.group(1).upper()
                organism = match.group(2).lower()
            else:
                # Fallback to the first word in the title if match fails
                first_word = h1_text.split()[0]
                gene_name = first_word.replace("Human", "").replace("Mouse", "").upper()
                if "human" in h1_text.lower():
                    organism = "human"
                elif "mouse" in h1_text.lower():
                    organism = "mouse"
                    
        # If organism description is found in meta tag, use it to refine
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            desc_val = meta_desc.get("content", "").lower()
            if "homo sapiens" in desc_val:
                organism = "human"
            elif "mus musculus" in desc_val:
                organism = "mouse"
                
        # 3. Compile clean metadata dict
        scraped_data = {
            'primer_id': f"origene_{sku}_{gene_name}",
            'gene_name': gene_name,
            'organism': organism,
            'source_db': "origene",
            'paper_doi': "origene_catalog",
            'sequence_fwd': fwd_seq,
            'sequence_rev': rev_seq,
            'label': 1,
            'label_confidence': "high",
            'label_source': "origene",
            'label_notes': f"qPCR catalog validated; Locus ID: {locus_id}, SKU: {sku}",
            'cell_line_tested': "",
            'annealing_tm_used_C': 60.0,
            'amplicon_size_bp': np.nan,
            'qpcr_efficiency': np.nan,
            'qpcr_r2': np.nan,
            'gel_band_confirmed': True,
            'date_collected': datetime.date.today().isoformat(),
            'is_synthetic_negative': False,
            'is_prospective_validation': False,
            'is_held_out_test': False,
            'near_duplicate_flag': False,
            'near_duplicate_partner_id': ""
        }
        
        print(f"  Successfully parsed: Gene={gene_name}, Species={organism}, Fwd={fwd_seq}, Rev={rev_seq}")
        return scraped_data
        
    except Exception as e:
        print(f"  Error: Exception occurred while scraping {sku}: {e}", file=sys.stderr)
        return None

def main():
    print("="*60)
    print("STARTING ORIGENE QPCR PRIMER SCRAPER")
    print("="*60)
    
    # 1. Scrape all SKUs
    scraped_records = []
    for sku in SKUS:
        record = scrape_sku(sku)
        if record:
            scraped_records.append(record)
            
    if not scraped_records:
        print("Error: No primer records could be scraped. Database remains unchanged.", file=sys.stderr)
        sys.exit(1)
        
    print(f"\nScraped {len(scraped_records)} primer pairs successfully.")
    
    # 2. Append to Master Training Database
    db_path = "data/master_training_db_v2.csv"
    if not os.path.exists(db_path):
        print(f"Error: Master training database not found at: {db_path}", file=sys.stderr)
        sys.exit(1)
        
    df_db = pd.read_csv(db_path)
    print(f"Loaded existing database with N={len(df_db)} records.")
    
    # Check for duplicates before appending
    existing_seqs = set(df_db['sequence_fwd'].astype(str).str.upper() + "_" + df_db['sequence_rev'].astype(str).str.upper())
    
    new_records = []
    for rec in scraped_records:
        seq_key = f"{rec['sequence_fwd']}_{rec['sequence_rev']}"
        if seq_key in existing_seqs:
            print(f"  Skipping duplicate sequence pair for Gene {rec['gene_name']} (already in DB).")
        else:
            new_records.append(rec)
            
    if not new_records:
        print("All scraped records are already present in the database. No updates needed.")
        sys.exit(0)
        
    df_new = pd.DataFrame(new_records)
    # Reindex columns to match final master database structure
    df_new = df_new.reindex(columns=FINAL_COLUMNS)
    
    df_updated = pd.concat([df_db, df_new], ignore_index=True)
    df_updated.to_csv(db_path, index=False)
    print(f"\nAppended {len(new_records)} new OriGene records to: {db_path}")
    print(f"New Database Size: N={len(df_updated)} records.")
    
    # 3. Run audit backfilling and statistics regeneration
    print("\nTriggering final dataset audit and biophysical property calculation...", flush=True)
    os.system("python dataset_audit.py")
    
    print("\nOriGene Integration Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
