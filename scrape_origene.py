#!/usr/bin/env python3
"""
scrape_origene.py

Scrapes a larger batch (500) of validated qPCR primer pairs from OriGene's sitemap,
integrates them into the master training database, and triggers biophysical auditing.
"""

import os
import sys
import re
import time
import datetime
import requests
import pandas as pd
import numpy as np
import concurrent.futures
from bs4 import BeautifulSoup
from typing import Dict, Any, List

SITEMAP_URL = "https://www.origene.com/media/sitemap/sitemap-1-55.xml"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
TARGET_COUNT = 500

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

def get_sitemap_urls(target_count: int) -> List[str]:
    """Fetches the sitemap and extracts up to target_count qPCR product URLs."""
    print(f"Fetching sitemap from: {SITEMAP_URL}...", flush=True)
    try:
        response = requests.get(SITEMAP_URL, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} for sitemap.", file=sys.stderr)
            return []
            
        # Parse XML locations using fast regex matching
        urls = re.findall(r'<loc>(https://www.origene.com/catalog/gene-expression/qpcr-primer-pairs/[^<]+)</loc>', response.text)
        
        # Exclude generic landing page if matched
        urls = [url for url in urls if not url.endswith("qpcr-primer-pairs")]
        
        selected_urls = urls[:target_count]
        print(f"Total qPCR primer URLs found in sitemap: {len(urls)}. Selected first {len(selected_urls)} for ingestion.")
        return selected_urls
    except Exception as e:
        print(f"Exception occurred while parsing sitemap: {e}", file=sys.stderr)
        return []

def scrape_product_page(url: str) -> Dict[str, Any]:
    """Scrapes sequences and metadata from a specific OriGene product page."""
    parts = url.split("/")
    last_part = parts[-1]
    sku_match = re.match(r"^([hm]p[0-9]+)", last_part, re.IGNORECASE)
    if not sku_match:
        print(f"Could not parse SKU from URL: {url}", file=sys.stderr)
        return None
    sku = sku_match.group(1).upper()
    
    # Introduce a small polite delay per worker
    time.sleep(0.8)
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None
            
        if "primer-primer" in response.url:
            return None
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 1. Parse Sequences and attributes
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
            return None
            
        fwd_seq = str(fwd_seq).upper().strip()
        rev_seq = str(rev_seq).upper().strip()
        
        # 2. Parse Gene Name and Species
        gene_name = "UNKNOWN"
        organism = "human" if sku.startswith("HP") else "mouse"
        
        h1 = soup.find("h1")
        if h1:
            h1_text = h1.text.strip()
            match = re.match(r"^([A-Za-z0-9_\-\(\)]+)\s+([A-Za-z]+)\s+qPCR\s+Primer\s+Pair", h1_text, re.IGNORECASE)
            if match:
                gene_name = match.group(1).upper()
                organism = match.group(2).lower()
            else:
                first_word = h1_text.split()[0]
                gene_name = first_word.replace("Human", "").replace("Mouse", "").upper()
                if "human" in h1_text.lower():
                    organism = "human"
                elif "mouse" in h1_text.lower():
                    organism = "mouse"
                    
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            desc_val = meta_desc.get("content", "").lower()
            if "homo sapiens" in desc_val:
                organism = "human"
            elif "mus musculus" in desc_val:
                organism = "mouse"
                
        if organism == "mouse":
            organism = "mus musculus"
            
        record = {
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
        return record
    except Exception as e:
        print(f"  Error parsing {sku}: {e}", file=sys.stderr)
        return None

def main():
    print("="*60)
    print(f"STARTING BULK ORIGENE CRAWLER (TARGET: {TARGET_COUNT} PRIMER PAIRS)")
    print("="*60)
    
    # 1. Fetch URLs
    urls = get_sitemap_urls(TARGET_COUNT)
    if not urls:
        print("Error: No URLs extracted from sitemap.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Crawl and parse product URLs concurrently (using 3 threads to be fast yet polite)
    scraped_records = []
    success_count = 0
    
    print("\nCrawling product pages concurrently with 3 workers...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_url = {executor.submit(scrape_product_page, url): url for url in urls}
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_url)):
            url = future_to_url[future]
            try:
                record = future.result()
                if record:
                    scraped_records.append(record)
                    success_count += 1
            except Exception as e:
                pass
            if (idx + 1) % 50 == 0 or idx + 1 == len(urls):
                print(f"  Processed {idx + 1}/{len(urls)} pages...", flush=True)
                
    print(f"\nCrawling complete. Successfully scraped {success_count} of {len(urls)} URLs.")
    
    if not scraped_records:
        print("Error: No records parsed. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    # 3. Append to Master Training Database
    db_path = "data/master_training_db_v2.csv"
    if not os.path.exists(db_path):
        print(f"Error: Master training database not found at: {db_path}", file=sys.stderr)
        sys.exit(1)
        
    df_db = pd.read_csv(db_path)
    
    # Remove existing OriGene entries to keep script idempotent and allow full refresh
    df_db = df_db[df_db['source_db'].str.lower() != 'origene'].copy()
    print(f"Existing database size (excluding previous OriGene entries): {len(df_db)}")
    
    existing_seqs = set(df_db['sequence_fwd'].astype(str).str.upper() + "_" + df_db['sequence_rev'].astype(str).str.upper())
    
    new_records = []
    for rec in scraped_records:
        seq_key = f"{rec['sequence_fwd']}_{rec['sequence_rev']}"
        if seq_key in existing_seqs:
            pass # Skip duplicates silently to avoid log spamming
        else:
            new_records.append(rec)
            
    if not new_records:
        print("All scraped records are already present in the database. No updates needed.")
        sys.exit(0)
        
    df_new = pd.DataFrame(new_records)
    df_new = df_new.reindex(columns=FINAL_COLUMNS)
    
    df_updated = pd.concat([df_db, df_new], ignore_index=True)
    df_updated.to_csv(db_path, index=False)
    print(f"\nAppended {len(new_records)} new unique OriGene records to: {db_path}")
    print(f"New Database Size: N={len(df_updated)} records.")
    
    # 4. Trigger dataset audit to compute biophysical features and update statistics report
    print("\nTriggering dataset audit backfilling and stats regeneration...", flush=True)
    os.system("python dataset_audit.py")
    
    print("\nOriGene Ingestion Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
