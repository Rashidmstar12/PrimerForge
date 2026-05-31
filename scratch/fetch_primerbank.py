import urllib.request
import urllib.parse
import re
import csv
import os
import time

def fetch_primerbank():
    url = 'https://pga.mgh.harvard.edu/cgi-bin/primerbank/new_search2.cgi'
    keywords = [
        "kinase", "receptor", "growth", "binding", "zinc", "homeobox", "membrane",
        "transcription", "factor", "transport", "signal", "nucle", "ribos", "mitoch",
        "immun", "synthase", "dehydrogenase", "phosphatase", "methyl", "acetyl",
        "cell", "histone", "oncogene", "tumor", "housekeeping", "interleukin", "cluster"
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    unique_primers = {} # key: (f_seq, r_seq)
    
    print(f"Starting PrimerBank fetch across {len(keywords)} keywords...")
    
    for kw in keywords:
        print(f"Querying keyword: {kw} ...")
        post_data = urllib.parse.urlencode({
            'selectBox': 'Keyword',
            'species': 'Human',
            'searchBox': kw,
            'Submit': 'Submit'
        }).encode('utf-8')
        
        try:
            req = urllib.request.Request(url, data=post_data, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Error querying {kw}: {e}")
            continue
            
        # Split by individual gene sections
        genes = html.split('Gene Descriptions:')
        if len(genes) <= 1:
            print(f"No matches found for keyword: {kw}")
            continue
            
        print(f"Found {len(genes) - 1} gene matches for {kw}.")
        
        for gene_sec in genes[1:]:
            # Parse accession
            acc_match = re.search(r'term=([A-Z0-9_]+)"', gene_sec)
            accession = acc_match.group(1) if acc_match else "unknown"
            
            # Parse gene description and extract symbol
            desc_match = re.search(r'Gene Description</b>\s*</td>\s*<td[^>]*>(.*?)</td>', gene_sec, re.DOTALL | re.IGNORECASE)
            desc = desc_match.group(1).strip() if desc_match else ""
            
            gene_name = "unknown"
            if desc:
                sym_match = re.search(r'\(([^)]+)\)', desc)
                if sym_match:
                    gene_name = sym_match.group(1)
                else:
                    # Fallback to description first word
                    gene_name = desc.split()[0]
            
            # Split by primer pairs
            pair_sections = gene_sec.split('PrimerBank ID')
            for pair_sec in pair_sections[1:]:
                # Parse Amplicon Size
                size_match = re.search(r'Amplicon Size</b>\s*</td>\s*<td[^>]*>(\d+)</td>', pair_sec, re.DOTALL | re.IGNORECASE)
                amplicon_size = size_match.group(1) if size_match else "unknown"
                
                # Parse sequences
                f_match = re.search(r'Forward Primer</td>\s*<td[^>]*><font[^>]*>([A-Z]+)</font>', pair_sec, re.DOTALL | re.IGNORECASE)
                r_match = re.search(r'Reverse Primer</td>\s*<td[^>]*><font[^>]*>([A-Z]+)</font>', pair_sec, re.DOTALL | re.IGNORECASE)
                
                if f_match and r_match:
                    f_seq = f_match.group(1).upper().strip()
                    r_seq = r_match.group(1).upper().strip()
                    
                    # Deduplicate and save
                    key = (f_seq, r_seq)
                    if key not in unique_primers:
                        unique_primers[key] = {
                            "gene_name": gene_name,
                            "forward_seq": f_seq,
                            "reverse_seq": r_seq,
                            "amplicon_size": amplicon_size,
                            "accession": accession,
                            "success": 1
                        }
        
        print(f"Total unique primers collected so far: {len(unique_primers)}")
        # Gentle rate limiting
        time.sleep(1)

    # Save to data/primerbank_full.csv
    os.makedirs("data", exist_ok=True)
    out_path = "data/primerbank_full.csv"
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gene_name", "forward_seq", "reverse_seq", "amplicon_size", "accession", "success"])
        writer.writeheader()
        for p in unique_primers.values():
            writer.writerow(p)
            
    print(f"Saved {len(unique_primers)} unique PrimerBank primers to {out_path}")

if __name__ == "__main__":
    fetch_primerbank()
