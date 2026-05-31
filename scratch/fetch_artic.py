import urllib.request
import urllib.error
import csv
import os
import re

def fetch_artic():
    # Try the user-provided URL first, fallback to the working one
    urls = [
        "https://raw.githubusercontent.com/artic-network/artic-ncov2019/master/primer_schemes/nCoV-2019/V4/SARS-CoV-2.primer.bed",
        "https://raw.githubusercontent.com/artic-network/artic-ncov2019/master/primer_schemes/nCoV-2019/V4/nCoV-2019.primer.bed"
    ]
    
    bed_content = None
    for url in urls:
        print(f"Trying to fetch from {url}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                bed_content = resp.read().decode('utf-8')
                print("Successfully fetched BED file.")
                break
        except Exception as e:
            print(f"Failed to fetch from {url}: {e}")
            
    if not bed_content:
        raise RuntimeError("Failed to download ARTIC BED scheme from all URLs.")
        
    primers = {} # name -> {seq, pool}
    for line in bed_content.strip().split('\n'):
        if not line or line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) >= 7:
            name = parts[3]
            pool = parts[4]
            seq = parts[6].upper().strip()
            primers[name] = {"seq": seq, "pool": pool}
            
    # Pair LEFT and RIGHT primers
    # Primer names look like: SARS-CoV-2_1_LEFT, SARS-CoV-2_1_RIGHT, SARS-CoV-2_1_LEFT_alt, etc.
    # Group them by base amplicon number
    base_names = set()
    for name in primers:
        base = name.replace("_LEFT", "").replace("_RIGHT", "").replace("_alt", "")
        base_names.add(base)
        
    # Omicron and general dropouts in V4
    dropout_amplicons = {"10", "23", "76", "79", "88", "89", "90"}
    
    pairs = []
    for base in sorted(base_names):
        # Find amplicon number from base name (e.g. SARS-CoV-2_76 -> 76)
        amp_match = re.search(r'_(\d+)$', base)
        amp_num = amp_match.group(1) if amp_match else ""
        
        lefts = [n for n in primers if n.startswith(base) and "_LEFT" in n]
        rights = [n for n in primers if n.startswith(base) and "_RIGHT" in n]
        
        for l_name in lefts:
            for r_name in rights:
                l_info = primers[l_name]
                r_info = primers[r_name]
                
                # Check if it's a dropout
                is_dropout = amp_num in dropout_amplicons
                
                # If Pool 1 primers: success=1
                # If dropout: success=0
                # Otherwise: success=1
                success = 0 if is_dropout else 1
                
                pairs.append({
                    "forward_seq": l_info["seq"],
                    "reverse_seq": r_info["seq"],
                    "success": success,
                    "source": "artic_v4",
                    "gene": base
                })
                
    os.makedirs("data", exist_ok=True)
    out_path = "data/artic_v4_primers.csv"
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["forward_seq", "reverse_seq", "success", "source", "gene"])
        writer.writeheader()
        writer.writerows(pairs)
        
    print(f"Saved {len(pairs)} ARTIC primer pairs to {out_path}")
    print(f"Dropout pairs (success=0): {sum(1 for p in pairs if p['success'] == 0)}")
    print(f"Success pairs (success=1): {sum(1 for p in pairs if p['success'] == 1)}")

if __name__ == "__main__":
    fetch_artic()
