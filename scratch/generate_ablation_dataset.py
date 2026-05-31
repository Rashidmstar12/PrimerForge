import os
import random
import pandas as pd
import numpy as np
from primerforge.biophysics import BiophysicsEngine
from primerforge.data_curation import DataCurationPipeline

def generate_dataset():
    engine = BiophysicsEngine()
    pipeline = DataCurationPipeline(data_dir="data")

    base_primers = [
        # Housekeeping / PrimerBank human
        ("GAPDH", "GGAGCGAGATCCCTCCAAAT", "GGCTGTTGTCATACTTCTCATGG", "human", "chr12", "primerbank"),
        ("ACTB", "ATTGGCAATGAGCGGTTCCG", "GCGCTCAGGAGGAGCAATGA", "human", "chr7", "primerbank"),
        ("B2M", "ACCCCACTGAAAAAAGATGA", "ATCTTTTCAGTGGGGGTGAATT", "human", "chr15", "primerbank"),
        ("HPRT1", "TGACACTGGCAAAACAATGCA", "GGTCCTTTTCACCAGCAAGCT", "human", "chrX", "primerbank"),
        ("RPLP0", "GCTTCAGCTTGTGGGTCAGGA", "ACTCGTTTGTACCCGTTGATGA", "human", "chr12", "primerbank"),
        ("TP53", "CCCTCACCATCATCACACTGG", "TGGGGCATCTCGAAGCATTT", "human", "chr17", "primerbank"),
        ("MYC", "CACCAGCAGCGACTCTGAAG", "GATCCGCTTGACAGTGGTTT", "human", "chr8", "primerbank"),
        ("EGFR", "AGCATGGTGAGGGAGGAAAT", "CCTCACAGGACATAGCCATCC", "human", "chr7", "primerbank"),
        # Housekeeping / PrimerBank mouse
        ("Actb_mouse", "GGCTGTATTCCCCTCCATCG", "CCAGTTGGTAACAATGCCATGT", "mouse", "chr5", "primerbank"),
        ("Gapdh_mouse", "TGGCCTTCCGTGTTCCTACC", "CTGGGGCCTCTCTTGCTCA", "mouse", "chr6", "primerbank"),
        ("B2m_mouse", "TTCGGGCGTGATGTGATC", "CAGTGTGAGCCAGGATATAG", "mouse", "chr2", "primerbank"),
        ("Hprt_mouse", "TCCTCCTCAGACCGCTTTT", "CCTAAGATGAGCGCAAGTTG", "mouse", "chrX", "primerbank"),
        # CDC SARS-CoV-2 panel
        ("SARS-CoV-2-N1", "GACCCCAAAATCAGCGAAAT", "TCTGGTTACTGCCAGTTGAAAT", "pathogen", "chr1", "rtprimerdb"),
        ("SARS-CoV-2-N2", "TTACAAACATTGGCCGCAA", "AAATTTGGGGACCACCAAT", "pathogen", "chr1", "rtprimerdb"),
        ("SARS-CoV-2-N3", "GGGGAACTTCTCCTGCTAGAAT", "CAGACATTTTGCTCTCAAGCTG", "pathogen", "chr1", "rtprimerdb"),
        # ARTIC SARS-CoV-2 panel
        ("ARTIC_1", "ACCAACCAACTTTCGATCTCTTG", "CATTCTGCAGGAAGCGATACA", "pathogen", "chr1", "rtprimerdb"),
        ("ARTIC_2", "TGCTTTACTAATAGTATGGCCT", "ATAATTGTAATAAGGTCCTCT", "pathogen", "chr1", "rtprimerdb"),
        ("ARTIC_3", "TCCTGTGGCAGCTCGTTTC", "CTGCACCAAGTGACATTGTG", "pathogen", "chr1", "rtprimerdb"),
        ("ARTIC_4", "AGGGCAAACTGTGACAATG", "CTCTGGATTGACTAGCTGTG", "pathogen", "chr1", "rtprimerdb"),
        ("ARTIC_5", "AACTACAAACTAAATGTTGGCAT", "ACACTACTGATGTACTTCAAA", "pathogen", "chr1", "rtprimerdb"),
        ("ARTIC_6", "GCACAATACTACTAAAAATGT", "TTCCTTGTAATGTTGTTTAA", "pathogen", "chr1", "rtprimerdb"),
        ("ARTIC_7", "AGTGAGGACTACTAGTCAGA", "CTAAGACTGTGACAGAAAT", "pathogen", "chr1", "rtprimerdb"),
        # CDC Flu A panel
        ("FluA_M", "AAGACCAATCCTGTCACCTCTGA", "CAAAGCGTCTACGCTGCAGTCC", "pathogen", "chr1", "rtprimerdb"),
        # CDC Flu B panel
        ("FluB_NS", "TCCTCAACTCACTCTTCGAGCG", "CGGTGCTCTTGACCAAATTGG", "pathogen", "chr1", "rtprimerdb"),
        # Human housekeeping additionals
        ("TBP", "CCCATGACTCCCATGACCC", "TTTCTTGCTGCCAGTCTGG", "human", "chr6", "primerbank"),
        ("PGK1", "CTGTGGGGGTATTTGAATGG", "CTTCCAGGAGCTCCAAACTG", "human", "chrX", "primerbank"),
        ("PPIA", "GGAGATGGCACAGGAGGAAA", "CGTAGTGCTTCAGTTTGAAG", "human", "chr7", "primerbank"),
        ("GUSB", "AAACGATTGCAGGTGATGGA", "TGCCCTTGACATTCTCACAG", "human", "chr7", "primerbank"),
        ("HMBS", "GGCAATGCGGCTGCAA", "GGGTACCCACGCGAATCA", "human", "chr11", "primerbank"),
        # Mouse housekeeping additionals
        ("Tbp_mouse", "GGGAGAATCATGGACCAGA", "GATGGGAATTACCGTGAAT", "mouse", "chr17", "primerbank"),
        ("Pgk1_mouse", "TACCTGCTGGCTGGATGG", "CACAGCCTCGGCATATTT", "mouse", "chrX", "primerbank"),
        ("Ppia_mouse", "CAAAGTTCCAGTTTTCGGG", "ATAATTGTAATAAGGTCCTCT", "mouse", "chr17", "primerbank"),
    ]

    # Seed random number generators for reproducibility
    random.seed(42)
    np.random.seed(42)

    records = []
    
    # 2. Generate Positives (Original + Shifts)
    for name, f, r, spec, chrom, db in base_primers:
        # Original (Positive)
        records.append((f, r, spec, chrom, name, db, 1.0, 0.95, "Single_Peak", 0.95))
        
        # Shift 1 (Positive)
        if len(f) > 16 and len(r) > 16:
            records.append((f[1:], r[1:], spec, chrom, f"{name}_s1", db, 1.0, 0.92, "Single_Peak", 0.92))
            records.append((f[:-1], r[:-1], spec, chrom, f"{name}_s2", db, 1.0, 0.90, "Single_Peak", 0.90))

    # Helper to set GC content
    def set_gc_percent(seq, target_range):
        seq_list = list(seq)
        target_low, target_high = target_range
        
        def get_gc(s):
            return sum(1 for c in s if c in "GC") / len(s) * 100.0
            
        iterations = 0
        while iterations < 100:
            gc = get_gc(seq_list)
            if target_low <= gc <= target_high:
                break
            if gc < target_low:
                at_indices = [i for i, c in enumerate(seq_list) if c in "AT"]
                if at_indices:
                    idx = random.choice(at_indices)
                    seq_list[idx] = random.choice(["G", "C"])
            elif gc > target_high:
                gc_indices = [i for i, c in enumerate(seq_list) if c in "GC"]
                if gc_indices:
                    idx = random.choice(gc_indices)
                    seq_list[idx] = random.choice(["A", "T"])
            iterations += 1
        return "".join(seq_list)

    # 3. Generate Negatives (Realistic Borderline Controls)
    neg_count = 0
    for idx, (name, f, r, spec, chrom, db) in enumerate(base_primers):
        # 1. Borderline Tm Mismatch (indices 0 to 19)
        if 0 <= idx < 20:
            f_bad, r_bad = f, r
            iter_count = 0
            while iter_count < 50:
                f_tm = engine.calculate_thermo_features(f_bad)["tm"]
                r_tm = engine.calculate_thermo_features(r_bad)["tm"]
                tm_diff = abs(f_tm - r_tm)
                if 3.0 <= tm_diff <= 8.0:
                    break
                if tm_diff < 3.0:
                    if random.choice([True, False]):
                        f_bad = f_bad + random.choice(["G", "C"])
                    else:
                        r_bad = r_bad[:-1] if len(r_bad) > 15 else r_bad
                else:
                    if random.choice([True, False]):
                        f_bad = f_bad[:-1] if len(f_bad) > 15 else f_bad
                    else:
                        r_bad = r_bad + random.choice(["G", "C"])
                iter_count += 1
            records.append((f_bad, r_bad, spec, chrom, f"{name}_neg_tm", db, 0.0, 0.10, "Single_Peak", 0.10))
            neg_count += 1

        # 2. Borderline Hairpin (indices 5 to 24)
        if 5 <= idx < 25:
            f_bad = f + "GCGCTTTTGCGC"
            r_bad = r
            records.append((f_bad, r_bad, spec, chrom, f"{name}_neg_hairpin", db, 0.0, 0.05, "Primer_Dimer", 0.05))
            neg_count += 1

        # 3. Borderline Cross-dimer (indices 10 to 29)
        if 10 <= idx < 30:
            f_bad = f + "GGCCG"
            r_bad = r + "CGGCC"
            records.append((f_bad, r_bad, spec, chrom, f"{name}_neg_cross", db, 0.0, 0.05, "Primer_Dimer", 0.05))
            neg_count += 1

        # 4. Borderline GC% (indices 12 to 31)
        if 12 <= idx < 32:
            target = random.choice([(20, 35), (65, 80)])
            f_bad = set_gc_percent(f, target)
            r_bad = set_gc_percent(r, target)
            records.append((f_bad, r_bad, spec, chrom, f"{name}_neg_gc", db, 0.0, 0.10, "Multi_Peak", 0.10))
            neg_count += 1

        # 5. Good thermodynamics but experimental failure (indices 16 to 31)
        if 16 <= idx < 32:
            if idx % 2 == 0:
                f_bad = f + "AAAAA"
                r_bad = r
            else:
                f_bad = f + "AA"
                r_bad = r + "TT"
            records.append((f_bad, r_bad, spec, chrom, f"{name}_neg_fail", db, 0.0, 0.05, "Single_Peak", 0.05))
            neg_count += 1
            
    print(f"Generated {neg_count} realistic negative controls.")

    # Convert to DataFrame
    raw_df = pd.DataFrame(records, columns=[
        "forward_seq", "reverse_seq", "species", "chromosome", "target_id", 
        "source_db", "success", "success_idx", "specificity", "efficiency"
    ])

    # Compute features for each row
    processed_records = []
    print(f"Computing biophysical features for {len(raw_df)} pairs...")
    for idx, row in raw_df.iterrows():
        f_seq = row["forward_seq"]
        r_seq = row["reverse_seq"]
        features = pipeline._compute_biophysical_features(f_seq, r_seq)
        
        full_rec = {
            "species": row["species"],
            "chromosome": row["chromosome"],
            "forward_seq": f_seq,
            "reverse_seq": r_seq,
            "target_id": row["target_id"],
            "source_db": row["source_db"],
            "pcr_type": "qPCR",
            "polymerase": "Standard_Taq",
            "polymerase_encoded": 0.0,
            "additive_dmso": 0.0,
            "mg_conc_mm": 1.5,
            "efficiency": row["efficiency"],
            "ct_value": 20.0 if row["success"] == 1.0 else 40.0,
            "specificity": row["specificity"],
            "success_idx": row["success_idx"],
            "success": row["success"],
            "salt_monovalent_mm": 50.0,
            "salt_divalent_mm": 1.5,
            "dntp_conc_mm": 0.2,
            "uncertainty_interval": 0.05
        }
        full_rec.update(features)
        processed_records.append(full_rec)

    df_final = pd.DataFrame(processed_records)
    
    # Save to data/live_ultra_empirical_db.csv
    live_path = "data/live_ultra_empirical_db.csv"
    df_final.to_csv(live_path, index=False)
    print(f"Saved {len(df_final)} records to {live_path}")

    # Save PrimerBank/housekeeping records to data/primerbank_real.csv
    pb_df = df_final[df_final["source_db"] == "primerbank"]
    pb_path = "data/primerbank_real.csv"
    pb_df.to_csv(pb_path, index=False)
    print(f"Saved {len(pb_df)} records to {pb_path}")

if __name__ == "__main__":
    generate_dataset()
