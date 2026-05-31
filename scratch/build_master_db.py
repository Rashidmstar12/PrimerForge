import pandas as pd
import os

def build_master_db():
    p_pb = "data/primerbank_full.csv"
    p_artic = "data/artic_v4_primers.csv"
    p_rt = "data/rtprimerdb.csv"
    p_lab = "data/lab_validation_primers.csv"
    p_live = "data/live_ultra_empirical_db.csv"
    
    dfs = []
    
    # 1. PrimerBank
    if os.path.exists(p_pb):
        df = pd.read_csv(p_pb)
        df_std = pd.DataFrame({
            "forward_seq": df["forward_seq"],
            "reverse_seq": df["reverse_seq"],
            "success": df["success"],
            "source": "primerbank",
            "gene": df["gene_name"]
        })
        dfs.append(df_std)
        print(f"Loaded {len(df_std)} from {p_pb}")
        
    # 2. ARTIC
    if os.path.exists(p_artic):
        df = pd.read_csv(p_artic)
        df_std = pd.DataFrame({
            "forward_seq": df["forward_seq"],
            "reverse_seq": df["reverse_seq"],
            "success": df["success"],
            "source": df["source"],
            "gene": df["gene"]
        })
        dfs.append(df_std)
        print(f"Loaded {len(df_std)} from {p_artic}")
        
    # 3. RTprimerDB
    if os.path.exists(p_rt):
        df = pd.read_csv(p_rt)
        df_std = pd.DataFrame({
            "forward_seq": df["forward_seq"],
            "reverse_seq": df["reverse_seq"],
            "success": df["success"],
            "source": df["source"],
            "gene": df["gene"]
        })
        dfs.append(df_std)
        print(f"Loaded {len(df_std)} from {p_rt}")
        
    # 4. Lab validation
    if os.path.exists(p_lab):
        df = pd.read_csv(p_lab)
        df_std = pd.DataFrame({
            "forward_seq": df["forward_seq"],
            "reverse_seq": df["reverse_seq"],
            "success": df["validated_success"],
            "source": "lab_validation_" + df["source"].astype(str),
            "gene": df["gene"]
        })
        dfs.append(df_std)
        print(f"Loaded {len(df_std)} from {p_lab}")
        
    # 5. Live Ultra Empirical DB
    if os.path.exists(p_live):
        df = pd.read_csv(p_live)
        df_std = pd.DataFrame({
            "forward_seq": df["forward_seq"],
            "reverse_seq": df["reverse_seq"],
            "success": df["success"],
            "source": "live_ultra_" + df["source_db"].fillna("empirical").astype(str),
            "gene": df["target_id"].fillna("unknown")
        })
        dfs.append(df_std)
        print(f"Loaded {len(df_std)} from {p_live}")
        
    if not dfs:
        raise ValueError("No datasets loaded. Check your data paths.")
        
    master_df = pd.concat(dfs, ignore_index=True)
    
    # Standardize sequences and types
    master_df["forward_seq"] = master_df["forward_seq"].astype(str).str.upper().str.strip()
    master_df["reverse_seq"] = master_df["reverse_seq"].astype(str).str.upper().str.strip()
    master_df["success"] = master_df["success"].astype(float).round().astype(int)
    master_df["source"] = master_df["source"].astype(str).str.strip()
    master_df["gene"] = master_df["gene"].astype(str).str.strip()
    
    initial_len = len(master_df)
    
    # Filter out N bases
    master_df = master_df[~master_df["forward_seq"].str.contains("N", na=False)]
    master_df = master_df[~master_df["reverse_seq"].str.contains("N", na=False)]
    
    # Filter length bounds: 15bp to 35bp
    master_df = master_df[
        (master_df["forward_seq"].str.len() >= 15) & (master_df["forward_seq"].str.len() <= 35) &
        (master_df["reverse_seq"].str.len() >= 15) & (master_df["reverse_seq"].str.len() <= 35)
    ]
    
    # Deduplicate on (forward_seq, reverse_seq)
    master_df = master_df.sort_values(by="success", ascending=False)
    master_df = master_df.drop_duplicates(subset=["forward_seq", "reverse_seq"], keep="first")
    
    final_len = len(master_df)
    
    # Save
    out_path = "data/master_training_db.csv"
    master_df.to_csv(out_path, index=False)
    
    # Prints
    print("=" * 40)
    print("MASTER DATABASE BUILD SUMMARY")
    print("=" * 40)
    print(f"Initial raw samples loaded: {initial_len}")
    print(f"Final clean samples saved: {final_len}")
    print(f"Removed / Deduplicated: {initial_len - final_len}")
    print("-" * 40)
    
    pos_count = sum(master_df["success"] == 1)
    neg_count = sum(master_df["success"] == 0)
    print(f"Positive Count (success=1): {pos_count} ({pos_count/final_len*100:.1f}%)")
    print(f"Negative Count (success=0): {neg_count} ({neg_count/final_len*100:.1f}%)")
    print("-" * 40)
    
    print("Source Breakdown:")
    breakdown = master_df["source"].value_counts()
    for src, count in breakdown.items():
        print(f"  - {src}: {count}")
    print("=" * 40)

if __name__ == "__main__":
    build_master_db()
