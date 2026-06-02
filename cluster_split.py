"""
Cluster-Aware Stratified Train/Test Dataset Splitter for PrimerForge.
Prevents homology sequence leakage across splits by grouping highly similar primer pairs 
(>=80% identity) into clusters before performing a stratified 80/20 partitioning.
"""

import os
import sys
import subprocess
import shutil
import tempfile
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Set stdout to handle UTF-8 printing safely on Windows
sys.stdout.reconfigure(encoding='utf-8')

def ensure_dataset_exists() -> str:
    """Checks if the target dataset exists, and if not, builds it from existing sources."""
    dataset_path = "data/primer_dataset.csv"
    if os.path.exists(dataset_path):
        return dataset_path

    print(f"'{dataset_path}' not found. Generating from master/empirical sources...")
    master_path = "data/master_training_db.csv"
    live_path = "data/live_ultra_empirical_db.csv"
    bank_path = "data/primerbank_real.csv"

    df_src = None
    if os.path.exists(live_path):
        df_src = pd.read_csv(live_path)
        print(f"Loaded {len(df_src)} rows from {live_path}")
        if os.path.exists(bank_path):
            df_bank = pd.read_csv(bank_path)
            df_src = pd.concat([df_src, df_bank], ignore_index=True)
            print(f"Merged with {len(df_bank)} rows from {bank_path}. Total: {len(df_src)}")
    elif os.path.exists(master_path):
        df_src = pd.read_csv(master_path)
        print(f"Loaded {len(df_src)} rows from {master_path}")

    if df_src is None:
        raise FileNotFoundError(
            "Could not locate any source data (data/live_ultra_empirical_db.csv or data/master_training_db.csv)"
        )

    # Standardize column mappings
    df_mapped = pd.DataFrame()
    
    if "forward_seq" in df_src.columns:
        df_mapped["sequence_fwd"] = df_src["forward_seq"]
    elif "f_seq" in df_src.columns:
        df_mapped["sequence_fwd"] = df_src["f_seq"]
    else:
        raise ValueError("Could not find forward sequence column in source data.")

    if "reverse_seq" in df_src.columns:
        df_mapped["sequence_rev"] = df_src["reverse_seq"]
    elif "r_seq" in df_src.columns:
        df_mapped["sequence_rev"] = df_src["r_seq"]
    else:
        raise ValueError("Could not find reverse sequence column in source data.")

    if "success" in df_src.columns:
        df_mapped["label"] = df_src["success"]
    elif "label" in df_src.columns:
        df_mapped["label"] = df_src["label"]
    else:
        # Default label to 1 if missing
        df_mapped["label"] = 1.0

    # Ensure label is binary (0 or 1)
    df_mapped["label"] = np.round(df_mapped["label"].astype(float)).astype(int)

    # Construct unique primer_ids
    if "gene" in df_src.columns:
        df_mapped["primer_id"] = df_src["gene"].astype(str) + "_" + df_src.index.astype(str)
    elif "primer_id" in df_src.columns:
        df_mapped["primer_id"] = df_src["primer_id"]
    else:
        df_mapped["primer_id"] = "p_" + df_src.index.astype(str)

    # Append any remaining feature columns
    excluded_cols = ["forward_seq", "f_seq", "reverse_seq", "r_seq", "success", "label", "gene", "primer_id"]
    for col in df_src.columns:
        if col not in excluded_cols:
            df_mapped[col] = df_src[col]

    # Deduplicate based on sequence pairs to avoid redundant data
    df_mapped.drop_duplicates(subset=["sequence_fwd", "sequence_rev"], inplace=True)

    os.makedirs("data", exist_ok=True)
    df_mapped.to_csv(dataset_path, index=False)
    print(f"Created standardized dataset '{dataset_path}' containing {len(df_mapped)} rows.")
    return dataset_path


def parse_cdhit_clstr(clstr_file: str) -> Dict[str, str]:
    """Parses a CD-HIT cluster (.clstr) output file to map sequence ID to cluster ID."""
    clusters = {}
    current_cluster = None
    with open(clstr_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = line[1:].strip() # e.g. "Cluster 0"
            elif line:
                # Format: 0	80nt, >gene_index... *
                parts = line.split(">")
                if len(parts) > 1:
                    seq_id = parts[1].split("...")[0].strip()
                    if current_cluster:
                        clusters[seq_id] = current_cluster
    return clusters


def parse_mmseqs_tsv(tsv_file: str) -> Dict[str, str]:
    """Parses a MMseqs2 cluster TSV output file to map sequence ID to cluster ID."""
    clusters = {}
    with open(tsv_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep_id, seq_id = parts[0].strip(), parts[1].strip()
                clusters[seq_id] = f"Cluster_{rep_id}"
    return clusters


def run_cdhit(fasta_path: str, output_prefix: str) -> Dict[str, str]:
    """Runs CD-HIT subprocess at 80% threshold."""
    cmd = ["cd-hit", "-i", fasta_path, "-o", output_prefix, "-c", "0.8", "-n", "5"]
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    clstr_file = f"{output_prefix}.clstr"
    if not os.path.exists(clstr_file):
        raise FileNotFoundError(f"CD-HIT execution completed but no cluster file '{clstr_file}' was found.")
    return parse_cdhit_clstr(clstr_file)


def run_mmseqs(fasta_path: str, output_prefix: str) -> Dict[str, str]:
    """Runs MMseqs2 easy-cluster subprocess at 80% threshold."""
    tmp_dir = os.path.join(os.path.dirname(fasta_path), "mmseqs_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    cmd = [
        "mmseqs", "easy-cluster", fasta_path, output_prefix, tmp_dir,
        "--min-seq-id", "0.8", "-c", "0.8", "--cov-mode", "0"
    ]
    print(f"Executing MMseqs2 Fallback: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    tsv_file = f"{output_prefix}_cluster.tsv"
    if not os.path.exists(tsv_file):
        raise FileNotFoundError(f"MMseqs2 execution completed but cluster file '{tsv_file}' was not found.")
    return parse_mmseqs_tsv(tsv_file)


def run_pure_python_clustering(df: pd.DataFrame, fwd_col: str, rev_col: str, id_col: str, threshold: float = 0.80) -> Dict[str, str]:
    """Pure-Python clustering fallback using SequenceMatcher ratios (threshold identity >= 80%)."""
    print(f"Executing pure-Python SequenceMatcher similarity clustering at threshold={threshold}...")
    from difflib import SequenceMatcher

    seqs = {}
    for _, row in df.iterrows():
        seq_id = str(row[id_col])
        seqs[seq_id] = str(row[fwd_col]) + str(row[rev_col])

    seq_ids = list(seqs.keys())
    adj = {sid: [] for sid in seq_ids}

    # Parallel pairwise checks with heuristic filter to avoid sequence comparison of different lengths
    n_seqs = len(seq_ids)
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            sid1, sid2 = seq_ids[i], seq_ids[j]
            s1, s2 = seqs[sid1], seqs[sid2]
            len1, len2 = len(s1), len(s2)
            
            # Simple length-ratio check to skip completely dissimilar pairs quickly
            if min(len1, len2) / max(len1, len2) < threshold:
                continue
                
            ratio = SequenceMatcher(None, s1, s2).ratio()
            if ratio >= threshold:
                adj[sid1].append(sid2)
                adj[sid2].append(sid1)

    # Connected Components (Breadth-First Search)
    visited = set()
    clusters = {}
    cluster_idx = 0

    for sid in seq_ids:
        if sid not in visited:
            component = []
            queue = [sid]
            visited.add(sid)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            for cid in component:
                clusters[cid] = f"Cluster_{cluster_idx}"
            cluster_idx += 1

    return clusters


def cluster_aware_split(df: pd.DataFrame, clusters: Dict[str, str], id_col: str, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits clusters using a greedy deviation-minimization bin-packing algorithm.
    
    Guarantees zero overlap of clusters between train and test sets while maintaining 80/20
    split and preserving stratified class label distribution.
    """
    df["cluster_id"] = df[id_col].map(clusters)
    
    # Assign unique fallback cluster IDs to any unmapped items
    unmapped_mask = df["cluster_id"].isna()
    if unmapped_mask.any():
        df.loc[unmapped_mask, "cluster_id"] = "unmapped_" + df.loc[unmapped_mask, id_col].astype(str)

    # Aggregate stats per cluster
    cluster_items = []
    for cid, group in df.groupby("cluster_id"):
        n_pos = int((group[label_col] == 1).sum())
        n_neg = int((group[label_col] == 0).sum())
        cluster_items.append({
            "cluster_id": cid,
            "size": len(group),
            "pos": n_pos,
            "neg": n_neg,
            "indices": list(group.index)
        })

    # Sort clusters by size in descending order
    cluster_items.sort(key=lambda x: x["size"], reverse=True)

    total_pos = sum(c["pos"] for c in cluster_items)
    total_neg = sum(c["neg"] for c in cluster_items)

    # Target test counts (20%)
    target_test_pos = 0.20 * total_pos
    target_test_neg = 0.20 * total_neg

    train_indices = []
    test_indices = []
    
    current_test_pos = 0
    current_test_neg = 0

    for c in cluster_items:
        # Determine whether adding this cluster to test yields a smaller absolute target deviation
        # Deviation calculation: |current - target| for both positive and negative classes
        dev_if_assigned_to_test = (
            abs((current_test_pos + c["pos"]) - target_test_pos) + 
            abs((current_test_neg + c["neg"]) - target_test_neg)
        )
        dev_if_assigned_to_train = (
            abs(current_test_pos - target_test_pos) + 
            abs(current_test_neg - target_test_neg)
        )

        # Greedy assignment to minimize distance to target counts
        if dev_if_assigned_to_test < dev_if_assigned_to_train:
            test_indices.extend(c["indices"])
            current_test_pos += c["pos"]
            current_test_neg += c["neg"]
        else:
            train_indices.extend(c["indices"])

    # Fallback safety: if either set is empty, force a manual assignment of smallest clusters
    if not test_indices and cluster_items:
        test_indices.extend(cluster_items[-1]["indices"])
        train_indices = [idx for idx in df.index if idx not in test_indices]

    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    return train_df, test_df


def verify_split_anti_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame, fwd_col: str, rev_col: str) -> float:
    """Verifies sequence leakage by computing maximum sequence identity overlap between train and test."""
    from difflib import SequenceMatcher
    train_seqs = (train_df[fwd_col].astype(str) + train_df[rev_col].astype(str)).tolist()
    test_seqs = (test_df[fwd_col].astype(str) + test_df[rev_col].astype(str)).tolist()

    max_identity = 0.0
    exact_matches = 0

    # Test sample of pairwise identities to keep check fast
    for test_s in test_seqs:
        for train_s in train_seqs:
            if test_s == train_s:
                exact_matches += 1
                max_identity = 1.0
            else:
                # Fast size check before calling sequence matcher
                if abs(len(test_s) - len(train_s)) / max(len(test_s), len(train_s)) > (1.0 - max_identity):
                    continue
                ratio = SequenceMatcher(None, test_s, train_s).ratio()
                if ratio > max_identity:
                    max_identity = ratio

    print(f"Overlap Check: Exact duplicate sequence pairs found: {exact_matches}")
    print(f"Overlap Check: Maximum pairwise sequence similarity: {max_identity * 100:.2f}%")
    return max_identity


def main():
    print("================================================================================")
    print("                PRIMERFORGE ANTI-LEAKAGE CLUSTER-SPLIT PIPELINE                 ")
    print("================================================================================")

    try:
        dataset_path = ensure_dataset_exists()
    except Exception as e:
        print(f"Error initializing input dataset: {e}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    
    # Auto-detect columns
    fwd_col = "sequence_fwd" if "sequence_fwd" in df.columns else "forward_seq"
    rev_col = "sequence_rev" if "sequence_rev" in df.columns else "reverse_seq"
    label_col = "label" if "label" in df.columns else "success"
    id_col = "primer_id" if "primer_id" in df.columns else "gene"
    if id_col not in df.columns:
        df["primer_id"] = "p_" + df.index.astype(str)
        id_col = "primer_id"

    print(f"Loaded {len(df)} rows. Standardized Columns: ID={id_col}, Fwd={fwd_col}, Rev={rev_col}, Label={label_col}")

    # Set up temp dir for subprocess runs
    temp_dir = tempfile.mkdtemp()
    fasta_path = os.path.join(temp_dir, "input_primers.fasta")
    output_prefix = os.path.join(temp_dir, "cdhit_res")

    try:
        # 1. Write to FASTA
        with open(fasta_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                seq_id = str(row[id_col])
                seq_val = str(row[fwd_col]) + str(row[rev_col])
                f.write(f">{seq_id}\n{seq_val}\n")

        # 2. Try CD-HIT -> MMseqs2 -> Python Fallback
        clusters = {}
        clustering_method = "Unknown"
        try:
            clusters = run_cdhit(fasta_path, output_prefix)
            clustering_method = "CD-HIT (Local Binary)"
        except (subprocess.SubprocessError, FileNotFoundError):
            print("CD-HIT not found or failed. Trying MMseqs2 easy-cluster fallback...")
            try:
                clusters = run_mmseqs(fasta_path, output_prefix)
                clustering_method = "MMseqs2 (Local Binary)"
            except (subprocess.SubprocessError, FileNotFoundError):
                print("MMseqs2 not found or failed. Using pure-Python clustering fallback...")
                clusters = run_pure_python_clustering(df, fwd_col, rev_col, id_col, threshold=0.80)
                clustering_method = "Pure-Python SequenceMatcher (80% ratio)"

        print(f"Clustering successfully completed via: {clustering_method}")
        n_clusters = len(set(clusters.values()))
        print(f"Identified {n_clusters} sequence homology clusters from {len(df)} sequences.")

        # 3. Cluster-Aware Stratified Train/Test Split
        train_df, test_df = cluster_aware_split(df, clusters, id_col, label_col)

        # 4. Save results
        train_path = "data/train_cluster_split.csv"
        test_path = "data/test_cluster_split.csv"
        
        # Cleanup split helper column before saving
        train_df_save = train_df.drop(columns=["cluster_id"], errors="ignore")
        test_df_save = test_df.drop(columns=["cluster_id"], errors="ignore")
        
        train_df_save.to_csv(train_path, index=False)
        test_df_save.to_csv(test_path, index=False)

        print("\n================================================================================")
        print("                           HOMOLOGY SPLIT REPORT                                ")
        print("================================================================================")
        print(f"Total Database Size   : {len(df)} pairs")
        print(f"Number of Clusters    : {n_clusters}")
        print(f"Training Split Size   : {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Testing Split Size    : {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        train_pos = int((train_df[label_col] == 1).sum())
        train_neg = len(train_df) - train_pos
        test_pos = int((test_df[label_col] == 1).sum())
        test_neg = len(test_df) - test_pos

        print(f"Train Label Balance   : functional={train_pos} ({train_pos/len(train_df)*100:.1f}%), non-functional={train_neg}")
        print(f"Test Label Balance    : functional={test_pos} ({test_pos/len(test_df)*100:.1f}%), non-functional={test_neg}")
        print("-" * 80)
        
        # Leakage Verification
        verify_split_anti_leakage(train_df, test_df, fwd_col, rev_col)
        print("================================================================================\n")
        print(f"Files saved successfully to:\n  - {train_path}\n  - {test_path}")

    finally:
        # Cleanup temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
