import numpy as np
from primerforge.biophysics import BiophysicsEngine, PrimerSequence, PrimerPair
from primerforge.multiplex import MultiplexOptimizer, MultiplexPanel

def verify_multiplex_panel_optimization():
    print("=" * 80)
    print("PRIMERFORGE: STEP 3 MULTIPLEX OPTIMIZATION LAYER VERIFICATION REPORT")
    print("=" * 80)
    
    optimizer = MultiplexOptimizer()

    print("Synthesizing diverse candidate pools for 3 distinct target loci...")
    
    # ── LOCUS 0: 1 baseline designed pair
    f0 = PrimerSequence("CACCATTGGCAATGAGCGGT", 0, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    r0 = PrimerSequence("CGCTCAGGAGGAGCAATGAT", 100, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    pair0 = PrimerPair(f0, r0, 150, -0.5, 0.0)
    pool0 = [pair0]
    
    # ── LOCUS 1: 2 candidate pairs
    # 1.1 Candidate Bad: Pairs perfectly with Locus 0 F to form a stable primer-dimer
    f1_bad = PrimerSequence("ACCGCTCATTGCCAATGGTG", 0, 20, 60.0, 50.0, -0.2, -0.3, 0.0)  # perfect complement of f0
    r1_bad = PrimerSequence("GCATGGAGTCCTGTGGCATC", 100, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    pair1_bad = PrimerPair(f1_bad, r1_bad, 120, -12.0, 0.0)  # Simulated extremely severe dimerization (-12.0 kcal/mol)
    
    # 1.2 Candidate Good: Highly compatible standard primers
    f1_good = PrimerSequence("AAGACCTGTACGCCAACACA", 0, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    r1_good = PrimerSequence("GCATGGAGTCCTGTGGCATC", 100, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    pair1_good = PrimerPair(f1_good, r1_good, 120, -0.5, 0.0)
    
    pool1 = [pair1_bad, pair1_good]

    # ── LOCUS 2: 2 candidate pairs
    # 2.1 Candidate Good: Highly compatible
    f2_good = PrimerSequence("TGGCATCCACGAAACTACCT", 0, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    r2_good = PrimerSequence("CATCATGAAGTGTGACGTGG", 100, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    pair2_good = PrimerPair(f2_good, r2_good, 140, -0.4, 0.0)
    
    # 2.2 Candidate Bad: Pairs perfectly with Locus 1 Good F to form a stable primer-dimer
    f2_bad = PrimerSequence("TGTGTTGGCGTACAGGTCTT", 0, 20, 60.0, 50.0, -0.2, -0.3, 0.0)  # complement of f1_good
    r2_bad = PrimerSequence("CATCATGAAGTGTGACGTGG", 100, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    pair2_bad = PrimerPair(f2_bad, r2_bad, 140, -11.0, 0.0)
    
    pool2 = [pair2_bad, pair2_good]

    print("\nExecuting greedy multiplex panel design with automated cross-reactivity rescue...")
    panel = optimizer.design_compatible_panel(
        candidate_pools=[pool0, pool1, pool2],
        threshold=-6.0,
        hard_limit=-9.0
    )
    
    print("\n" + "=" * 50)
    print("OPTIMIZED PANEL DETAILS:")
    print("=" * 50)
    print(f"Total Selected Primer Pairs: {len(panel.pairs)}")
    for idx, pair in enumerate(panel.pairs):
        print(f"  Locus {idx} Forward: {pair.forward.sequence}")
        print(f"  Locus {idx} Reverse: {pair.reverse.sequence}")
    
    print(f"\nGlobal Cross-Reactivity Penalty: {panel.global_penalty:.4f} kcal/mol")
    
    print("\nDimerization Matrix D (2M x 2M) (in kcal/mol):")
    # Print the dimerization matrix nicely aligned
    np.set_printoptions(precision=2, suppress=True)
    
    header = "          " + "".join(f"{lbl:<12}" for lbl in panel.primer_labels)
    print(header)
    for i in range(len(panel.primer_labels)):
        row_lbl = f"{panel.primer_labels[i]:<10}"
        row_vals = "".join(f"{panel.dimerization_matrix[i, k]:<12.2f}" for k in range(len(panel.primer_labels)))
        print(row_lbl + row_vals)
        
    # Verify that the rejected bad primers are indeed not in the panel
    f_seqs = [p.forward.sequence for p in panel.pairs]
    assert "ACCGCTCATTGCCAATGGTG" not in f_seqs, "Failed to filter out Locus 1 bad primer!"
    assert "TGTGTTGGCGTACAGGTCTT" not in f_seqs, "Failed to filter out Locus 2 bad primer!"
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE: MULTIPLEX CROSS-REACTIVITY SUCCESSFULLY OPTIMIZED & RESCUED")
    print("=" * 80)

if __name__ == "__main__":
    verify_multiplex_panel_optimization()
