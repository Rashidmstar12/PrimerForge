"""
Step 2 (Gap-Filling) Quality Control Tests: Nussinov DP Amplicon MFE.

Verifies that AmpliconFolder / NussinovMFE correctly implements:
  - Nussinov & Jacobson (1980). PNAS 77(11), 6309-6313.
  - Turner & Mathews (2010). NAR 38, D280. doi:10.1093/nar/gkp892

Tests:
  1.  Poly-A (N=40) returns MFE ≈ 0.0 (no Watson-Crick pairs possible).
  2.  Perfect palindrome has deeply negative MFE (strong hairpin).
  3.  MFE is always ≤ 0 for any sequence (energetically unfavorable pairing yields 0).
  4.  MFE is finite for all inputs.
  5.  Longer sequences have MFE ≤ MFE of their prefix (monotonicity).
  6.  GC-rich sequences have more negative MFE than AT-rich sequences.
  7.  Dot-bracket structure has matched parentheses.
  8.  Fold returns (mfe, frac_paired, largest_loop) all in valid ranges.
  9.  frac_paired ∈ [0, 1] for any input.
 10.  extract_features() target_mfe is no longer hardcoded -5.0.
"""

import math
import pytest
import numpy as np

from primerforge.secondary_structure import (
    AmpliconFolder,
    NussinovMFE,
    NNStackingParams,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_folder() -> AmpliconFolder:
    return AmpliconFolder()


def make_primer_pair(f_seq: str, r_seq: str, product_size: int = 100):
    """Builds a minimal PrimerPair for extract_features() integration tests."""
    from primerforge.biophysics import BiophysicsEngine, PrimerSequence, PrimerPair
    engine = BiophysicsEngine()
    f_th = engine.calculate_thermo_features(f_seq)
    r_th = engine.calculate_thermo_features(r_seq)
    f_gc = 100.0 * sum(1 for b in f_seq.upper() if b in "GC") / len(f_seq)
    r_gc = 100.0 * sum(1 for b in r_seq.upper() if b in "GC") / len(r_seq)
    fwd = PrimerSequence(
        sequence=f_seq, start=0, length=len(f_seq), tm=f_th["tm"],
        gc_percent=f_gc, hairpin_dg=f_th["hairpin_dg"],
        homodimer_dg=f_th["homodimer_dg"], penalty=0.5,
    )
    rev = PrimerSequence(
        sequence=r_seq, start=product_size, length=len(r_seq), tm=r_th["tm"],
        gc_percent=r_gc, hairpin_dg=r_th["hairpin_dg"],
        homodimer_dg=r_th["homodimer_dg"], penalty=0.5,
    )
    return PrimerPair(
        forward=fwd, reverse=rev, product_size=product_size,
        cross_dimer_dg=engine.calculate_heterodimer_dg(f_seq, r_seq),
        penalty=0.5,
    )


# ---------------------------------------------------------------------------
# Test 1: Poly-A has MFE ≈ 0 (no WC pairs)
# ---------------------------------------------------------------------------

def test_polya_has_zero_mfe() -> None:
    """Poly-A (N=40) cannot form Watson-Crick base pairs, so MFE must be 0.

    Scientific basis: A-A is not a Watson-Crick pair; no self-complementarity
    exists in homopolymer sequences. Nussinov DP returns 0 for such sequences.
    """
    folder = make_folder()
    mfe = folder.compute_mfe("A" * 40)
    assert abs(mfe) < 1e-6, f"Poly-A MFE should be 0.0, got {mfe:.4f}"


def test_polyt_has_zero_mfe() -> None:
    """Poly-T also cannot form WC pairs; MFE must be 0."""
    folder = make_folder()
    mfe = folder.compute_mfe("T" * 40)
    assert abs(mfe) < 1e-6, f"Poly-T MFE should be 0.0, got {mfe:.4f}"


# ---------------------------------------------------------------------------
# Test 2: Perfect palindrome has deeply negative MFE
# ---------------------------------------------------------------------------

def test_palindrome_has_negative_mfe() -> None:
    """A perfect palindromic sequence forms a stable hairpin with negative MFE.

    GCGCATGCGC is self-complementary — it can form:
        5'-GCGC-loop-GCGC-3'
    giving a strongly negative MFE from GC/CG and CG/GC stacking.
    """
    folder = make_folder()
    # GCATGC is self-complementary (palindrome): G-C, C-G, A-T, T-A, G-C, C-G
    palindrome = "GCATGCATGCGCATGCATGC"
    mfe = folder.compute_mfe(palindrome)
    assert mfe < -1.0, (
        f"Palindromic sequence should have MFE < -1.0, got {mfe:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: MFE is always ≤ 0
# ---------------------------------------------------------------------------

def test_mfe_always_nonpositive() -> None:
    """MFE must be ≤ 0 for any DNA sequence (structure can only be stabilizing).

    Scientific basis: In the Nussinov model, unpaired bases contribute 0 energy.
    Base pairing adds negative (stabilizing) energy. Therefore MFE ≤ 0 always.
    """
    import random
    random.seed(42)
    folder = make_folder()
    bases = "ATGC"
    for _ in range(20):
        length = random.randint(5, 60)
        seq = "".join(random.choices(bases, k=length))
        mfe = folder.compute_mfe(seq)
        assert mfe <= 1e-9, (
            f"MFE must be ≤ 0 for seq='{seq[:20]}...', got {mfe:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 4: MFE is finite for all inputs
# ---------------------------------------------------------------------------

def test_mfe_is_finite() -> None:
    """Verifies that MFE is always a finite float — no NaN or Inf."""
    folder = make_folder()
    test_seqs = [
        "AT",                           # minimal 2-mer
        "GCGCGCGCGCGCGCGCGCGCGCGCGCGC",  # 28-mer GC repeat
        "ATATATATATATATAT",              # 16-mer AT alternating
        "AAAAAAAAAAAAAAAAAAAAAAAA",       # 24-mer poly-A
        "GCATGCATGCATGCATGCATGC",         # 22-mer mixed
        "TTTTTTTTTTTTTTTTTTTT",          # 20-mer poly-T
    ]
    for seq in test_seqs:
        mfe = folder.compute_mfe(seq)
        assert math.isfinite(mfe), (
            f"MFE={mfe} is not finite for seq='{seq[:20]}'"
        )
        assert isinstance(mfe, float), f"Expected float, got {type(mfe)}"


# ---------------------------------------------------------------------------
# Test 5: MFE monotonicity (longer ≤ shorter)
# ---------------------------------------------------------------------------

def test_mfe_monotone_with_length() -> None:
    """MFE of seq[:N] ≥ MFE of seq[:N+k] for any valid extension.

    Scientific basis: adding more bases can only create new pairing opportunities
    (or none), never destroy them. Therefore MFE is monotonically non-increasing
    with sequence length for the same sequence prefix.
    """
    folder = make_folder()
    # Use a structured sequence likely to form secondary structure
    base_seq = "GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"
    mfe_prev = 0.0
    for length in range(10, 51, 10):
        sub_seq = base_seq[:length]
        mfe = folder.compute_mfe(sub_seq)
        assert mfe <= mfe_prev + 1e-9, (
            f"MFE should not increase with length: "
            f"mfe(len={length})={mfe:.3f} > mfe(len={length-10})={mfe_prev:.3f}"
        )
        mfe_prev = mfe


# ---------------------------------------------------------------------------
# Test 6: GC-rich sequences more negative MFE than AT-rich
# ---------------------------------------------------------------------------

def test_gc_rich_more_negative_mfe() -> None:
    """GC-rich amplicons form stronger secondary structure than AT-rich ones.

    Scientific basis: GC pairs have ΔG ≈ -1.82 kcal/mol vs AT ≈ -0.73.
    A 40-mer with high GC content should have more negative MFE.
    """
    folder = make_folder()
    gc_rich = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC"  # 42-mer GC repeat
    at_rich = "ATATATATATATATATATATATATATATATATATATATATAT"  # 42-mer AT repeat
    mfe_gc = folder.compute_mfe(gc_rich)
    mfe_at = folder.compute_mfe(at_rich)
    assert mfe_gc < mfe_at, (
        f"GC-rich MFE ({mfe_gc:.3f}) must be < AT-rich MFE ({mfe_at:.3f})"
    )


# ---------------------------------------------------------------------------
# Test 7: Dot-bracket structure has matched parentheses
# ---------------------------------------------------------------------------

def test_dot_bracket_balanced() -> None:
    """Verifies dot-bracket notation has matched parentheses."""
    dp = NussinovMFE()
    test_seqs = [
        "GCATGCATGCATGCATGC",
        "AAAAAAAAAAAAAAAA",
        "GCGCGCGCGCGCGCGCGC",
        "ATGCATGCATGCATGCATGC",
    ]
    for seq in test_seqs:
        mfe, dot_bracket = dp.compute_mfe(seq)
        n_open = dot_bracket.count("(")
        n_close = dot_bracket.count(")")
        n_dots = dot_bracket.count(".")
        assert n_open == n_close, (
            f"Unbalanced brackets in '{dot_bracket}': {n_open} open, {n_close} close"
        )
        assert n_open + n_close + n_dots == len(seq), (
            f"Structure length {n_open+n_close+n_dots} ≠ seq length {len(seq)}"
        )
        assert all(c in "()." for c in dot_bracket), (
            f"Invalid character in dot-bracket: '{dot_bracket}'"
        )


# ---------------------------------------------------------------------------
# Test 8: fold() returns three values in valid ranges
# ---------------------------------------------------------------------------

def test_fold_returns_valid_ranges() -> None:
    """Verifies AmpliconFolder.fold() returns (mfe, frac_paired, largest_loop) in valid ranges."""
    folder = make_folder()
    test_seqs = [
        "GCATGCATGCATGCATGCATGC",
        "AAAAAAAAAAAAAAAAAAAAAAAA",
        "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
    ]
    for seq in test_seqs:
        mfe, frac_paired, largest_loop = folder.fold(seq)

        assert math.isfinite(mfe), f"MFE not finite: {mfe}"
        assert mfe <= 1e-9, f"MFE={mfe:.4f} must be ≤ 0"

        assert 0.0 <= frac_paired <= 1.0, (
            f"frac_paired={frac_paired:.4f} must be in [0, 1]"
        )

        assert isinstance(largest_loop, int), (
            f"largest_loop must be int, got {type(largest_loop)}"
        )
        assert 0 <= largest_loop <= len(seq), (
            f"largest_loop={largest_loop} > len(seq)={len(seq)}"
        )


# ---------------------------------------------------------------------------
# Test 9: frac_paired ∈ [0, 1]
# ---------------------------------------------------------------------------

def test_frac_paired_range() -> None:
    """Verifies frac_paired is always in [0, 1] across diverse sequences."""
    import random
    random.seed(123)
    folder = make_folder()
    bases = "ATGC"
    for _ in range(30):
        length = random.randint(4, 80)
        seq = "".join(random.choices(bases, k=length))
        _, frac_paired, _ = folder.fold(seq)
        assert 0.0 <= frac_paired <= 1.0, (
            f"frac_paired={frac_paired:.4f} out of [0,1] for seq='{seq[:20]}...'"
        )


# ---------------------------------------------------------------------------
# Test 10: extract_features() target_mfe is not hardcoded -5.0
# ---------------------------------------------------------------------------

def test_extract_features_mfe_is_not_stub() -> None:
    """Verifies extract_features() target_mfe is computed (not hardcoded -5.0).

    Two primer pairs with very different GC content must produce different
    target_mfe values. If the value were still hardcoded, they would be identical.
    """
    from primerforge.ml_scorer import MLScorer

    scorer = MLScorer()

    # GC-rich primers (high structure probability)
    pair_gc = make_primer_pair(
        f_seq="GCGCGCGCGCGCGCGCGCGC",
        r_seq="CGCGCGCGCGCGCGCGCGCG",
    )
    # AT-rich primers (low structure probability)
    pair_at = make_primer_pair(
        f_seq="ATATATATATATATATATATAT",
        r_seq="TATATATATATATATATATATA",
    )

    feats_gc = scorer.extract_features(pair_gc)
    feats_at = scorer.extract_features(pair_at)

    assert len(feats_gc) == 40, f"Expected 40 features, got {len(feats_gc)}"

    # target_mfe is at index 24 (Section 3 starts after 24 features: 8+8+4+4=24)
    # Feature vector order from ml_scorer.py line 299-307:
    #   [0-7]  thermodynamics (8)
    #   [8-15] sequence composition (8)
    #   [16-19] 3' dinucleotide (4)
    #   [20-23] r_3_dinuc + stability (4)
    #   [24-27] target_mfe, target_gc, target_len, primer_overlap (4)
    mfe_gc = feats_gc[24]
    mfe_at = feats_at[24]

    # Must differ (not both hardcoded -5.0)
    assert mfe_gc != mfe_at, (
        f"target_mfe should differ for GC-rich ({mfe_gc:.3f}) vs "
        f"AT-rich ({mfe_at:.3f}) — still hardcoded?"
    )

    # GC-rich should have more negative MFE (more secondary structure)
    assert mfe_gc < mfe_at, (
        f"GC-rich target_mfe ({mfe_gc:.3f}) should be < AT-rich ({mfe_at:.3f})"
    )

    # Neither should be exactly -5.0 (old stub value)
    assert mfe_gc != -5.0, "target_mfe is still the hardcoded stub -5.0"
    assert mfe_at != -5.0, "target_mfe is still the hardcoded stub -5.0"
