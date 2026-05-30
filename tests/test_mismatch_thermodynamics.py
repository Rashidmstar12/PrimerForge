"""Unit tests for position-specific mismatch thermodynamics and extension penalties in the BiophysicsEngine."""

import pytest
import numpy as np

from primerforge.biophysics import BiophysicsEngine, PrimerSequence, PrimerPair
from primerforge.ml_scorer import MLScorer

@pytest.fixture
def biophysics() -> BiophysicsEngine:
    return BiophysicsEngine()

def test_perfect_match_zero_penalty(biophysics) -> None:
    """Verifies that a perfectly complementary primer-template pairing yields a 0.0 mismatch penalty."""
    primer = "CACCATTGGCAATGAGCGGT"
    template = "GTGGTAACCGTTACTCGCCA"  # 3' to 5' complement of primer
    
    penalty = biophysics.calculate_mismatch_penalty(primer, template)
    assert penalty == 0.0

def test_position_specific_weights(biophysics) -> None:
    """Verifies that mismatches closer to the 3' end receive exponentially higher penalties."""
    primer = "CACCATTGGCAATGAGCGGT"
    
    # 1. 3' terminal base mismatch (position index 19, distance 0 from 3' end)
    template_3prime_mismatch = "GTGGTAACCGTTACTCGCCG"  # T-G mismatch at the 3' terminal base
    penalty_3prime = biophysics.calculate_mismatch_penalty(primer, template_3prime_mismatch)
    
    # 2. 5' end mismatch (position index 0, distance 19 from 3' end)
    template_5prime_mismatch = "ATGGTAACCGTTACTCGCCA"  # C-A mismatch at the 5' end
    penalty_5prime = biophysics.calculate_mismatch_penalty(primer, template_5prime_mismatch)
    
    # Even though C-A is a more severe mismatch base than T-G wobble (4.0 vs 1.0 base penalty),
    # the 3' position weight is so large that the 3' terminal mismatch penalty should be much higher.
    assert penalty_3prime > penalty_5prime
    
    # Test identical mismatch (e.g. C-C) at different positions:
    # Position index 19 (3' terminal) vs Position index 0 (5' end)
    primer_cc = "CACCATTGGCAATGAGCGGC"  # ends in C
    template_perfect = "GTGGTAACCGTTACTCGCCG"
    penalty_cc_3prime = biophysics.calculate_mismatch_penalty(primer_cc, "GTGGTAACCGTTACTCGCCC")  # C-C mismatch at 3' end
    penalty_cc_5prime = biophysics.calculate_mismatch_penalty(primer_cc, "ATGGTAACCGTTACTCGCCG")  # C-C mismatch at 5' end
    
    assert penalty_cc_3prime > penalty_cc_5prime * 10.0  # 3' mismatch is at least 10x more severe due to e^(-0.15 * 19)

def test_mismatch_hierarchy(biophysics) -> None:
    """Verifies that severe mismatches (like C-C) yield larger penalties than mild wobbles (like G-T) at the same position."""
    primer = "CACCATTGGCAATGAGCGGT"  # ends in T
    
    # G-T wobble at 3' end (template base G, primer base T, formed base-pair T-G)
    template_wobble = "GTGGTAACCGTTACTCGCCG"
    penalty_wobble = biophysics.calculate_mismatch_penalty(primer, template_wobble)
    
    # C-T mismatch at 3' end (template base C, primer base T, formed base-pair T-C)
    template_heavy = "GTGGTAACCGTTACTCGCCC"
    penalty_heavy = biophysics.calculate_mismatch_penalty(primer, template_heavy)
    
    assert penalty_heavy > penalty_wobble

def test_ml_scorer_variant_prediction() -> None:
    """Verifies that MLScorer properly discounts success probability when a template carries a mismatch."""
    scorer = MLScorer()
    
    f_seq = "CACCATTGGCAATGAGCGGT"
    r_seq = "CGCTCAGGAGGAGCAATGAT"
    f_seq_obj = PrimerSequence(f_seq, 0, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    r_seq_obj = PrimerSequence(r_seq, 100, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    pair = PrimerPair(f_seq_obj, r_seq_obj, 150, -0.5, 0.0)
    
    # Perfect templates
    f_template_perfect = "GTGGTAACCGTTACTCGCCA"
    r_template_perfect = "GCGAGTCCTCCTCGTTACTA"
    
    p_perfect = scorer.predict_success_with_variant_mismatches(
        pair, f_template_perfect, r_template_perfect
    )
    
    # Templates with severe 3' terminal base mismatches
    f_template_mismatch = "GTGGTAACCGTTACTCGCCG"  # T-G mismatch at 3' end of forward primer
    r_template_mismatch = "GCGAGTCCTCCTCGTTACTG"  # T-G mismatch at 3' end of reverse primer
    
    p_mismatch = scorer.predict_success_with_variant_mismatches(
        pair, f_template_mismatch, r_template_mismatch
    )
    
    # Success probability should drop significantly due to thermodynamic destabilization
    assert p_perfect > p_mismatch
    assert p_mismatch < 0.5  # Should fail to amplify reliably
