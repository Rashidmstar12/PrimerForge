import pytest
try:
    import lightgbm
except ImportError:
    pytest.skip("lightgbm not available", allow_module_level=True)

"""Unit tests for multiplex panel Optimization and dimerization matrix calculation in PrimerForge."""

import pytest
import numpy as np

from primerforge.biophysics import PrimerSequence, PrimerPair, BiophysicsEngine
from primerforge.multiplex import MultiplexOptimizer, MultiplexPanel


@pytest.fixture
def optimizer() -> MultiplexOptimizer:
    return MultiplexOptimizer()


def test_build_dimerization_matrix(optimizer) -> None:
    """Verifies dimerization matrix construction matches symmetric dimensions and values."""
    f1 = PrimerSequence("CACCATTGGCAATGAGCGGT", 0, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    r1 = PrimerSequence("CGCTCAGGAGGAGCAATGAT", 100, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    pair1 = PrimerPair(f1, r1, 150, -0.5, 0.0)

    f2 = PrimerSequence("AAGACCTGTACGCCAACACA", 0, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    r2 = PrimerSequence("GCATGGAGTCCTGTGGCATC", 100, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    pair2 = PrimerPair(f2, r2, 120, -0.6, 0.0)

    pairs = [pair1, pair2]
    matrix, labels = optimizer.build_dimerization_matrix(pairs)

    # 2 pairs -> 4 unique primers -> shape (4, 4)
    assert matrix.shape == (4, 4)
    assert len(labels) == 4
    assert labels == ["Pair_0_F", "Pair_0_R", "Pair_1_F", "Pair_1_R"]

    # Verify symmetry
    assert np.allclose(matrix, matrix.T)

    # Verify homodimers are present on diagonal and heterodimers are present off-diagonal
    for i in range(4):
        assert matrix[i, i] <= 0.0  # homodimer stability is zero or negative (kcal/mol)
        for j in range(4):
            assert matrix[i, j] <= 0.0


def test_calculate_multiplex_penalty(optimizer) -> None:
    """Verifies that the penalty is accumulated only for dimerization stabilities exceeding the threshold."""
    # Create a mock dimerization matrix
    matrix = np.array(
        [
            [-0.2, -3.0, -7.5, -0.5],
            [-3.0, -0.3, -1.2, -8.2],
            [-7.5, -1.2, -0.1, -2.0],
            [-0.5, -8.2, -2.0, -0.4],
        ],
        dtype=np.float32,
    )

    # Using threshold of -6.0:
    # Stable pairings:
    # matrix[0, 2] = -7.5 (exceeds by 1.5)
    # matrix[1, 3] = -8.2 (exceeds by 2.2)
    # Total expected penalty = 1.5 + 2.2 = 3.7
    penalty = optimizer.calculate_multiplex_penalty(matrix, threshold=-6.0)
    assert abs(penalty - 3.7) < 1e-4

    # Using threshold of -9.0 (no entries exceed)
    penalty_stringent = optimizer.calculate_multiplex_penalty(matrix, threshold=-9.0)
    assert penalty_stringent == 0.0


def test_greedy_panel_selection_and_rescue(optimizer) -> None:
    """Verifies that design_compatible_panel successfully filters out cross-reactive candidates and rescues the panel."""
    # Locus 0 Candidate Pool: 1 good pair
    f1 = PrimerSequence("CACCATTGGCAATGAGCGGT", 0, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    r1 = PrimerSequence("CGCTCAGGAGGAGCAATGAT", 100, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    pair1 = PrimerPair(f1, r1, 150, -0.5, 0.0)
    pool0 = [pair1]

    # Locus 1 Candidate Pool:
    # 1. Candidate A: Pairs perfectly with Locus 0 primers to form a massive primer-dimer (simulated cross-reactivity)
    # Let's use the actual complementary sequence of pair1.forward (CACCATTGGCAATGAGCGGT) as Candidate A's primer.
    # When matched, they form an extremely stable perfect double-stranded complex (very low negative dG)
    f_bad = PrimerSequence(
        "ACCGCTCATTGCCAATGGTG", 0, 20, 60.0, 50.0, -0.2, -0.3, 0.0
    )  # perfect complement of f1
    r_bad = PrimerSequence("GCATGGAGTCCTGTGGCATC", 100, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    pair_bad = PrimerPair(
        f_bad, r_bad, 120, -10.0, 0.0
    )  # Bad pair due to severe dimerization

    # 2. Candidate B: Highly compatible, standard primer sequence
    f_good = PrimerSequence("AAGACCTGTACGCCAACACA", 0, 20, 60.0, 50.0, -0.2, -0.3, 0.0)
    r_good = PrimerSequence(
        "GCATGGAGTCCTGTGGCATC", 100, 20, 60.0, 50.0, -0.2, -0.3, 0.0
    )
    pair_good = PrimerPair(f_good, r_good, 120, -0.5, 0.0)

    # Pool 1 contains both bad (index 0) and good (index 1) candidates
    pool1 = [pair_bad, pair_good]

    # We expect the optimizer to reject pair_bad because its dimerization is too stable,
    # and instead select pair_good to save the panel from cross-reactivity failure.
    panel = optimizer.design_compatible_panel(
        candidate_pools=[pool0, pool1], threshold=-6.0, hard_limit=-9.0
    )

    assert len(panel.pairs) == 2
    assert panel.pairs[0] == pair1
    assert (
        panel.pairs[1] == pair_good
    )  # Successfully selected Candidate B and filtered out Candidate A!
    assert panel.global_penalty < 1.0  # Dimerization should be highly minor
