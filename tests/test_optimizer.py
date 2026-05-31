import pytest
try:
    import lightgbm
except ImportError:
    pytest.skip("lightgbm not available", allow_module_level=True)

"""Unit tests for the MultiplexOptimizer module in PrimerForge."""

import pytest
from typing import Any, Dict, List
from unittest.mock import patch

from primerforge.biophysics import BiophysicsEngine, PrimerPair, PrimerSequence
from primerforge.optimizer import MultiplexOptimizer


@pytest.fixture
def biophys() -> BiophysicsEngine:
    """Fixture to initialize a BiophysicsEngine."""
    return BiophysicsEngine()


@pytest.fixture
def optimizer(biophys) -> MultiplexOptimizer:
    """Fixture to initialize a MultiplexOptimizer."""
    return MultiplexOptimizer(biophys_engine=biophys)


@pytest.fixture
def mock_pairs() -> List[Dict[str, Any]]:
    """Fixture to generate a mock list of scored candidate primer pairs.

    - Pair 0: Locus A, sequence 'ATGCGATCGATCGATCGATC', success 0.95
    - Pair 1: Locus B, sequence 'GATCGATCGATCGATCGATC', success 0.90
    - Pair 2: Locus B (alternative), sequence 'GCATCGATCGATCGATCGAT', success 0.85
    - Pair 3: Locus C (dimerizing), sequence 'GATCGATCGATCGATCGATC' (exact match to Pair 1), success 0.80
    """
    f_seq1 = "ATGCGATCGATCGATCGATC"
    r_seq1 = "GATCGATCGATCGATCGATC"
    pair1 = PrimerPair(
        forward=PrimerSequence(f_seq1, 0, 20, 60.0, 50.0, 0.0, 0.0, 0.0),
        reverse=PrimerSequence(r_seq1, 100, 20, 60.0, 50.0, 0.0, 0.0, 0.0),
        product_size=120,
        cross_dimer_dg=0.0,
        penalty=0.0,
    )

    f_seq2 = "GCATCGATCGATCGATCGAT"
    r_seq2 = "TTCGATCGATCGATCGATCG"
    pair2 = PrimerPair(
        forward=PrimerSequence(f_seq2, 0, 20, 60.0, 50.0, 0.0, 0.0, 0.0),
        reverse=PrimerSequence(r_seq2, 100, 20, 60.0, 50.0, 0.0, 0.0, 0.0),
        product_size=120,
        cross_dimer_dg=0.0,
        penalty=0.0,
    )

    # Let's construct scored dicts
    return [
        {
            "pair": pair1,
            "predicted_success": 0.95,
            "target_id": "locus_A",
            "is_valid": True,
        },
        {
            "pair": pair2,
            "predicted_success": 0.90,
            "target_id": "locus_B",
            "is_valid": True,
        },
        {
            "pair": pair1,
            "predicted_success": 0.85,
            "target_id": "locus_B",  # Alternative for Locus B, should trigger locus constraint
            "is_valid": True,
        },
    ]


def test_locus_constraint(optimizer: MultiplexOptimizer, mock_pairs) -> None:
    """Verifies that at most one primer pair is selected per target locus."""
    # We have 3 candidates representing 2 loci (A and B).
    # Since locus_B has two candidate options, the solver must select at most one for B!
    selected, obj_val = optimizer.optimize_panel(mock_pairs, max_plex=3)

    assert len(selected) <= 2
    selected_loci = [item["target_id"] for item in selected]
    # Locus B must not be selected twice!
    assert selected_loci.count("locus_B") <= 1


def test_dimer_constraint(optimizer: MultiplexOptimizer) -> None:
    """Verifies that dimerizing primers are rejected to ensure a dimer-free panel."""
    # Create two pairs that have a strong cross-dimer potential
    # Let's mock calculate_heterodimer_dg to return a very stable dimer (e.g. -6.0 kcal/mol)
    f_seq = "ATGCGATCGATCGATCGATC"
    pair1 = PrimerPair(
        forward=PrimerSequence(f_seq, 0, 20, 60.0, 50.0, 0.0, 0.0, 0.0),
        reverse=PrimerSequence(f_seq, 100, 20, 60.0, 50.0, 0.0, 0.0, 0.0),
        product_size=120,
        cross_dimer_dg=0.0,
        penalty=0.0,
    )

    scored = [
        {
            "pair": pair1,
            "predicted_success": 0.95,
            "target_id": "locus_A",
            "is_valid": True,
        },
        {
            "pair": pair1,
            "predicted_success": 0.90,
            "target_id": "locus_B",
            "is_valid": True,
        },
    ]

    with patch.object(BiophysicsEngine, "calculate_heterodimer_dg", return_value=-8.0):
        # With dimer_dg of -8.0 kcal/mol, they are highly incompatible (limit is -4.5)
        selected, obj = optimizer.optimize_panel(
            scored, max_plex=2, delta_g_threshold=-4.5
        )

        # The solver must choose at most one because they dimerize severely!
        assert len(selected) <= 1


def test_greedy_fallback(optimizer: MultiplexOptimizer, mock_pairs) -> None:
    """Verifies that the optimizer falls back to a clean greedy heuristic if the ILP solver fails."""
    with patch("pulp.LpProblem.solve", side_effect=Exception("Solver crash")):
        selected, obj = optimizer.optimize_panel(mock_pairs, max_plex=2)

        assert len(selected) <= 2
        selected_loci = [item["target_id"] for item in selected]
        assert selected_loci.count("locus_B") <= 1
