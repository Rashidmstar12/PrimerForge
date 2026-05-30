"""Unit tests for the BiophysicsEngine module in PrimerForge."""

import pytest
from primerforge.biophysics import BiophysicsEngine, PrimerPair, PrimerSequence


@pytest.fixture
def engine() -> BiophysicsEngine:
    """Fixture to initialize a standard BiophysicsEngine instance."""
    return BiophysicsEngine(
        opt_tm=60.0,
        min_tm=57.0,
        max_tm=63.0,
        opt_size=20,
        min_size=18,
        max_size=24,
    )


def test_calculate_thermo_features(engine: BiophysicsEngine) -> None:
    """Tests the thermodynamic calculations on individual sequences."""
    seq = "ATGCGATCGATCGATCGATC"
    features = engine.calculate_thermo_features(seq)

    assert "tm" in features
    assert "hairpin_dg" in features
    assert "homodimer_dg" in features

    assert isinstance(features["tm"], float)
    assert isinstance(features["hairpin_dg"], float)
    assert isinstance(features["homodimer_dg"], float)

    # Tm for a 20bp sequence should be in a reasonable biophysical range (typically 50-70C)
    assert 40.0 < features["tm"] < 80.0


def test_calculate_heterodimer_dg(engine: BiophysicsEngine) -> None:
    """Tests cross-dimerisation free energy calculation between two sequences."""
    seq1 = "ATGCGATCGATCGATCGATC"
    seq2 = "GATCGATCGATCGATCGATC"
    dg = engine.calculate_heterodimer_dg(seq1, seq2)

    assert isinstance(dg, float)
    # The dg should be formatted as negative or zero values representing stabilization
    assert dg <= 0.0


def test_generate_candidates(engine: BiophysicsEngine) -> None:
    """Tests the design of primer pairs for a valid DNA template sequence."""
    # A standard target sequence from human beta-actin (ACTB) gene
    target_seq = (
        "CACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGCATGGAGTCCT"
        "GTGGCATCCACGAAACTACCTTCAACTCCATCATGAAGTGTGACGTGGACATCCGCAAAGACCTGTACGCC"
        "AACACAGTGCTGTCTGGCGGCACCACCATGTACCCTGGCATTGCTGACAGGATGCAGAAGGAGATCACTGC"
        "CCTGGCACCCAGCACAATGAAGATCAAGATCATTGCTCCTCCTGAGCGC"
    )

    pairs = engine.generate_candidates(target_seq, num_return=5)

    assert isinstance(pairs, list)
    assert len(pairs) > 0

    for pair in pairs:
        assert isinstance(pair, PrimerPair)
        assert isinstance(pair.forward, PrimerSequence)
        assert isinstance(pair.reverse, PrimerSequence)

        # Validate structural specifications
        assert len(pair.forward.sequence) >= 18
        assert len(pair.reverse.sequence) >= 18

        # Validate Tm ranges are consistent with engine defaults
        assert 57.0 <= pair.forward.tm <= 63.0
        assert 57.0 <= pair.reverse.tm <= 63.0

        # Validate structural energy attributes
        assert pair.forward.hairpin_dg <= 0.0
        assert pair.reverse.homodimer_dg <= 0.0
        assert pair.cross_dimer_dg <= 0.0
        assert pair.product_size > 0
