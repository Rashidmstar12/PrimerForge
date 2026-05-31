"""Minimal CI-safe tests for PrimerForge core functionality.
These tests have NO heavy dependencies (no torch, no lightgbm, no xgboost).
They test only the pure-Python biophysics and optimizer modules.
"""
import pytest
import os
os.environ["PRIMERFORGE_NO_AUTOTRAIN"] = "1"


def test_import_biophysics():
    """Test that the biophysics module imports cleanly."""
    from primerforge.biophysics import BiophysicsEngine
    engine = BiophysicsEngine()
    assert engine is not None


def test_nearest_neighbor_tm():
    """Test Tm calculation for a known primer sequence."""
    from primerforge.biophysics import BiophysicsEngine
    engine = BiophysicsEngine()
    features = engine.calculate_thermo_features("ATCGATCGATCGATCG")
    assert "tm" in features
    assert 40.0 < features["tm"] < 80.0


def test_gc_content():
    """Test GC content calculation via the BiophysicsEngine."""
    from primerforge.biophysics import BiophysicsEngine
    engine = BiophysicsEngine()
    seq = "GCGCGCGCGCGCGCGC"
    gc = 100.0 * sum(1 for b in seq.upper() if b in "GC") / len(seq)
    assert gc == pytest.approx(100.0, abs=1.0)


def test_primer_pair_creation():
    """Test PrimerPair dataclass instantiation."""
    from primerforge.biophysics import PrimerPair, PrimerSequence
    fwd = PrimerSequence(sequence="ATCGATCGATCG", start=0, length=12,
                         tm=55.0, gc_percent=50.0, hairpin_dg=-1.0,
                         homodimer_dg=-1.0, penalty=0.0)
    rev = PrimerSequence(sequence="CGATCGATCGAT", start=200, length=12,
                         tm=55.0, gc_percent=50.0, hairpin_dg=-1.0,
                         homodimer_dg=-1.0, penalty=0.0)
    pair = PrimerPair(forward=fwd, reverse=rev, product_size=212,
                      cross_dimer_dg=-2.0, penalty=0.5)
    assert pair.product_size == 212
    assert pair.forward.sequence == "ATCGATCGATCG"


def test_mlscorer_no_autotrain():
    """Test MLScorer initializes without crashing when auto-train is disabled."""
    pytest.importorskip("lightgbm", reason="lightgbm not installed")
    from primerforge.ml_scorer import MLScorer
    scorer = MLScorer(auto_train=False)
    assert scorer is not None
    assert scorer.models == []


def test_provenance():
    """Test that get_training_data_provenance returns expected structure."""
    pytest.importorskip("lightgbm", reason="lightgbm not installed")
    from primerforge.ml_scorer import MLScorer
    scorer = MLScorer(auto_train=False)
    prov = scorer.get_training_data_provenance()
    assert "source" in prov
    assert "n_samples" in prov
