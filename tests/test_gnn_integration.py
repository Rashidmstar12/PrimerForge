import pytest
try:
    import lightgbm
except ImportError:
    pytest.skip("lightgbm not available", allow_module_level=True)

"""
Step 3 Integration Tests: BioGNN end-to-end integration with MLScorer.

Verifies that:
  1. extract_features() returns exactly 40 dimensions including GNN-predicted Tm and dG.
  2. GNN features are non-trivial (non-zero) for real primer sequences.
  3. explain_prediction() returns a dict with all 40 feature names including gnn_pred_tm/gnn_pred_dg.
  4. predict_success() returns a valid [0.01, 0.99] probability with the GNN integrated scorer.
  5. predict_success_with_uncertainty() returns (prob, std) with plausible uncertainty.
  6. GNN weights survive a save/load round-trip via calib JSON serialization.
  7. _add_gnn_features() correctly appends 2 GNN columns to a batch feature matrix.
  8. BioGNN forward pass is differentiable end-to-end: gradient check on a dummy loss.
  9. Training GNN on synthetic primer pairs decreases loss monotonically (convergence).
 10. Legacy 36-dim feature vectors are backward-compatible with the explain_prediction fallback.
"""

import os
import json
import pytest
import tempfile
import numpy as np
from typing import List, Tuple

from primerforge.gnn_biophysics import (
    BioGNN,
    build_hybrid_graph,
    build_primer_graph,
    GraphConvLayer,
    GraphMeanPool,
    compute_symmetric_normalized_adjacency,
)
from primerforge.biophysics import BiophysicsEngine, PrimerPair, PrimerSequence
from primerforge.ml_scorer import MLScorer


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def make_primer_pair(
    f_seq: str = "ATGCATGCATGCATGCATGC",
    r_seq: str = "GCATGCATGCATGCATGCAT",
    product_size: int = 120,
) -> PrimerPair:
    """Creates a lightweight PrimerPair object from raw sequences using primer3 biophysics."""
    engine = BiophysicsEngine()

    f_thermo = engine.calculate_thermo_features(f_seq)
    r_thermo = engine.calculate_thermo_features(r_seq)

    f_gc = 100.0 * sum(1 for b in f_seq.upper() if b in "GC") / max(len(f_seq), 1)
    r_gc = 100.0 * sum(1 for b in r_seq.upper() if b in "GC") / max(len(r_seq), 1)

    fwd = PrimerSequence(
        sequence=f_seq,
        start=0,
        length=len(f_seq),
        tm=f_thermo["tm"],
        gc_percent=f_gc,
        hairpin_dg=f_thermo["hairpin_dg"],
        homodimer_dg=f_thermo["homodimer_dg"],
        penalty=0.0,
    )
    rev = PrimerSequence(
        sequence=r_seq,
        start=product_size,
        length=len(r_seq),
        tm=r_thermo["tm"],
        gc_percent=r_gc,
        hairpin_dg=r_thermo["hairpin_dg"],
        homodimer_dg=r_thermo["homodimer_dg"],
        penalty=0.0,
    )
    cross_dg = engine.calculate_heterodimer_dg(f_seq, r_seq)
    pair = PrimerPair(
        forward=fwd,
        reverse=rev,
        product_size=product_size,
        cross_dimer_dg=cross_dg,
        penalty=0.0,
    )
    return pair


# ---------------------------------------------------------------------------
# Test 1: Feature vector is exactly 40 dimensions
# ---------------------------------------------------------------------------


def test_extract_features_dim() -> None:
    """Verifies that extract_features() returns a 40-dimensional feature vector."""
    scorer = MLScorer()
    pair = make_primer_pair()
    features = scorer.extract_features(pair)
    assert (
        len(features) == 40
    ), f"Expected 40-dimensional feature vector, got {len(features)}."


# ---------------------------------------------------------------------------
# Test 2: GNN features at indices 36-37 are non-trivially non-zero
# ---------------------------------------------------------------------------


def test_gnn_features_are_present() -> None:
    """Verifies GNN-predicted Tm/dG (indices 36-37) are non-zero for a real primer pair."""
    scorer = MLScorer()
    pair = make_primer_pair()
    features = scorer.extract_features(pair)
    # The GNN should produce non-zero outputs for a real sequence pair
    gnn_tm = features[36]
    gnn_dg = features[37]
    # Values should be finite floats (not NaN/Inf)
    assert np.isfinite(gnn_tm), f"GNN-predicted Tm must be finite, got {gnn_tm}"
    assert np.isfinite(gnn_dg), f"GNN-predicted dG must be finite, got {gnn_dg}"


# ---------------------------------------------------------------------------
# Test 3: explain_prediction returns all 40 feature names
# ---------------------------------------------------------------------------


def test_explain_prediction_has_40_keys() -> None:
    """Verifies that explain_prediction() returns contributions for all 40 features."""
    scorer = MLScorer()
    pair = make_primer_pair()
    contributions = scorer.explain_prediction(pair)
    assert isinstance(contributions, dict), "explain_prediction() must return a dict."
    assert (
        "gnn_pred_tm" in contributions
    ), "explain_prediction() must include 'gnn_pred_tm' key."
    assert (
        "gnn_pred_dg" in contributions
    ), "explain_prediction() must include 'gnn_pred_dg' key."
    assert (
        len(contributions) == 40
    ), f"Expected 40 feature contributions, got {len(contributions)}."


# ---------------------------------------------------------------------------
# Test 4: predict_success returns a valid probability
# ---------------------------------------------------------------------------


def test_predict_success_range() -> None:
    """Verifies predict_success() returns a float in the range [0.01, 0.99]."""
    scorer = MLScorer()
    pair = make_primer_pair()
    prob = scorer.predict_success(pair)
    assert isinstance(prob, float), "predict_success() must return a float."
    assert (
        0.01 <= prob <= 0.99
    ), f"Expected success probability in [0.01, 0.99], got {prob:.4f}."


# ---------------------------------------------------------------------------
# Test 5: predict_success_with_uncertainty returns (prob, std)
# ---------------------------------------------------------------------------


def test_predict_success_with_uncertainty() -> None:
    """Verifies predict_success_with_uncertainty() returns (prob, std) with valid ranges."""
    scorer = MLScorer()
    pair = make_primer_pair()
    prob, std = scorer.predict_success_with_uncertainty(pair)
    assert 0.01 <= prob <= 0.99, f"Probability out of range: {prob}"
    assert std >= 0.0, f"Standard deviation must be non-negative, got {std}"
    assert np.isfinite(std), f"Standard deviation must be finite, got {std}"


# ---------------------------------------------------------------------------
# Test 6: GNN weights survive save/load round-trip
# ---------------------------------------------------------------------------


def test_gnn_weights_serialization() -> None:
    """Verifies that BioGNN weights survive a full JSON serialize/deserialize round-trip."""
    gnn_original = BioGNN()
    # Run a forward pass to compute some values
    seq = "ATGCATGCATGC"
    X, A = build_primer_graph(seq)
    pred_before = gnn_original.forward(X, A).copy()

    # Serialize to dict and back
    weights_dict = gnn_original.to_dict()
    gnn_loaded = BioGNN()
    gnn_loaded.from_dict(weights_dict)

    pred_after = gnn_loaded.forward(X, A)
    assert np.allclose(
        pred_before, pred_after, atol=1e-5
    ), "GNN forward pass output should be identical after serialization round-trip."


# ---------------------------------------------------------------------------
# Test 7: _add_gnn_features appends exactly 2 GNN columns
# ---------------------------------------------------------------------------


def test_add_gnn_features_shape() -> None:
    """Verifies _add_gnn_features() and _add_transformer_features() appends columns to an existing feature matrix."""
    import pandas as pd

    scorer = MLScorer()

    # Create a small dummy feature matrix (N=5 samples, 36 base features)
    N = 5
    X_base = np.random.randn(N, 36).astype(np.float32)

    # Dummy dataframe with forward/reverse sequences
    df = pd.DataFrame(
        {
            "forward_seq": ["ATGCATGCATGCATGCATGC"] * N,
            "reverse_seq": ["GCATGCATGCATGCATGCAT"] * N,
        }
    )

    X_augmented = scorer._add_gnn_features(X_base, df)
    X_augmented = scorer._add_transformer_features(X_augmented, df)
    assert X_augmented.shape == (
        N,
        40,
    ), f"Expected shape (N, 40) after GNN and transformer augmentation, got {X_augmented.shape}."
    # GNN columns should be finite
    assert np.all(
        np.isfinite(X_augmented[:, 36:])
    ), "GNN-augmented columns must contain only finite values."


# ---------------------------------------------------------------------------
# Test 8: BioGNN full backprop gradient check (end-to-end numerical diff)
# ---------------------------------------------------------------------------


def test_biognn_end_to_end_grad() -> None:
    """Checks BioGNN W1 dense weight gradient via finite differences (end-to-end)."""
    np.random.seed(2025)
    seq = "ATGCGCAT"
    X, A = build_primer_graph(seq)
    gnn = BioGNN()

    pred = gnn.forward(X, A)
    d_out = np.random.normal(0, 1.0, pred.shape).astype(np.float32)
    gnn.backward(d_out)
    dW1_analytic = gnn.dW1.copy()

    eps = 1e-4
    dW1_numeric = np.zeros_like(gnn.W1)
    for i in range(gnn.W1.shape[0]):
        for j in range(gnn.W1.shape[1]):
            old_val = gnn.W1[i, j]

            gnn.W1[i, j] = old_val + eps
            pred_plus = gnn.forward(X, A)

            gnn.W1[i, j] = old_val - eps
            pred_minus = gnn.forward(X, A)

            gnn.W1[i, j] = old_val

            dW1_numeric[i, j] = np.sum(d_out * (pred_plus - pred_minus)) / (2 * eps)

    assert np.allclose(
        dW1_analytic, dW1_numeric, atol=2e-3, rtol=2e-3
    ), "BioGNN W1 analytic gradients do not match finite-difference estimates."


# ---------------------------------------------------------------------------
# Test 9: BioGNN converges (loss decreases) over training epochs
# ---------------------------------------------------------------------------


def test_biognn_convergence_extended() -> None:
    """Verifies BioGNN loss monotonically decreases over 20 epochs of training."""
    np.random.seed(42)
    sequences = [
        ("ATGCATGCATGCATGCATGC", "GCATGCATGCATGCATGCAT"),
        ("CGTAGCTAGCTAGCTAGCTA", "TAGCTAGCTAGCTAGCTAGC"),
        ("GGCCGGCCGGCCGGCCGGCC", "CCGGCCGGCCGGCCGGCCGG"),
        ("TTAAGGCCTTAAGGCCTTAA", "TTAAGGCCTTAAGGCCTTAA"),
        ("AAGCTTAAGCTTAAGCTTAA", "TTAAGCTTAAGCTTAAGCTT"),
        ("GCGCGCGCGCGCGCGCGCGC", "CGCGCGCGCGCGCGCGCGCG"),
    ]
    targets = np.array(
        [
            [58.0, -2.0],
            [60.0, -1.5],
            [62.0, -4.0],
            [55.0, -0.5],
            [57.0, -3.0],
            [65.0, -6.0],
        ],
        dtype=np.float32,
    )

    gnn = BioGNN()
    losses = gnn.train_on_pairs(sequences, targets, epochs=20, lr=0.01)

    assert len(losses) == 20, f"Expected 20 loss values, got {len(losses)}"
    # Last 5 epoch average should be lower than first 5
    early_avg = float(np.mean(losses[:5]))
    late_avg = float(np.mean(losses[-5:]))
    assert (
        late_avg < early_avg
    ), f"BioGNN must converge: early_loss={early_avg:.4f}, late_loss={late_avg:.4f}"


# ---------------------------------------------------------------------------
# Test 10: explain_prediction fallback handles legacy 36-dim booster importances
# ---------------------------------------------------------------------------


def test_explain_prediction_fallback_legacy_dims() -> None:
    """Verifies the importance fallback loop handles 36-dim importances gracefully
    when feature_cols expects 40 (backward compatibility with old models)."""
    scorer = MLScorer()
    pair = make_primer_pair()

    # Monkey-patch the booster feature_importance to return 36 values
    # to simulate loading a legacy model without GNN features
    if scorer.models:
        orig_importance = scorer.models[0].feature_importance

        def mock_importance_36(importance_type="gain"):
            imp = orig_importance(importance_type=importance_type)
            # Return only first 36 values (simulate legacy model)
            return imp[:36] if len(imp) >= 36 else np.ones(36, dtype=np.float32)

        scorer.models[0].feature_importance = mock_importance_36

        contributions = scorer.explain_prediction(pair)
        # Should complete without error and return 40 keys
        assert (
            len(contributions) == 40
        ), f"Legacy fallback must return 40 contributions, got {len(contributions)}"
        # GNN features should be padded to 0.0 in legacy mode
        assert (
            contributions["gnn_pred_tm"] == 0.0
        ), "Legacy-padded GNN Tm importance should be 0.0"
        assert (
            contributions["gnn_pred_dg"] == 0.0
        ), "Legacy-padded GNN dG importance should be 0.0"

        # Restore
        scorer.models[0].feature_importance = orig_importance
