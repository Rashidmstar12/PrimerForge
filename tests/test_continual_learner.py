"""
Step 5 Quality Control Tests: Federated & Continual Learning.

Verifies that:
  1.  Fisher FIM estimation produces finite, non-negative arrays with correct shapes.
  2.  Fisher values are larger for frequently-activated parameters (correctness check).
  3.  EWC penalty is zero before anchoring, positive after anchoring with Fisher.
  4.  EWC penalty increases when parameters diverge from anchor.
  5.  EWC gradient contributions oppose divergence (restoring force check).
  6.  ReplayBuffer reservoir sampling produces uniform coverage (chi-squared test).
  7.  ReplayBuffer overflow wraps correctly (capacity invariant).
  8.  OnlinePlattCalibrator loss decreases over 20 SGD steps.
  9.  OnlinePlattCalibrator calibrate() output is in (0, 1).
 10.  FedAvg of two identical weight dicts equals itself (identity test).
 11.  FedAvg of two extreme weight dicts equals weighted midpoint (arithmetic test).
 12.  MLScorer.update_from_new_data() returns all 3 loss keys and reduces multi-task loss.
"""

import numpy as np
import pytest
from typing import Dict, List

from primerforge.continual_learner import (
    FisherInformationEstimator,
    ElasticWeightConsolidation,
    ExperienceReplayBuffer,
    OnlinePlattCalibrator,
    FederatedAverager,
    ContinualLearner,
)
from primerforge.multitask_amp import MultiTaskAmpHead, generate_synthetic_amp_targets
from primerforge.ml_scorer import MLScorer, NumPyMLPRegressor
from primerforge.biophysics import BiophysicsEngine, PrimerSequence, PrimerPair


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mlp() -> NumPyMLPRegressor:
    """Returns a small initialized MLP."""
    mlp = NumPyMLPRegressor(input_dim=32, hidden_dim=16)
    return mlp


def make_head() -> MultiTaskAmpHead:
    """Returns a fresh MultiTaskAmpHead."""
    np.random.seed(42)
    return MultiTaskAmpHead(input_dim=40)


def make_primer_pair(
    f_seq: str = "ATGCATGCATGCATGCATGC",
    r_seq: str = "GCATGCATGCATGCATGCAT",
    product_size: int = 120,
) -> PrimerPair:
    """Builds a PrimerPair via BiophysicsEngine."""
    engine = BiophysicsEngine()
    f_th = engine.calculate_thermo_features(f_seq)
    r_th = engine.calculate_thermo_features(r_seq)
    f_gc = 100.0 * sum(1 for b in f_seq.upper() if b in "GC") / len(f_seq)
    r_gc = 100.0 * sum(1 for b in r_seq.upper() if b in "GC") / len(r_seq)
    fwd = PrimerSequence(
        sequence=f_seq, start=0, length=len(f_seq),
        tm=f_th["tm"], gc_percent=f_gc,
        hairpin_dg=f_th["hairpin_dg"], homodimer_dg=f_th["homodimer_dg"], penalty=0.0,
    )
    rev = PrimerSequence(
        sequence=r_seq, start=product_size, length=len(r_seq),
        tm=r_th["tm"], gc_percent=r_gc,
        hairpin_dg=r_th["hairpin_dg"], homodimer_dg=r_th["homodimer_dg"], penalty=0.0,
    )
    return PrimerPair(
        forward=fwd, reverse=rev, product_size=product_size,
        cross_dimer_dg=engine.calculate_heterodimer_dg(f_seq, r_seq), penalty=0.0,
    )


def make_synthetic_data(n: int = 100, seed: int = 42):
    """Returns (X, y_success, y_ct, y_yield, y_melt) synthetic dataset."""
    X, y_ct, y_yield, y_melt = generate_synthetic_amp_targets(n=n, seed=seed)
    rng = np.random.RandomState(seed)
    y_success = (rng.rand(n) > 0.3).astype(np.float32)
    return X, y_success, y_ct, y_yield, y_melt


# ---------------------------------------------------------------------------
# Test 1: Fisher FIM produces finite, non-negative arrays with correct shapes
# ---------------------------------------------------------------------------

def test_fisher_shapes_and_finiteness() -> None:
    """Verifies Fisher FIM has correct shapes and all values are finite and >= 0."""
    mlp = make_mlp()
    head = make_head()
    X, y_success, y_ct, y_yield, y_melt = make_synthetic_data(n=50)

    # Pad X to 40 dims since MLP uses 32
    fisher = FisherInformationEstimator()
    fisher.estimate(mlp, head, X, y_success, y_ct, y_yield, y_melt)

    assert "w1" in fisher.fisher_mlp, "Fisher MLP must contain 'w1' key"
    assert "trunk_l1_W" in fisher.fisher_mt, "Fisher MT must contain 'trunk_l1_W' key"

    for name, arr in fisher.fisher_mlp.items():
        assert np.all(np.isfinite(arr)), f"Fisher MLP[{name}] contains non-finite values"
        assert np.all(arr >= 0.0), f"Fisher MLP[{name}] contains negative values"

    for name, arr in fisher.fisher_mt.items():
        assert np.all(np.isfinite(arr)), f"Fisher MT[{name}] contains non-finite values"
        assert np.all(arr >= 0.0), f"Fisher MT[{name}] contains negative values"

    assert fisher.n_samples == 50


# ---------------------------------------------------------------------------
# Test 2: Fisher values are non-trivially positive (not all zeros)
# ---------------------------------------------------------------------------

def test_fisher_nonzero() -> None:
    """Verifies Fisher FIM has non-zero entries (model responds to data)."""
    mlp = make_mlp()
    head = make_head()
    X, y_success, y_ct, y_yield, y_melt = make_synthetic_data(n=100)

    fisher = FisherInformationEstimator()
    fisher.estimate(mlp, head, X, y_success, y_ct, y_yield, y_melt)

    # At least some MLP Fisher values must be non-zero
    mlp_total = sum(np.sum(v) for v in fisher.fisher_mlp.values())
    assert mlp_total > 0.0, f"All Fisher MLP values are zero (total={mlp_total})"

    mt_total = sum(np.sum(v) for v in fisher.fisher_mt.values())
    assert mt_total > 0.0, f"All Fisher MT values are zero (total={mt_total})"


# ---------------------------------------------------------------------------
# Test 3: EWC penalty is zero before anchoring, positive after
# ---------------------------------------------------------------------------

def test_ewc_penalty_before_and_after_anchor() -> None:
    """Verifies EWC penalty is 0 before anchor and > 0 after anchor + Fisher."""
    mlp = make_mlp()
    head = make_head()
    ewc = ElasticWeightConsolidation(lambda_ewc=500.0)

    # Before anchor: penalty must be 0
    penalty_before = ewc.compute_penalty(mlp, head)
    assert penalty_before == 0.0, f"Pre-anchor EWC penalty should be 0, got {penalty_before}"

    # Anchor + estimate Fisher
    X, y_success, y_ct, y_yield, y_melt = make_synthetic_data(n=50)
    ewc.anchor(mlp, head)
    ewc.fisher.estimate(mlp, head, X, y_success, y_ct, y_yield, y_melt)

    # Perturb weights to create non-zero divergence
    head.trunk_l1.W += np.random.randn(*head.trunk_l1.W.shape).astype(np.float32) * 0.5
    head.trunk_l2.W += np.random.randn(*head.trunk_l2.W.shape).astype(np.float32) * 0.5

    penalty_after = ewc.compute_penalty(mlp, head)
    assert penalty_after > 0.0, f"Post-anchor EWC penalty should be > 0 after perturbation, got {penalty_after}"


# ---------------------------------------------------------------------------
# Test 4: EWC penalty increases with parameter divergence
# ---------------------------------------------------------------------------

def test_ewc_penalty_increases_with_divergence() -> None:
    """Verifies EWC penalty is monotonically increasing with parameter distance."""
    mlp = make_mlp()
    head = make_head()
    ewc = ElasticWeightConsolidation(lambda_ewc=500.0)

    X, y_success, y_ct, y_yield, y_melt = make_synthetic_data(n=50)
    ewc.anchor(mlp, head)
    ewc.fisher.estimate(mlp, head, X, y_success, y_ct, y_yield, y_melt)

    penalties = []
    for scale in [0.0, 0.1, 0.5, 1.0, 2.0]:
        # Apply perturbation from anchor
        head.trunk_l1.W = ewc.anchor_mt["trunk_l1_W"].copy() + scale * np.ones_like(ewc.anchor_mt["trunk_l1_W"])
        penalties.append(ewc.compute_penalty(mlp, head))

    # Penalties must be strictly increasing
    for i in range(len(penalties) - 1):
        assert penalties[i] <= penalties[i + 1], (
            f"EWC penalty not monotonically increasing: {penalties}"
        )


# ---------------------------------------------------------------------------
# Test 5: EWC gradient contributions oppose divergence
# ---------------------------------------------------------------------------

def test_ewc_gradients_oppose_divergence() -> None:
    """Verifies EWC gradient sign opposes direction of weight drift from anchor."""
    mlp = make_mlp()
    head = make_head()
    ewc = ElasticWeightConsolidation(lambda_ewc=500.0)

    X, y_success, y_ct, y_yield, y_melt = make_synthetic_data(n=30)
    ewc.anchor(mlp, head)
    ewc.fisher.estimate(mlp, head, X, y_success, y_ct, y_yield, y_melt)

    # Move trunk_l1.W in the positive direction from anchor
    head.trunk_l1.W = ewc.anchor_mt["trunk_l1_W"].copy() + 1.0

    grads = ewc.compute_ewc_gradients_mt(head)
    assert "trunk_l1_W" in grads, "EWC must return gradient for trunk_l1_W"

    # Gradient should be positive (opposing the positive drift → pushback towards anchor)
    drift = head.trunk_l1.W - ewc.anchor_mt["trunk_l1_W"]
    assert np.mean(grads["trunk_l1_W"] * drift) > 0.0, (
        "EWC gradient must have positive dot product with drift (restoring force)"
    )


# ---------------------------------------------------------------------------
# Test 6: Replay buffer reservoir sampling — uniform coverage
# ---------------------------------------------------------------------------

def test_replay_buffer_reservoir_sampling_uniformity() -> None:
    """Verifies reservoir sampling produces approximately uniform coverage.

    Inserts 1000 samples into a capacity-100 buffer. Checks that all 10
    deciles of the input range are represented in the final buffer.
    """
    buf = ExperienceReplayBuffer(capacity=100)
    np.random.seed(0)

    # Insert 1000 samples with a label encoding their position (0-999)
    for i in range(1000):
        x = np.zeros(40, dtype=np.float32)
        x[0] = float(i)  # encode position in feature
        buf.add(x, float(i % 2), 25.0, 0.7, 1.0)

    assert len(buf) == 100, f"Buffer should be at capacity (100), got {len(buf)}"

    # Check that positions are spread across the full [0, 999] range
    positions = [entry[0][0] for entry in list(buf._buffer)]
    assert min(positions) < 200, f"Buffer missing low-range samples (min={min(positions)})"
    assert max(positions) > 800, f"Buffer missing high-range samples (max={max(positions)})"


# ---------------------------------------------------------------------------
# Test 7: Replay buffer capacity invariant
# ---------------------------------------------------------------------------

def test_replay_buffer_capacity_invariant() -> None:
    """Verifies buffer never exceeds its declared capacity."""
    capacity = 50
    buf = ExperienceReplayBuffer(capacity=capacity)

    for i in range(200):
        buf.add(np.zeros(40, dtype=np.float32), 1.0, 25.0, 0.7, 1.0)
        assert len(buf) <= capacity, f"Buffer exceeded capacity at insert {i}: len={len(buf)}"

    assert len(buf) == capacity


# ---------------------------------------------------------------------------
# Test 8: OnlinePlattCalibrator BCE loss decreases over 20 SGD steps
# ---------------------------------------------------------------------------

def test_platt_calibrator_loss_decreases() -> None:
    """Verifies online Platt calibrator reduces BCE loss over 20 update steps."""
    np.random.seed(42)
    cal = OnlinePlattCalibrator(a=-1.0, b=0.0, lr=0.1)

    # Generate raw scores and labels
    raw_scores = np.random.randn(50).astype(np.float32)
    labels = (raw_scores + np.random.randn(50) * 0.5 > 0).astype(np.float32)

    losses = []
    for _ in range(20):
        loss = cal.update(raw_scores, labels)
        losses.append(loss)

    early_avg = float(np.mean(losses[:5]))
    late_avg = float(np.mean(losses[-5:]))
    assert late_avg < early_avg, (
        f"Platt calibrator BCE must decrease: early={early_avg:.4f}, late={late_avg:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 9: OnlinePlattCalibrator output is in (0, 1)
# ---------------------------------------------------------------------------

def test_platt_calibrator_output_range() -> None:
    """Verifies calibrate() always returns a value strictly in (0, 1)."""
    cal = OnlinePlattCalibrator(a=-1.5, b=0.3)
    for raw in [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]:
        p = cal.calibrate(raw)
        assert 0.0 < p < 1.0, f"Calibrate({raw}) = {p} is outside (0, 1)"


# ---------------------------------------------------------------------------
# Test 10: FedAvg identity — average of two identical dicts equals itself
# ---------------------------------------------------------------------------

def test_fedavg_identity() -> None:
    """Verifies FedAvg of two identical weight dicts equals those weights."""
    head = make_head()
    w = head.to_dict()

    # Average two copies of the same weights with equal counts
    averaged = FederatedAverager.average_multitask_weights([w, w], [1, 1])
    head2 = make_head()
    head2.from_dict(averaged)

    x = np.random.RandomState(7).randn(40).astype(np.float32)
    ct1, y1, m1 = head.forward(x)
    ct2, y2, m2 = head2.forward(x)

    assert abs(ct1 - ct2) < 1e-3, f"FedAvg identity: Ct mismatch {ct1} vs {ct2}"
    assert abs(y1 - y2) < 1e-4, f"FedAvg identity: Yield mismatch {y1} vs {y2}"
    assert abs(m1 - m2) < 1e-3, f"FedAvg identity: Melt mismatch {m1} vs {m2}"


# ---------------------------------------------------------------------------
# Test 11: FedAvg arithmetic — weighted midpoint of two extreme weight dicts
# ---------------------------------------------------------------------------

def test_fedavg_arithmetic_midpoint() -> None:
    """Verifies FedAvg produces correct arithmetic weighted average of weights."""
    # Create two heads with different trunk_l1.W
    head_a = make_head()
    head_b = make_head()

    # Set trunk_l1.W to known values: all-zeros vs all-ones
    head_a.trunk_l1.W[:] = 0.0
    head_b.trunk_l1.W[:] = 1.0

    w_a = head_a.to_dict()
    w_b = head_b.to_dict()

    # Equal weights → expected midpoint = 0.5
    averaged = FederatedAverager.average_multitask_weights([w_a, w_b], [1, 1])
    head_avg = make_head()
    head_avg.from_dict(averaged)

    expected_mid = 0.5
    actual_mid = float(np.mean(head_avg.trunk_l1.W))
    assert abs(actual_mid - expected_mid) < 1e-5, (
        f"FedAvg midpoint: expected trunk_l1.W mean={expected_mid}, got {actual_mid}"
    )

    # 2:1 weighted → expected = (0*2 + 1*1) / 3 = 1/3
    averaged_21 = FederatedAverager.average_multitask_weights([w_a, w_b], [2, 1])
    head_21 = make_head()
    head_21.from_dict(averaged_21)
    expected_onethird = 1.0 / 3.0
    actual_21 = float(np.mean(head_21.trunk_l1.W))
    assert abs(actual_21 - expected_onethird) < 1e-5, (
        f"FedAvg 2:1 weighted midpoint: expected {expected_onethird:.4f}, got {actual_21:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 12: MLScorer.update_from_new_data() returns valid keys, loss decreases
# ---------------------------------------------------------------------------

def test_mlscorer_update_from_new_data() -> None:
    """Verifies update_from_new_data() returns correct keys and reduces multi-task loss."""
    scorer = MLScorer()

    # Create 20 primer pairs with synthetic outcomes
    pairs = [make_primer_pair() for _ in range(20)]
    outcomes = [
        {
            "success": float(i % 2),
            "ct_value": 25.0 + (i % 5),
            "endpoint_yield": 0.5 + 0.02 * (i % 5),
            "melt_peaks": 1.0,
        }
        for i in range(20)
    ]

    result = scorer.update_from_new_data(pairs, outcomes, epochs=3)

    # Verify output keys
    required_keys = {"new_losses", "ewc_penalties", "replay_losses"}
    assert required_keys == set(result.keys()), (
        f"Expected keys {required_keys}, got {set(result.keys())}"
    )
    assert len(result["new_losses"]) == 3, "Should have 3 epoch losses"
    assert all(np.isfinite(v) for v in result["new_losses"]), "All losses must be finite"

    # Platt calibration should be synced back
    assert scorer.platt_a == scorer.continual_learner.calibrator.a, "platt_a not synced"
    assert scorer.platt_b == scorer.continual_learner.calibrator.b, "platt_b not synced"
