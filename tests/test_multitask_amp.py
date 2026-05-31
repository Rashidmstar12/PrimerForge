import pytest
try:
    import lightgbm
except ImportError:
    pytest.skip("lightgbm not available", allow_module_level=True)

"""
Step 4 Quality Control Tests: Multi-Task Amplification Profiling.

Verifies that:
  1.  MultiTaskAmpHead forward() returns 3 finite floats.
  2.  Ct value is in [CT_MIN, CT_MAX] = [15, 40].
  3.  Endpoint yield is in [0, 1].
  4.  Melt peak count is in [MELT_MIN, MELT_MAX] = [1, 6].
  5.  Analytical gradients for trunk_l1.W match finite differences (numerical verification).
  6.  Analytical gradients for ct_out.W match finite differences.
  7.  MultiTaskAmpHead loss decreases over 30 training epochs (convergence guaranteed
      by sigmoid normalized outputs — no hard clamp gradient blocking).
  8.  JSON weight serialization round-trip preserves forward() output exactly.
  9.  MLScorer.predict_amplification_profile() returns all 4 required keys.
 10.  generate_synthetic_amp_targets() returns correct shapes and biologically plausible ranges.
"""

import numpy as np
import pytest
from typing import Tuple

from primerforge.multitask_amp import MultiTaskAmpHead, generate_synthetic_amp_targets
from primerforge.biophysics import BiophysicsEngine, PrimerSequence, PrimerPair
from primerforge.ml_scorer import MLScorer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def make_multitask_head() -> MultiTaskAmpHead:
    """Returns a freshly initialized MultiTaskAmpHead."""
    np.random.seed(42)
    return MultiTaskAmpHead(input_dim=40)


def make_random_features(seed: int = 0) -> np.ndarray:
    """Returns a random 40-dim feature vector."""
    return np.random.RandomState(seed).randn(40).astype(np.float32)


def make_primer_pair(
    f_seq: str = "ATGCATGCATGCATGCATGC",
    r_seq: str = "GCATGCATGCATGCATGCAT",
    product_size: int = 120,
) -> PrimerPair:
    """Builds a PrimerPair using the correct BiophysicsEngine API."""
    engine = BiophysicsEngine()
    f_th = engine.calculate_thermo_features(f_seq)
    r_th = engine.calculate_thermo_features(r_seq)
    f_gc = 100.0 * sum(1 for b in f_seq.upper() if b in "GC") / len(f_seq)
    r_gc = 100.0 * sum(1 for b in r_seq.upper() if b in "GC") / len(r_seq)
    fwd = PrimerSequence(
        sequence=f_seq,
        start=0,
        length=len(f_seq),
        tm=f_th["tm"],
        gc_percent=f_gc,
        hairpin_dg=f_th["hairpin_dg"],
        homodimer_dg=f_th["homodimer_dg"],
        penalty=0.0,
    )
    rev = PrimerSequence(
        sequence=r_seq,
        start=product_size,
        length=len(r_seq),
        tm=r_th["tm"],
        gc_percent=r_gc,
        hairpin_dg=r_th["hairpin_dg"],
        homodimer_dg=r_th["homodimer_dg"],
        penalty=0.0,
    )
    return PrimerPair(
        forward=fwd,
        reverse=rev,
        product_size=product_size,
        cross_dimer_dg=engine.calculate_heterodimer_dg(f_seq, r_seq),
        penalty=0.0,
    )


# ---------------------------------------------------------------------------
# Test 1: Forward pass returns 3 finite floats
# ---------------------------------------------------------------------------


def test_forward_returns_three_finite_floats() -> None:
    """Verifies that forward() returns exactly 3 finite float values."""
    head = make_multitask_head()
    x = make_random_features(0)
    ct, yield_, melt = head.forward(x)
    assert isinstance(ct, float) and np.isfinite(
        ct
    ), f"Ct must be finite float, got {ct}"
    assert isinstance(yield_, float) and np.isfinite(
        yield_
    ), f"Yield must be finite float, got {yield_}"
    assert isinstance(melt, float) and np.isfinite(
        melt
    ), f"Melt must be finite float, got {melt}"


# ---------------------------------------------------------------------------
# Test 2: Ct value is in [15, 40]
# ---------------------------------------------------------------------------


def test_ct_value_range() -> None:
    """Verifies Ct value is within [15.0, 40.0] for diverse inputs."""
    head = make_multitask_head()
    for seed in range(20):
        x = np.random.RandomState(seed).randn(40).astype(np.float32) * 10
        ct, _, _ = head.forward(x)
        assert (
            MultiTaskAmpHead.CT_MIN <= ct <= MultiTaskAmpHead.CT_MAX
        ), f"Ct={ct:.3f} is outside [{MultiTaskAmpHead.CT_MIN}, {MultiTaskAmpHead.CT_MAX}]"


# ---------------------------------------------------------------------------
# Test 3: Endpoint yield is in [0, 1]
# ---------------------------------------------------------------------------


def test_yield_range() -> None:
    """Verifies endpoint yield is always in [0.0, 1.0]."""
    head = make_multitask_head()
    for seed in range(20):
        x = np.random.RandomState(seed).randn(40).astype(np.float32) * 10
        _, yield_, _ = head.forward(x)
        assert 0.0 <= yield_ <= 1.0, f"Yield={yield_:.4f} is outside [0, 1]"


# ---------------------------------------------------------------------------
# Test 4: Melt peak count is in [1, 6]
# ---------------------------------------------------------------------------


def test_melt_peaks_range() -> None:
    """Verifies melt peak count is always in [1.0, 6.0]."""
    head = make_multitask_head()
    for seed in range(20):
        x = np.random.RandomState(seed).randn(40).astype(np.float32) * 10
        _, _, melt = head.forward(x)
        assert (
            MultiTaskAmpHead.MELT_MIN <= melt <= MultiTaskAmpHead.MELT_MAX
        ), f"MeltPeaks={melt:.2f} is outside [{MultiTaskAmpHead.MELT_MIN}, {MultiTaskAmpHead.MELT_MAX}]"


# ---------------------------------------------------------------------------
# Test 5: Analytical gradient check — trunk L1 weight W (finite differences)
# ---------------------------------------------------------------------------


def test_trunk_l1_gradient_numerical() -> None:
    """Verifies trunk_l1 weight gradient against finite differences.

    Uses sigmoid outputs (no hard clamp), so gradients are always non-zero
    and the finite-difference check works from random initialization.
    """
    np.random.seed(42)
    head = make_multitask_head()
    x = np.random.randn(40).astype(np.float32)
    ct_t, y_t, m_t = 28.0, 0.7, 2.0

    # Normalized targets (internal loss space)
    ct_t_norm = head._ct_to_norm(ct_t)
    melt_t_norm = head._melt_to_norm(m_t)

    # Run backward to get analytic gradients
    head.backward(x, ct_t, y_t, m_t)
    dW_analytic = head.trunk_l1.dW.copy()

    eps = 1e-4
    dW_numeric = np.zeros_like(head.trunk_l1.W)

    def loss_fn() -> float:
        head.forward(x)
        s_ct = float(head._ct_sig[0, 0])
        s_yield = float(head._yield_sig[0, 0])
        s_melt = float(head._melt_sig[0, 0])
        return (
            head.w_ct * (s_ct - ct_t_norm) ** 2
            + head.w_yield * (s_yield - y_t) ** 2
            + head.w_melt * (s_melt - melt_t_norm) ** 2
        )

    # Check a 4x4 subblock to keep test fast
    for i in range(4):
        for j in range(4):
            old = head.trunk_l1.W[i, j]
            head.trunk_l1.W[i, j] = old + eps
            loss_plus = loss_fn()
            head.trunk_l1.W[i, j] = old - eps
            loss_minus = loss_fn()
            head.trunk_l1.W[i, j] = old
            dW_numeric[i, j] = (loss_plus - loss_minus) / (2 * eps)

    assert np.allclose(dW_analytic[:4, :4], dW_numeric[:4, :4], atol=1e-2, rtol=1e-2), (
        "Trunk L1 analytic gradients diverge from finite differences.\n"
        f"Analytic:\n{dW_analytic[:4, :4]}\nNumeric:\n{dW_numeric[:4, :4]}"
    )


# ---------------------------------------------------------------------------
# Test 6: Analytical gradient check — Ct output head weight W
# ---------------------------------------------------------------------------


def test_ct_output_head_gradient_numerical() -> None:
    """Verifies ct_out weight gradient against finite differences.

    Uses sigmoid outputs so gradients are non-zero from random initialization.
    """
    np.random.seed(7)
    head = make_multitask_head()
    x = np.random.randn(40).astype(np.float32)
    ct_t, y_t, m_t = 25.0, 0.8, 1.0

    ct_t_norm = head._ct_to_norm(ct_t)
    melt_t_norm = head._melt_to_norm(m_t)

    # Run backward
    head.backward(x, ct_t, y_t, m_t)
    dW_analytic = head.ct_out.dW.copy()

    eps = 1e-4
    dW_numeric = np.zeros_like(head.ct_out.W)

    def loss_fn() -> float:
        head.forward(x)
        s_ct = float(head._ct_sig[0, 0])
        s_yield = float(head._yield_sig[0, 0])
        s_melt = float(head._melt_sig[0, 0])
        return (
            head.w_ct * (s_ct - ct_t_norm) ** 2
            + head.w_yield * (s_yield - y_t) ** 2
            + head.w_melt * (s_melt - melt_t_norm) ** 2
        )

    for i in range(head.ct_out.W.shape[0]):
        for j in range(head.ct_out.W.shape[1]):
            old = head.ct_out.W[i, j]
            head.ct_out.W[i, j] = old + eps
            loss_plus = loss_fn()
            head.ct_out.W[i, j] = old - eps
            loss_minus = loss_fn()
            head.ct_out.W[i, j] = old
            dW_numeric[i, j] = (loss_plus - loss_minus) / (2 * eps)

    assert np.allclose(dW_analytic, dW_numeric, atol=5e-3, rtol=5e-3), (
        f"Ct output head analytic gradients diverge from finite differences.\n"
        f"Analytic:\n{dW_analytic.T}\nNumeric:\n{dW_numeric.T}"
    )


# ---------------------------------------------------------------------------
# Test 7: Training convergence (loss decreases over 30 epochs)
# ---------------------------------------------------------------------------


def test_training_convergence() -> None:
    """Verifies that training reduces the joint multi-task loss over 30 epochs.

    Sigmoid-normalized outputs guarantee gradient flow from any initialization,
    making convergence deterministic.
    """
    np.random.seed(42)
    head = MultiTaskAmpHead(input_dim=40)

    X, Y_ct, Y_yield, Y_melt = generate_synthetic_amp_targets(n=400, seed=42)

    losses = head.train(X, Y_ct, Y_yield, Y_melt, epochs=30, lr=2e-3, batch_size=32)

    assert len(losses) == 30, f"Expected 30 loss values, got {len(losses)}"
    early_avg = float(np.mean(losses[:5]))
    late_avg = float(np.mean(losses[-5:]))
    assert (
        late_avg < early_avg
    ), f"MultiTaskAmpHead must converge: early_avg={early_avg:.5f}, late_avg={late_avg:.5f}"


# ---------------------------------------------------------------------------
# Test 8: JSON serialization round-trip
# ---------------------------------------------------------------------------


def test_json_serialization_roundtrip() -> None:
    """Verifies that to_dict() / from_dict() round-trip preserves forward() exactly."""
    np.random.seed(99)
    head = make_multitask_head()
    x = np.random.randn(40).astype(np.float32)

    ct_before, y_before, m_before = head.forward(x)

    data = head.to_dict()
    head2 = MultiTaskAmpHead(input_dim=40)
    head2.from_dict(data)

    ct_after, y_after, m_after = head2.forward(x)

    assert (
        abs(ct_before - ct_after) < 1e-4
    ), f"Ct round-trip mismatch: {ct_before} vs {ct_after}"
    assert (
        abs(y_before - y_after) < 1e-5
    ), f"Yield round-trip mismatch: {y_before} vs {y_after}"
    assert (
        abs(m_before - m_after) < 1e-4
    ), f"Melt round-trip mismatch: {m_before} vs {m_after}"


# ---------------------------------------------------------------------------
# Test 9: MLScorer.predict_amplification_profile() returns all 4 keys
# ---------------------------------------------------------------------------


def test_predict_amplification_profile_keys() -> None:
    """Verifies predict_amplification_profile() returns a dict with all 4 required keys."""
    scorer = MLScorer()
    pair = make_primer_pair()
    profile = scorer.predict_amplification_profile(pair)

    assert isinstance(
        profile, dict
    ), "predict_amplification_profile() must return a dict."
    required_keys = {"ct_value", "endpoint_yield", "melt_peaks", "success_prob"}
    assert required_keys == set(
        profile.keys()
    ), f"Profile dict must have exactly keys {required_keys}, got {set(profile.keys())}"

    assert (
        MultiTaskAmpHead.CT_MIN <= profile["ct_value"] <= MultiTaskAmpHead.CT_MAX
    ), f"ct_value={profile['ct_value']} out of range"
    assert (
        0.0 <= profile["endpoint_yield"] <= 1.0
    ), f"endpoint_yield={profile['endpoint_yield']} out of range"
    assert (
        MultiTaskAmpHead.MELT_MIN <= profile["melt_peaks"] <= MultiTaskAmpHead.MELT_MAX
    ), f"melt_peaks={profile['melt_peaks']} out of range"
    assert (
        0.01 <= profile["success_prob"] <= 0.99
    ), f"success_prob={profile['success_prob']} out of range"


# ---------------------------------------------------------------------------
# Test 10: generate_synthetic_amp_targets() shapes and ranges
# ---------------------------------------------------------------------------


def test_synthetic_data_shapes_and_ranges() -> None:
    """Verifies generate_synthetic_amp_targets() produces correctly shaped and bounded data."""
    N = 500
    X, Y_ct, Y_yield, Y_melt = generate_synthetic_amp_targets(n=N, seed=0)

    assert X.shape == (N, 40), f"Feature matrix should be ({N}, 40), got {X.shape}"
    assert Y_ct.shape == (N,), f"Ct targets should be ({N},), got {Y_ct.shape}"
    assert Y_yield.shape == (N,), f"Yield targets should be ({N},), got {Y_yield.shape}"
    assert Y_melt.shape == (N,), f"Melt targets should be ({N},), got {Y_melt.shape}"

    assert np.all(Y_ct >= MultiTaskAmpHead.CT_MIN) and np.all(
        Y_ct <= MultiTaskAmpHead.CT_MAX
    ), f"Y_ct contains values outside [{MultiTaskAmpHead.CT_MIN}, {MultiTaskAmpHead.CT_MAX}]"
    assert np.all(Y_yield >= 0.0) and np.all(
        Y_yield <= 1.0
    ), "Y_yield contains values outside [0, 1]"
    assert np.all(Y_melt >= MultiTaskAmpHead.MELT_MIN) and np.all(
        Y_melt <= MultiTaskAmpHead.MELT_MAX
    ), f"Y_melt contains values outside [{MultiTaskAmpHead.MELT_MIN}, {MultiTaskAmpHead.MELT_MAX}]"
    assert np.all(np.isfinite(X)), "Feature matrix contains NaN or Inf values"
