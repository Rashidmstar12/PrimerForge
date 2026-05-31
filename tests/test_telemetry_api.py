"""Step 4 (Lab Automation) Quality Control Tests: Telemetry API and SDM Cq Parsing.

Verifies:
  1. Central difference derivatives match expected curves.
  2. Sigmoidal raw fluorescence data returns valid Cq via SDM.
  3. Noise-only negative control raw cycles return Cq = 40.0, is_success = False.
  4. Melt curve negative derivative successfully counts single vs multiple peaks.
  5. FastAPI REST endpoint `/api/v1/telemetry/ingest` end-to-end.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from primerforge.web.telemetry_api import (
    app,
    analyze_qpcr_curve,
    analyze_melt_curve,
)


@pytest.fixture
def test_client() -> TestClient:
    return TestClient(app)


def test_health_check(test_client) -> None:
    """Verifies the health check API is alive and responsive."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze_qpcr_curve_sigmoidal() -> None:
    """Verifies that a simulated sigmoidal curve successfully extracts a valid Cq."""
    # Simulate a classic sigmoidal amplification curve: f(t) = baseline + max_val / (1 + exp(-k * (t - Cq)))
    Cq_true = 22.0
    baseline = 10.0
    max_val = 100.0
    k = 0.5

    cycles = []
    # Add a tiny bit of random noise
    np.random.seed(42)
    for t in range(40):
        val = baseline + max_val / (1.0 + np.exp(-k * (t - Cq_true)))
        noise = np.random.normal(0, 0.05)
        cycles.append(float(val + noise))

    cq, max_deriv, is_success = analyze_qpcr_curve(cycles)

    assert is_success is True
    # The SDM Cq peak is close to the true Cq value inflection point
    assert abs(cq - Cq_true) <= 3.0
    assert max_deriv > 5.0


def test_analyze_qpcr_curve_negative_control() -> None:
    """Verifies that noise-only negative control cycles return Cq=40 and is_success=False."""
    # Simulate flat noise-only negative control
    np.random.seed(99)
    cycles = [float(10.0 + np.random.normal(0, 0.1)) for _ in range(40)]

    cq, max_deriv, is_success = analyze_qpcr_curve(cycles)

    assert is_success is False
    assert cq == 40.0
    assert max_deriv < 1.0


def test_analyze_melt_curve_single_vs_double_peak() -> None:
    """Verifies melt peak counting detects single pure peaks vs double multiplex contamination peaks."""
    # 1. Single pure peak at T=78C
    temps = list(np.arange(60.0, 95.0, 1.0))
    # Gaussian peak: F(T) = exp(-((T - 78) / 3)^2)
    single_deriv = [float(np.exp(-((t - 78.0) / 3.0) ** 2)) for t in temps]
    # Reconstruct raw fluorescence as cumulative negative sum of negative derivative
    single_fluor = list(np.cumsum(single_deriv)[::-1])

    peaks = analyze_melt_curve(temps, single_fluor)
    assert peaks == 1

    # 2. Double peak (primer dimers) at T=72C (dimer) and T=82C (target)
    double_deriv = [
        float(np.exp(-((t - 72.0) / 2.0) ** 2) + 0.8 * np.exp(-((t - 82.0) / 3.0) ** 2))
        for t in temps
    ]
    double_fluor = list(np.cumsum(double_deriv)[::-1])

    peaks_double = analyze_melt_curve(temps, double_fluor)
    assert peaks_double == 2


def test_telemetry_ingestion_endpoint(test_client) -> None:
    """Verifies end-to-end FastAPI endpoint ingestion and EWC automatic refitting."""
    # Create simulated curves for 2 experiments: 1 successful, 1 negative control
    Cq_true = 20.0
    baseline = 10.0
    max_val = 120.0
    k = 0.6

    cycles_pos = []
    np.random.seed(12)
    for t in range(40):
        val = baseline + max_val / (1.0 + np.exp(-k * (t - Cq_true)))
        cycles_pos.append(float(val + np.random.normal(0, 0.02)))

    cycles_neg = [float(10.0 + np.random.normal(0, 0.05)) for _ in range(40)]

    payload = {
        "experiments": [
            {
                "forward_seq": "ATGCATGCATGCATGC",
                "reverse_seq": "GCATGCATGCATGCAT",
                "fluorescence_cycles": cycles_pos,
                "temperature_dissociation": [60.0, 70.0, 80.0, 90.0],
                "fluorescence_dissociation": [10.0, 8.0, 3.0, 0.1],
            },
            {
                "forward_seq": "AAAATTTTGCGCATGC",
                "reverse_seq": "GCGCGCGCATATATAT",
                "fluorescence_cycles": cycles_neg,
            },
        ]
    }

    response = test_client.post("/api/v1/telemetry/ingest", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["processed"] == 2

    # Verify results
    res_pos = data["results"][0]
    assert res_pos["empirical_success"] is True
    assert abs(res_pos["Cq"] - Cq_true) <= 3.0

    res_neg = data["results"][1]
    assert res_neg["empirical_success"] is False
    assert res_neg["Cq"] == 40.0

    # Verify ensembled calibration metrics are returned
    metrics = data["calibration_metrics"]
    assert "new_losses" in metrics
    assert "ewc_penalties" in metrics
    assert "platt_a" in metrics
