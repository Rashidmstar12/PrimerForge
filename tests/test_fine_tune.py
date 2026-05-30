"""Unit tests for the ensembled transfer learning fine-tuning module in PrimerForge."""

import os
import pytest
import pandas as pd
import numpy as np

from primerforge.ml_scorer import MLScorer


@pytest.fixture
def mock_user_data() -> pd.DataFrame:
    """Fixture to generate a mock user wet-lab outcomes database (N=10 primer pairs)."""
    # 10 biologically realistic sequences with positive/negative wet-lab outcomes
    data = {
        "forward_seq": [
            "CACCATTGGCAATGAGCGGT", "CACCATTGGCAATGAGCGGT", "CACCATTGGCAATGAGCGGT",
            "CACCATTGGCAATGAGCGGT", "CACCATTGGCAATGAGCGGT", "CACCATTGGCAATGAGCGGT",
            "CACCATTGGCAATGAGCGGT", "CACCATTGGCAATGAGCGGT", "CACCATTGGCAATGAGCGGT",
            "CACCATTGGCAATGAGCGGT"
        ],
        "reverse_seq": [
            "CGCTCAGGAGGAGCAATGAT", "CGCTCAGGAGGAGCAATGAT", "CGCTCAGGAGGAGCAATGAT",
            "CGCTCAGGAGGAGCAATGAT", "CGCTCAGGAGGAGCAATGAT", "CGCTCAGGAGGAGCAATGAT",
            "CGCTCAGGAGGAGCAATGAT", "CGCTCAGGAGGAGCAATGAT", "CGCTCAGGAGGAGCAATGAT",
            "CGCTCAGGAGGAGCAATGAT"
        ],
        "success": [0.95, 0.90, 0.05, 0.98, 0.12, 0.88, 0.95, 0.08, 0.85, 0.92],
        "product_size": [150] * 10,
        "f_off_targets": [0.0] * 10,
        "r_off_targets": [0.0] * 10,
        "f_var_dist": [20.0] * 10,
        "r_var_dist": [20.0] * 10,
        "salt_monovalent_mm": [50.0] * 10,
        "salt_divalent_mm": [1.5] * 10,
        "dntp_conc_mm": [0.2] * 10,
        "polymerase": ["Standard_Taq"] * 10
    }
    return pd.DataFrame(data)


def test_fine_tune_on_user_data(tmp_path, mock_user_data) -> None:
    """Verifies that fine-tuning executes cleanly, computes metrics, and serializes ensembled assets."""
    model_file = tmp_path / "temp_lgbm.model"
    scorer = MLScorer(model_path=str(model_file))

    # Pre-train an MVP model to ensure boosters exist in self.models
    if not scorer.models:
        scorer.train_mvp_model()
        scorer.load()

    out_dir = tmp_path / "fine_tuned"
    
    # Run ensembled transfer learning
    results = scorer.fine_tune_on_user_data(mock_user_data, str(out_dir))

    # 1. Assert comparative metrics dictionary holds proper keys and floats
    assert isinstance(results, dict)
    for key in ["Brier_Before", "ECE_Before", "Brier_After", "ECE_After", "platt_a", "platt_b"]:
        assert key in results
        assert isinstance(results[key], float)

    # 2. Assert serialized fine-tuned models exist under the custom directory
    expected_ultra_prefix = out_dir / "primerforge_lightgbm_ultra"
    for idx in range(len(scorer.models)):
        expected_path = f"{str(expected_ultra_prefix)}_{idx}.model"
        assert os.path.exists(expected_path)

    # 3. Assert calibration parameter and MLP weights JSON is created
    calib_json = out_dir / "primerforge_lightgbm_ultra_calib.json"
    assert os.path.exists(calib_json)

    # 4. Verify that we can instantiate and load a fresh MLScorer targeting the fine-tuned assets
    new_model_file = out_dir / "primerforge_lightgbm.model"
    new_scorer = MLScorer(model_path=str(new_model_file))
    
    assert len(new_scorer.models) == len(scorer.models)
    assert new_scorer.platt_a == scorer.platt_a
    assert new_scorer.platt_b == scorer.platt_b
    assert hasattr(new_scorer.mlp, "w1")
