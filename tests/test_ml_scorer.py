"""Unit tests for the MLScorer module in PrimerForge."""

import os
import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np

from primerforge.biophysics import BiophysicsEngine
from primerforge.ml_scorer import MLScorer


@pytest.fixture
def mock_pair():
    """Fixture to generate a standard PrimerPair using BiophysicsEngine."""
    engine = BiophysicsEngine()
    target_seq = (
        "CACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGCATGGAGTCCT"
        "GTGGCATCCACGAAACTACCTTCAACTCCATCATGAAGTGTGACGTGGACATCCGCAAAGACCTGTACGCC"
        "AACACAGTGCTGTCTGGCGGCACCACCATGTACCCTGGCATTGCTGACAGGATGCAGAAGGAGATCACTGC"
        "CCTGGCACCCAGCACAATGAAGATCAAGATCATTGCTCCTCCTGAGCGC"
    )
    pairs = engine.generate_candidates(target_seq, num_return=1)
    return pairs[0]


@pytest.fixture
def scorer(tmp_path) -> MLScorer:
    """Fixture to initialize an MLScorer targeting a temporary model path."""
    model_file = tmp_path / "temp_lgbm.model"
    return MLScorer(model_path=str(model_file))


def test_feature_extraction(scorer: MLScorer, mock_pair) -> None:
    """Tests the 36-dimensional feature extraction algorithm."""
    spec_data = {
        "f_off_targets": 0.0,
        "r_off_targets": 1.0,
        "f_var_dist": 20.0,
        "r_var_dist": 3.0,
        "f_var_maf": 0.0,
        "r_var_maf": 0.05,
    }

    features = scorer.extract_features(mock_pair, spec_data)

    assert isinstance(features, list)
    assert len(features) == 40
    for val in features:
        assert isinstance(val, float)


def test_predict_success(scorer: MLScorer, mock_pair) -> None:
    """Tests probability predictions returned by the LightGBM booster."""
    spec_data = {
        "f_off_targets": 0.0,
        "r_off_targets": 0.0,
        "f_var_dist": 20.0,
        "r_var_dist": 20.0,
        "f_var_maf": 0.0,
        "r_var_maf": 0.0,
    }

    prob = scorer.predict_success(mock_pair, spec_data)

    assert isinstance(prob, float)
    assert 0.01 <= prob <= 0.99


def test_model_serialization(tmp_path) -> None:
    """Verifies that the GBDT regressor trains, saves, and loads correctly from disk."""
    model_file = tmp_path / "serialization_test.model"
    if os.path.exists(model_file):
        os.remove(model_file)

    # Instantiation should trigger training and save the booster
    scorer = MLScorer(model_path=str(model_file))
    assert os.path.exists(model_file)
    assert scorer.model is not None

    # Load from disk and verify
    new_scorer = MLScorer(model_path=str(model_file))
    assert new_scorer.model is not None


def test_train_full_model(tmp_path) -> None:
    """Tests GBDT full model retraining utilizing mock database curation pipelines."""
    model_file = tmp_path / "full_model_test.model"
    if os.path.exists(model_file):
        os.remove(model_file)

    scorer = MLScorer(model_path=str(model_file))

    # 1. Generate the small mock DataFrame using the real, unmocked pipeline generator
    from primerforge.data_curation import DataCurationPipeline

    pipeline = DataCurationPipeline(data_dir=str(tmp_path))
    df = pipeline.generate_empirical_db(n_samples=200)

    # 2. Enter patch contexts to mock for train_full_model execution
    with patch.object(DataCurationPipeline, "generate_empirical_db") as mock_db:
        mock_db.return_value = df

        # We patch the initialization of the DataCurationPipeline to target the tmp_path instead of 'data/'
        with patch.object(
            DataCurationPipeline,
            "__init__",
            lambda self, data_dir=str(tmp_path): setattr(self, "data_dir", data_dir),
        ):
            scorer.train_full_model()

    assert os.path.exists(model_file)
    assert scorer.model is not None


def test_predict_success_with_uncertainty(scorer: MLScorer, mock_pair) -> None:
    """Verifies that predict_success_with_uncertainty returns calibrated mean and std dev."""
    mean_pred, std_pred = scorer.predict_success_with_uncertainty(mock_pair)
    assert isinstance(mean_pred, float)
    assert isinstance(std_pred, float)
    assert 0.01 <= mean_pred <= 0.99
    assert std_pred >= 0.0


def test_train_ultra_hybrid_model(tmp_path) -> None:
    """Tests ensembled ultra-scale hybrid model training and serialization."""
    model_file = tmp_path / "ultra_model_test.model"
    scorer = MLScorer(model_path=str(model_file))

    from primerforge.data_curation import DataCurationPipeline

    pipeline = DataCurationPipeline(data_dir=str(tmp_path))

    # Generate the actual DataFrames before patching
    real_df_data = pipeline.scrape_real_data_ultra(target_size=20)
    synthetic_df_data = pipeline.generate_empirical_db(n_samples=20)

    # Mock the internal generators to use pre-generated DataFrames
    with patch.object(
        DataCurationPipeline, "scrape_real_data_ultra"
    ) as mock_scrape, patch.object(
        DataCurationPipeline, "generate_empirical_db"
    ) as mock_db, patch.object(
        DataCurationPipeline,
        "__init__",
        lambda self, data_dir=str(tmp_path): setattr(self, "data_dir", data_dir),
    ):

        mock_scrape.return_value = real_df_data
        mock_db.return_value = synthetic_df_data

        scorer.train_ultra_hybrid_model(target_size=20, n_samples=20)

    # Verify ensembled models are saved in tmp_path
    for idx in range(3):
        expected_path = os.path.join(
            str(tmp_path), f"primerforge_lightgbm_ultra_{idx}.model"
        )
        assert os.path.exists(expected_path)


def test_train_ultra_ensemble(tmp_path) -> None:
    """Tests the advanced stacked ensemble training with Platt calibration and MLP sequence model."""
    model_file = tmp_path / "ultra_ensemble_test.model"
    scorer = MLScorer(model_path=str(model_file))

    from primerforge.data_curation import DataCurationPipeline

    pipeline = DataCurationPipeline(data_dir=str(tmp_path))

    real_df_data = pipeline.scrape_real_data_live_ultra(target_size=20)
    synthetic_df_data = pipeline.generate_empirical_db(n_samples=20)

    with patch.object(
        DataCurationPipeline, "scrape_real_data_live_ultra"
    ) as mock_scrape, patch.object(
        DataCurationPipeline, "generate_empirical_db"
    ) as mock_db, patch.object(
        DataCurationPipeline,
        "__init__",
        lambda self, data_dir=str(tmp_path): setattr(self, "data_dir", data_dir),
    ):

        mock_scrape.return_value = real_df_data
        mock_db.return_value = synthetic_df_data

        scorer.train_ultra_ensemble(target_size=20, n_samples=20)

    # 1. Verify standard boosters are saved
    ultra_path = os.path.join(str(tmp_path), "primerforge_lightgbm_ultra")
    # GBDT reg boosters (3) + GBDT quantile boosters (2) = 5 total models
    for idx in range(5):
        assert os.path.exists(f"{ultra_path}_{idx}.model")

    # 2. Verify Platt Calibration and MLP JSON file is saved
    calib_json = os.path.join(str(tmp_path), "primerforge_lightgbm_ultra_calib.json")
    assert os.path.exists(calib_json)

    # 3. Reload and verify MLP weights and Platt coefficients exist
    new_scorer = MLScorer(model_path=str(model_file))
    assert len(new_scorer.models) == 5
    assert new_scorer.platt_a != -1.0 or new_scorer.platt_b != 0.0
    assert hasattr(new_scorer.mlp, "w1")


def test_explain_prediction(scorer: MLScorer, mock_pair) -> None:
    """Verifies that explain_prediction computes SHAP values for all 36 biophysical features."""
    spec_data = {
        "f_off_targets": 0.0,
        "r_off_targets": 0.0,
        "f_var_dist": 20.0,
        "r_var_dist": 20.0,
        "f_var_maf": 0.0,
        "r_var_maf": 0.0,
    }

    # Ensure model is fitted
    if not scorer.models:
        scorer.train_mvp_model()

    shap_explanations = scorer.explain_prediction(mock_pair, spec_data)

    assert isinstance(shap_explanations, dict)
    assert (
        len(shap_explanations) == 40
    )  # 36 biophysical + 2 GNN + 2 transformer features

    # Verify all expected columns exist in the output dictionary
    expected_cols = [
        "f_tm",
        "r_tm",
        "tm_diff",
        "f_hairpin_dg",
        "r_hairpin_dg",
        "f_homodimer_dg",
        "r_homodimer_dg",
        "cross_dimer_dg",
        "f_gc",
        "r_gc",
        "f_len",
        "r_len",
        "f_clamp_gc",
        "r_clamp_gc",
        "f_poly_run",
        "r_poly_run",
        "f_3_dinuc_gc",
        "r_3_dinuc_gc",
        "f_3_dinuc_aa",
        "f_3_dinuc_tt",
        "r_3_dinuc_aa",
        "r_3_dinuc_tt",
        "f_3_stability",
        "r_3_stability",
        "target_mfe",
        "target_gc",
        "target_len",
        "primer_overlap",
        "f_off_targets",
        "r_off_targets",
        "f_var_dist",
        "r_var_dist",
        "salt_monovalent_mm",
        "salt_divalent_mm",
        "dntp_conc_mm",
        "polymerase_encoded",
        # GNN-derived biophysical predictions (Step 3: Graph Neural Network integration)
        "gnn_pred_tm",
        "gnn_pred_dg",
        # Transformer features
        "transformer_p_success",
        "transformer_confidence",
    ]
    for col in expected_cols:
        assert col in shap_explanations
        assert isinstance(shap_explanations[col], float)
