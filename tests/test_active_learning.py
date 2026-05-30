"""Unit tests for the Active Learning & Uncertainty sampling modules in PrimerForge."""

import pytest
import numpy as np
from typing import Dict, Any

from primerforge.biophysics import BiophysicsEngine
from primerforge.ml_scorer import MLScorer
from primerforge.active_learning import BiophysicalOracle, ActiveLearningEngine


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
    model_file = tmp_path / "temp_lgbm_al.model"
    return MLScorer(model_path=str(model_file))


def test_biophysical_oracle(mock_pair) -> None:
    """Tests the BiophysicalOracle outputs and deterministic flags."""
    oracle = BiophysicalOracle(noise_std=0.0)

    # 1. Clean data should yield a success (1)
    spec_clean = {
        "f_off_targets": 0.0,
        "r_off_targets": 0.0,
        "f_var_dist": 20.0,
        "r_var_dist": 20.0,
        "f_var_maf": 0.0,
        "r_var_maf": 0.0
    }
    outcome_clean = oracle.evaluate(mock_pair, spec_clean, deterministic=True)
    assert outcome_clean == 1

    # 2. High off-target count or close variants should penalize and yield fail (0)
    spec_bad = {
        "f_off_targets": 10.0,
        "r_off_targets": 10.0,
        "f_var_dist": 1.0,
        "r_var_dist": 1.0,
        "f_var_maf": 0.99,
        "r_var_maf": 0.99
    }
    outcome_bad = oracle.evaluate(mock_pair, spec_bad, deterministic=True)
    assert outcome_bad == 0


def test_active_learning_engine(scorer: MLScorer, mock_pair) -> None:
    """Tests the ActiveLearningEngine data operations and acquisition scorers."""
    oracle = BiophysicalOracle(noise_std=0.05)
    engine = ActiveLearningEngine(scorer, oracle)

    # Create dummy pool data
    spec1 = {"f_off_targets": 0, "r_off_targets": 0}
    spec2 = {"f_off_targets": 2, "r_off_targets": 1}
    spec3 = {"f_off_targets": 0, "r_off_targets": 0, "r_var_dist": 2.0, "r_var_maf": 0.8}

    unlabeled = [
        (mock_pair, spec1),
        (mock_pair, spec2),
        (mock_pair, spec3)
    ]
    engine.load_unlabeled_pool(unlabeled)
    assert len(engine.unlabeled_pool) == 3

    # Initial seed data
    initial_seeds = [
        (mock_pair, spec1, 1),
        (mock_pair, spec2, 0)
    ]
    engine.load_initial_labeled_data(initial_seeds)
    assert len(engine.labeled_pool) == 2

    # Verify score shapes and values for different strategies
    for strategy in ["random", "entropy", "epistemic", "aleatoric", "hybrid"]:
        scores = engine.compute_acquisition_scores(strategy)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        assert np.all(~np.isnan(scores))

    # Query next batch
    queried = engine.query_and_label_next_batch(batch_size=2, strategy="entropy", deterministic=True)
    assert len(queried) == 2
    assert len(engine.unlabeled_pool) == 1
    assert len(engine.labeled_pool) == 4

    # Retrain ensemble
    # Add a few more mock samples to satisfy the minimum retraining constraint
    extra_data = [
        (mock_pair, spec1, 1),
        (mock_pair, spec1, 1),
        (mock_pair, spec1, 1),
        (mock_pair, spec2, 0),
        (mock_pair, spec2, 0),
        (mock_pair, spec2, 0)
    ]
    engine.load_initial_labeled_data(extra_data)
    
    # Run retraining
    engine.retrain_ensemble()

    # Verify that the underlying scorer models list is updated
    assert len(scorer.models) > 0
