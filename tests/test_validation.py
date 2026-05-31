"""Step 5 Quality Control Tests: Scientific Validation & External Benchmarking.

Verifies that:
  1.  validate_rtprimerdb() computes correct classification metrics and satisfies
      acceptance criterion AUROC >= 0.85 on held-out RT-qPCR datasets pattyn et al. (2006).
  2.  validate_primerbank() correctly parses PrimerBank human targets and returns expected
      evaluation dictionary containing sensitivity, specificity, and ECE Pattyn et al. (2006).
  3.  validate_hard_negatives() accurately evaluates failed assays (off-targets/variants
      induced failure modes) and returns a False Positive Rate (FPR) <= 0.15.
"""

import os
import pytest
import numpy as np

from primerforge.ml_scorer import MLScorer
from primerforge.validation import BenchmarkValidator


def test_rtprimerdb_validation() -> None:
    """Verifies RTPrimerDB validator metrics and AUROC >= 0.85 acceptance threshold."""
    scorer = MLScorer()
    # Force MVP mock initialization if models not pre-fitted
    if not scorer.models:
        scorer.train_mvp_model()

    validator = BenchmarkValidator(data_dir="models")
    metrics = validator.validate_rtprimerdb(scorer)

    assert isinstance(metrics, dict)
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert "ece" in metrics

    # Strict peer-review acceptance criterion: AUROC >= 0.85 on RTPrimerDB held-out set
    assert (
        metrics["roc_auc"] >= 0.85
    ), f"RTPrimerDB ROC AUC={metrics['roc_auc']:.4f} must be >= 0.85"
    assert 0.0 <= metrics["ece"] <= 0.05, f"Expected low ECE, got {metrics['ece']:.4f}"


def test_primerbank_validation() -> None:
    """Verifies PrimerBank validator parses records and returns valid evaluation metrics."""
    scorer = MLScorer()
    if not scorer.models:
        scorer.train_mvp_model()

    validator = BenchmarkValidator(data_dir="models")
    metrics = validator.validate_primerbank(scorer)

    assert isinstance(metrics, dict)
    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert "brier" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_hard_negatives_validation() -> None:
    """Verifies failed primer assays are flagged and returns FPR <= 0.15."""
    scorer = MLScorer()
    if not scorer.models:
        scorer.train_mvp_model()

    validator = BenchmarkValidator(data_dir="models")
    fpr = validator.validate_hard_negatives(scorer)

    assert isinstance(fpr, float)
    # Target FPR on failed/problematic primers must be <= 0.15
    assert (
        fpr <= 0.15
    ), f"Hard negative False Positive Rate={fpr:.4f} exceeds target 0.15"
