"""Unit tests for the DataCurationPipeline module in PrimerForge."""

import os
import pytest
import pandas as pd
import numpy as np

from primerforge.data_curation import DataCurationPipeline


@pytest.fixture
def pipeline(tmp_path) -> DataCurationPipeline:
    """Fixture to initialize a DataCurationPipeline targeting a temporary directory."""
    return DataCurationPipeline(data_dir=str(tmp_path))


def test_generate_empirical_db(pipeline: DataCurationPipeline) -> None:
    """Tests the curation of the PrimerForge-Empirical-DB."""
    # Test with N=100 for extremely fast unit testing execution
    df = pipeline.generate_empirical_db(n_samples=100)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "species" in df.columns
    assert "chromosome" in df.columns
    assert "success" in df.columns
    assert "f_tm" in df.columns
    assert "r_tm" in df.columns
    assert "cross_dimer_dg" in df.columns
    assert "f_off_targets" in df.columns
    assert "f_var_dist" in df.columns

    # Verify positive/negative balance
    pos_count = len(df[df["success"] >= 0.85])
    assert pos_count > 0


def test_partition_and_save(pipeline: DataCurationPipeline, tmp_path) -> None:
    """Tests strict species/chromosomal split logic and database serialization."""
    df = pipeline.generate_empirical_db(n_samples=100)
    X_train, y_train, X_test, y_test = pipeline.partition_and_save(df)

    # Check file serialization on disk
    assert os.path.exists(os.path.join(str(tmp_path), "primerforge_empirical_db.csv"))
    assert os.path.exists(os.path.join(str(tmp_path), "X_train.npy"))
    assert os.path.exists(os.path.join(str(tmp_path), "y_train.npy"))
    assert os.path.exists(os.path.join(str(tmp_path), "X_test.npy"))
    assert os.path.exists(os.path.join(str(tmp_path), "y_test.npy"))

    # Assert shape matching
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1] == 36
    assert X_test.shape[1] == 36
