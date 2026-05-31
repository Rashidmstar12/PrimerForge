"""Unit tests for the pre-trained weights loading, shape validation, and MLScorer integration."""

import os
import json
import pytest
import numpy as np

from primerforge.transformer import DNATransformerEncoder
from primerforge.ml_scorer import MLScorer


@pytest.fixture
def temp_weights_path(tmp_path) -> str:
    """Fixture to generate a temporary valid weights JSON file."""
    transformer = DNATransformerEncoder(
        vocab_size=8, embed_dim=16, num_heads=2, hidden_dim=32, max_len=24
    )
    weights_dict = transformer.to_dict()
    output_path = tmp_path / "dna_transformer_valid.json"
    with open(output_path, "w") as f:
        json.dump(weights_dict, f)
    return str(output_path)


def test_load_valid_pretrained_weights(temp_weights_path) -> None:
    """Verifies that DNATransformerEncoder can load valid pre-trained weights and sets the loaded flag."""
    transformer = DNATransformerEncoder(
        vocab_size=8,
        embed_dim=16,
        num_heads=2,
        hidden_dim=32,
        max_len=24,
        pretrained_weights_path=temp_weights_path,
    )
    assert transformer.pretrained_loaded is True


def test_load_malformed_missing_keys(tmp_path, temp_weights_path) -> None:
    """Verifies that from_dict raises ValueError when required weight keys are missing."""
    with open(temp_weights_path, "r") as f:
        data = json.load(f)

    # Delete a required key
    del data["emb_W"]

    malformed_path = tmp_path / "dna_transformer_missing.json"
    with open(malformed_path, "w") as f:
        json.dump(data, f)

    transformer = DNATransformerEncoder(
        vocab_size=8, embed_dim=16, num_heads=2, hidden_dim=32, max_len=24
    )
    with pytest.raises(ValueError, match="Missing required key 'emb_W'"):
        transformer.from_dict(data)


def test_load_shape_mismatch(tmp_path, temp_weights_path) -> None:
    """Verifies that from_dict raises ValueError when a loaded matrix has an incorrect shape."""
    with open(temp_weights_path, "r") as f:
        data = json.load(f)

    # Change shape of embedding matrix from (8, 16) to (10, 16)
    data["emb_W"] = np.zeros((10, 16)).tolist()

    mismatch_path = tmp_path / "dna_transformer_mismatch.json"
    with open(mismatch_path, "w") as f:
        json.dump(data, f)

    transformer = DNATransformerEncoder(
        vocab_size=8, embed_dim=16, num_heads=2, hidden_dim=32, max_len=24
    )
    with pytest.raises(ValueError, match="Shape mismatch for 'emb_W'"):
        transformer.from_dict(data)


def test_ml_scorer_integration() -> None:
    """Verifies that MLScorer automatically detects and loads pre-trained weights if present in models/."""
    # Instantiating MLScorer should cleanly load the packaged pre-trained weights
    scorer = MLScorer()
    assert hasattr(scorer.transformer, "pretrained_loaded")

    # If the pre-trained weights file is present in models/, it should be loaded successfully
    if os.path.exists("models/dna_transformer_pretrained.json"):
        assert scorer.transformer.pretrained_loaded is True
