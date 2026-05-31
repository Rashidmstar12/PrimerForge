import os
os.environ["PRIMERFORGE_NO_AUTOTRAIN"] = "1"

import pytest
from unittest.mock import MagicMock

try:
    from primerforge.ml_scorer import MLScorer
    
    orig_load = MLScorer.load
    orig_save = MLScorer.save
    
    def patched_load(self):
        # Only perform loading if the path is in a sandboxed temp/pytest directory
        path_str = str(self.model_path).lower()
        if "temp" in path_str or "pytest" in path_str or "mock" in path_str:
            return orig_load(self)
        return None

    def patched_save(self):
        # Only perform saving if the path is in a sandboxed temp/pytest directory
        path_str = str(self.model_path).lower()
        if "temp" in path_str or "pytest" in path_str or "mock" in path_str:
            return orig_save(self)
        return None

    # Globally patch load/save with smart sandboxed filtering
    MLScorer.load = patched_load
    MLScorer.save = patched_save
except ImportError:
    pass

@pytest.fixture
def minimal_ml_scorer():
    """Returns a minimal MLScorer instance that does not require model files on disk."""
    try:
        from primerforge.ml_scorer import MLScorer
        scorer = MLScorer(model_path="dummy_path.model")
        scorer.model = MagicMock()
        scorer.models = [scorer.model]
        return scorer
    except ImportError:
        pytest.skip("lightgbm not available")
