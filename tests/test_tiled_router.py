"""Unit tests for the TiledAmpliconRouter module in PrimerForge."""

import pytest
from typing import Any, Dict, List
from unittest.mock import patch

from primerforge.biophysics import BiophysicsEngine, PrimerPair, PrimerSequence
from primerforge.ml_scorer import MLScorer
from primerforge.optimizer import TiledAmpliconRouter


@pytest.fixture
def biophys() -> BiophysicsEngine:
    """Fixture to initialize a BiophysicsEngine."""
    return BiophysicsEngine()


@pytest.fixture
def ml_scorer() -> MLScorer:
    """Fixture to initialize a MLScorer."""
    return MLScorer()


@pytest.fixture
def router(biophys, ml_scorer) -> TiledAmpliconRouter:
    """Fixture to initialize a TiledAmpliconRouter."""
    return TiledAmpliconRouter(biophys_engine=biophys, ml_scorer=ml_scorer)


def test_tiled_router_window_generation(router: TiledAmpliconRouter) -> None:
    """Verifies that the router correctly divides a template into sliding windows."""
    # Length 1000 sequence, tile_size 400, overlap 50 -> step 350
    # Expected windows:
    # 0: 0-400
    # 1: 350-750
    # 2: 650-1000 (Adjusted last window)
    long_seq = "A" * 1000

    # Let's mock _generate_tile_candidates to return mock pairs so the router completes
    mock_seq = "ATGCGATCGATCGATCGATC"
    mock_pair = PrimerPair(
        forward=PrimerSequence(mock_seq, 0, 20, 60.0, 50.0, 0.0, 0.0, 0.0),
        reverse=PrimerSequence(mock_seq, 100, 20, 60.0, 50.0, 0.0, 0.0, 0.0),
        product_size=120,
        cross_dimer_dg=0.0,
        penalty=0.0,
    )

    with patch.object(
        TiledAmpliconRouter, "_generate_tile_candidates", return_value=[mock_pair]
    ):
        tiles = router.design_tiled_amplicons(long_seq, tile_size=400, overlap=50)

        # We have three windows, each should design exactly one tile
        assert len(tiles) == 3

        # Check window coordinates project to absolute coordinates
        # Tile 0: win_start=0, forward.start=0, product_size=120 -> abs_start=0, abs_end=120
        assert tiles[0]["abs_start"] == 0
        assert tiles[0]["abs_end"] == 120

        # Tile 1: win_start=350, forward.start=0, product_size=120 -> abs_start=350, abs_end=470
        assert tiles[1]["abs_start"] == 350
        assert tiles[1]["abs_end"] == 470


def test_tiled_router_overlap_constraints(router: TiledAmpliconRouter) -> None:
    """Verifies that the router avoids paths that violate overlap constraints."""
    long_seq = "A" * 700

    # Let's create mock pairs where candidate 0 for window 1 overlaps perfectly,
    # but candidate 1 has a gap (no overlap).
    pair_good = PrimerPair(
        forward=PrimerSequence(
            "ATGCGATCGATCGATCGATC", 10, 20, 60.0, 50.0, 0.0, 0.0, 0.0
        ),
        reverse=PrimerSequence(
            "ATGCGATCGATCGATCGATC", 390, 20, 60.0, 50.0, 0.0, 0.0, 0.0
        ),
        product_size=380,
        cross_dimer_dg=0.0,
        penalty=0.0,
    )

    pair_bad = PrimerPair(
        forward=PrimerSequence(
            "ATGCGATCGATCGATCGATC", 300, 20, 60.0, 50.0, 0.0, 0.0, 0.0
        ),
        reverse=PrimerSequence(
            "ATGCGATCGATCGATCGATC", 390, 20, 60.0, 50.0, 0.0, 0.0, 0.0
        ),
        product_size=90,
        cross_dimer_dg=0.0,
        penalty=0.0,
    )

    def side_effect(sub_seq: str, tile_size: int, num_return: int) -> List[PrimerPair]:
        if "tile_0" in sub_seq or len(sub_seq) == 400:
            return [pair_good]
        # For the second window (start at 300), return a good pair and a bad pair
        return [pair_good, pair_bad]

    with patch.object(
        TiledAmpliconRouter, "_generate_tile_candidates", side_effect=side_effect
    ):
        tiles = router.design_tiled_amplicons(long_seq, tile_size=400, overlap=50)
        assert len(tiles) == 2
        # Good tile 0 ends at: 0 + 10 + 380 = 390
        # Good tile 1 starts at: 300 + 10 = 310 -> Overlap = 80bp (Within constraints)
        # Bad tile 1 starts at: 300 + 300 = 600 -> Overlap = 390 - 600 = -210bp (Gap! Violates constraints)
        # The router must choose the good tile for the second window!
        assert tiles[1]["abs_start"] == 310


def test_tiled_router_relaxed_fallback(router: TiledAmpliconRouter) -> None:
    """Verifies that the router runs the relaxed fallback DP when constraints are too tight."""
    long_seq = "A" * 700

    # Two amplicons that have no way of overlapping (large gap)
    pair = PrimerPair(
        forward=PrimerSequence(
            "ATGCGATCGATCGATCGATC", 10, 20, 60.0, 50.0, 0.0, 0.0, 0.0
        ),
        reverse=PrimerSequence(
            "ATGCGATCGATCGATCGATC", 50, 20, 60.0, 50.0, 0.0, 0.0, 0.0
        ),
        product_size=40,
        cross_dimer_dg=0.0,
        penalty=0.0,
    )

    with patch.object(
        TiledAmpliconRouter, "_generate_tile_candidates", return_value=[pair]
    ):
        # Overlap = 50, but amplicons end at 50, and next starts at 300 -> gap of 250bp
        # The strict DP solver will fail because overlap is <= 0.
        # It must trigger the relaxed fallback and successfully return tiles!
        tiles = router.design_tiled_amplicons(long_seq, tile_size=400, overlap=50)
        assert len(tiles) == 2
        assert tiles[0]["abs_start"] == 10
        assert tiles[1]["abs_start"] == 310
