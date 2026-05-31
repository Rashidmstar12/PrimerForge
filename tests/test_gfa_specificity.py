import pytest
try:
    import mappy
except ImportError:
    pytest.skip("mappy not available", allow_module_level=True)

"""Unit tests for Phase 3 Pangenome Graph (GFA) Specificity Engine."""

import os
import pytest
import numpy as np

from primerforge.specificity import SpecificityEngine, AlignmentHit


@pytest.fixture
def temp_gfa_file(tmp_path) -> str:
    """Fixture to generate a temporary GFA-1 file with a bubble (variation)."""
    gfa_content = (
        "H\tVN:Z:1.0\n"
        "S\tseg1\tCAAATAAG\n"
        "S\tseg2\tGCGCTCAG\n"
        "S\tseg3\tTCCGCTGC\n"
        "S\tseg_var\tGCGCTAAG\n"  # parallel segment representing variation
        "L\tseg1\t+\tseg2\t+\t0M\n"
        "L\tseg1\t+\tseg_var\t+\t0M\n"
        "L\tseg2\t+\tseg3\t+\t0M\n"
        "L\tseg_var\t+\tseg3\t+\t0M\n"
    )
    gfa_file = tmp_path / "test_pangenome.gfa"
    with open(gfa_file, "w") as f:
        f.write(gfa_content)
    return str(gfa_file)


@pytest.fixture
def temp_gfa_file_overlap(tmp_path) -> str:
    """Fixture to generate a GFA-1 file with overlapping links."""
    gfa_content = (
        "H\tVN:Z:1.0\n"
        "S\t1\tATGCATGC\n"
        "S\t2\tTGCATGCG\n"  # has overlap with segment 1
        "L\t1\t+\t2\t+\t5M\n"  # 5-bp overlap
    )
    gfa_file = tmp_path / "test_overlap.gfa"
    with open(gfa_file, "w") as f:
        f.write(gfa_content)
    return str(gfa_file)


def test_gfa_parsing(temp_gfa_file) -> None:
    """Verifies that GFA parser reads segment nodes, orientation edges, and header lines correctly."""
    engine = SpecificityEngine()
    engine.index_pangenome(temp_gfa_file)

    assert engine.gfa_path == temp_gfa_file
    assert len(engine.graph.segments) == 4
    assert engine.graph.segments["seg1"] == "CAAATAAG"
    assert engine.graph.segments["seg2"] == "GCGCTCAG"
    assert engine.graph.segments["seg3"] == "TCCGCTGC"
    assert engine.graph.segments["seg_var"] == "GCGCTAAG"

    # Verify links
    assert ("seg1", "+") in engine.graph.adjacency
    assert ("seg2", "+") in engine.graph.adjacency
    assert ("seg_var", "+") in engine.graph.adjacency


def test_gfa_overlap_parsing(temp_gfa_file_overlap) -> None:
    """Verifies that GFA parser correctly parses link overlaps and performs correct path sequence assembly."""
    engine = SpecificityEngine()
    engine.index_pangenome(temp_gfa_file_overlap)

    assert len(engine.graph.segments) == 2
    # seg1 = ATGCATGC, seg2 = TGCATGCG, overlap 5M -> sequence of seg2 appended should be seg2[5:] = 'GCG'
    # So sequence path 1+ -> 2+ is 'ATGCATGC' + 'GCG' = 'ATGCATGCGCG'
    traversals = engine.graph.traverse_local_paths(max_len=30)
    assembled_seqs = [t[0] for t in traversals]
    assert any("ATGCATGCGCG" in seq for seq in assembled_seqs)


def test_check_specificity_gfa_exact_match(temp_gfa_file) -> None:
    """Verifies that exact match queries return alignment hits with correct segment name and coordinate."""
    engine = SpecificityEngine()
    engine.index_pangenome(temp_gfa_file)

    # Segment 1 exact match
    hits = engine.check_specificity("CAAATAAG")
    assert len(hits) > 0
    hit = hits[0]
    assert hit.contig == "seg1"
    assert hit.start == 0
    assert hit.end == 8
    assert hit.strand == 1
    assert hit.mismatches == 0


def test_check_specificity_gfa_junction_match(temp_gfa_file) -> None:
    """Verifies that primer overlapping a GFA node transition (link junction) is aligned across segment boundaries."""
    engine = SpecificityEngine()
    engine.index_pangenome(temp_gfa_file)

    # seg1 (CAAATAAG) + seg2 (GCGCTCAG) -> sequence "CAAATAAGGCGCTCAG"
    # primer "TAAGGCGC" crosses junction (length 8)
    hits = engine.check_specificity("TAAGGCGC")
    assert len(hits) > 0

    # It should hit both segment 1 (at position 4) and segment 2 (at position 0) because it crosses the junction!
    contigs = [h.contig for h in hits]
    assert "seg1" in contigs
    assert "seg2" in contigs

    seg1_hit = next(h for h in hits if h.contig == "seg1")
    assert seg1_hit.start == 4
    assert seg1_hit.end == 8
    assert seg1_hit.strand == 1


def test_check_specificity_gfa_mismatches(temp_gfa_file) -> None:
    """Verifies that sliding window alignment correctly reports mismatch counts up to max_mismatches."""
    engine = SpecificityEngine()
    engine.index_pangenome(temp_gfa_file)

    # seg1 is CAAATAAG. query CAAACAAG has 1 mismatch
    hits = engine.check_specificity("CAAACAAG", max_mismatches=2)
    assert len(hits) > 0
    assert hits[0].mismatches == 1

    # query CGGACAAG has 3 mismatches. Should be filtered out if max_mismatches=1
    hits_filtered = engine.check_specificity("CGGACAAG", max_mismatches=1)
    assert len(hits_filtered) == 0


def test_check_specificity_gfa_bubble_branching(temp_gfa_file) -> None:
    """Verifies that GFA alignment checks both alleles in a branching variant bubble."""
    engine = SpecificityEngine()
    engine.index_pangenome(temp_gfa_file)

    # seg2 has "GCGCTCAG"
    hits_seg2 = engine.check_specificity("GCGCTCAG")
    assert len(hits_seg2) > 0
    assert any(h.contig == "seg2" for h in hits_seg2)

    # seg_var has "GCGCTAAG"
    hits_seg_var = engine.check_specificity("GCGCTAAG")
    assert len(hits_seg_var) > 0
    assert any(h.contig == "seg_var" for h in hits_seg_var)
