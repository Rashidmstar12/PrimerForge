"""Unit tests for the SpecificityEngine and VariantAwareFilter modules in PrimerForge."""

import os
import pytest
from primerforge.specificity import SpecificityEngine, VariantAwareFilter, AlignmentHit


@pytest.fixture
def mock_fasta(tmp_path) -> str:
    """Fixture to generate a small mock reference FASTA file on-the-fly."""
    fasta_content = (
        ">chr1\n"
        "CACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGCATGGAGTCCT\n"
        ">chr2\n"
        "GATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC\n"
    )
    fasta_file = tmp_path / "mock_ref.fasta"
    fasta_file.write_text(fasta_content)
    return str(fasta_file)


@pytest.fixture
def mock_vcf(tmp_path) -> str:
    """Fixture to generate a mock VCF file containing variants on-the-fly."""
    vcf_content = (
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t10\t.\tT\tC\t100\tPASS\tMAF=0.05\n"
        "chr1\t68\t.\tC\tA\t100\tPASS\tAF=0.02\n"  # Variant inside forward primer 3' clamp (pos 68)
        "chr1\t2\t.\tA\tG\t100\tPASS\tAF=0.04\n"  # Variant inside reverse primer 3' clamp (pos 2)
    )
    vcf_file = tmp_path / "mock_variants.vcf"
    vcf_file.write_text(vcf_content)
    return str(vcf_file)


def test_index_and_pangenome_load(mock_fasta: str) -> None:
    """Verifies that reference genomes are parsed and indexed correctly."""
    engine = SpecificityEngine()
    engine.index_pangenome(mock_fasta)

    assert engine.fasta_path == mock_fasta
    # Check that in-memory fallback loader successfully read contigs
    assert "chr1" in engine._fallback_db
    assert "chr2" in engine._fallback_db
    assert len(engine._fallback_db["chr1"]) == 71


def test_check_specificity_exact(mock_fasta: str) -> None:
    """Verifies exact primer sequence alignment matching."""
    engine = SpecificityEngine()
    engine.index_pangenome(mock_fasta)

    # Exact match on chr1 (forward strand orientation)
    primer = "GAGGCACTCTTCCAGCC"
    hits = engine.check_specificity(primer, max_mismatches=0)

    assert len(hits) >= 1
    hit = hits[0]
    assert isinstance(hit, AlignmentHit)
    assert hit.contig == "chr1"
    assert hit.strand == 1
    assert hit.mismatches == 0


def test_check_specificity_mismatches(mock_fasta: str) -> None:
    """Verifies primer sequence alignment matching with allowed mismatches."""
    engine = SpecificityEngine()
    engine.index_pangenome(mock_fasta)

    # Sequence containing 2 mismatches compared to target chr1 ("GAGGCACTCTTCCAGCC")
    primer_mismatch = "GAcGCACaCTTCCAGCC"

    # Assert it gets rejected if max_mismatches = 0
    hits_fail = engine.check_specificity(primer_mismatch, max_mismatches=0)
    assert len(hits_fail) == 0

    # Assert it gets aligned if max_mismatches = 2
    hits_pass = engine.check_specificity(primer_mismatch, max_mismatches=2)
    assert len(hits_pass) >= 1
    assert hits_pass[0].mismatches == 2


def test_variant_filter_eval(mock_vcf: str) -> None:
    """Verifies VariantAwareFilter loaded records and coordinate checks."""
    var_filter = VariantAwareFilter()
    var_filter.load_variants(mock_vcf)

    assert len(var_filter.variants) == 3
    assert var_filter.variants[0].chrom == "chr1"
    assert var_filter.variants[0].pos == 10
    assert var_filter.variants[0].maf == 0.05

    # 1. Forward primer: starts at 50, len 20 (bounds 50-70).
    # Pos 68 has a high MAF variant, which falls in the critical 3' end (last 5bp).
    penalty_f, valid_f = var_filter.evaluate_primer(
        primer_seq="TCCAGCCTTCCTTCCTGGGC", start_pos=50, strand=1, maf_threshold=0.01
    )
    assert not valid_f  # Should be flagged as invalid due to 3' anchor SNP
    assert penalty_f >= 100.0

    # 2. Reverse primer: starts at 0, len 20 (bounds 0-20).
    # Strand -1 (reverse complement). 3' end corresponds to start region (low coordinate).
    # Pos 2 has a high MAF variant, which falls in the critical 3' end.
    penalty_r, valid_r = var_filter.evaluate_primer(
        primer_seq="CACCATTGGCAATGAGCGGT", start_pos=0, strand=-1, maf_threshold=0.01
    )
    assert not valid_r  # Should be flagged as invalid
    assert penalty_r >= 100.0

    # 3. Clean primer context: primer lies in region 20-40, no variants
    penalty_c, valid_c = var_filter.evaluate_primer(
        primer_seq="TTCCTTCCTGGGCATGGAGT", start_pos=20, strand=1, maf_threshold=0.01
    )
    assert valid_c
    assert penalty_c == 0.0
