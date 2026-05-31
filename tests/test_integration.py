import pytest
try:
    import lightgbm
except ImportError:
    pytest.skip("lightgbm not available", allow_module_level=True)

import pytest
try:
    import mappy
except ImportError:
    pytest.skip("mappy not available", allow_module_level=True)

"""Integration and end-to-end testing suite for PrimerForge CLI."""

import os
from click.testing import CliRunner
import pytest

from primerforge.cli import main


@pytest.fixture
def mock_fasta(tmp_path) -> str:
    """Fixture to generate a small mock reference FASTA file on-the-fly."""
    fasta_content = (
        ">chr1\n"
        "CACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGCATGGAGTCCT\n"
        "GTGGCATCCACGAAACTACCTTCAACTCCATCATGAAGTGTGACGTGGACATCCGCAAAGACCTGTACGCC\n"
        "AACACAGTGCTGTCTGGCGGCACCACCATGTACCCTGGCATTGCTGACAGGATGCAGAAGGAGATCACTGC\n"
        "CCTGGCACCCAGCACAATGAAGATCAAGATCATTGCTCCTCCTGAGCGC\n"
    )
    fasta_file = tmp_path / "integration_ref.fasta"
    fasta_file.write_text(fasta_content)
    return str(fasta_file)


@pytest.fixture
def mock_vcf(tmp_path) -> str:
    """Fixture to generate a mock VCF file containing variants on-the-fly."""
    # Place a SNP at position 210, which corresponds to the 3' end of the reverse primer
    vcf_content = (
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t210\t.\tC\tG\t100\tPASS\tMAF=0.05\n"
    )
    vcf_file = tmp_path / "integration_variants.vcf"
    vcf_file.write_text(vcf_content)
    return str(vcf_file)


def test_cli_design_pure_biophysical() -> None:
    """Tests the CLI design command running purely biophysical thermodynamic design."""
    runner = CliRunner()
    target_seq = (
        "CACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGCATGGAGTCCT"
        "GTGGCATCCACGAAACTACCTTCAACTCCATCATGAAGTGTGACGTGGACATCCGCAAAGACCTGTACGCC"
        "AACACAGTGCTGTCTGGCGGCACCACCATGTACCCTGGCATTGCTGACAGGATGCAGAAGGAGATCACTGC"
        "CCTGGCACCCAGCACAATGAAGATCAAGATCATTGCTCCTCCTGAGCGC"
    )

    result = runner.invoke(
        main,
        [
            "design",
            "--target",
            target_seq,
            "--num-return",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "PRIMERFORGE OPTIMIZED DESIGN RESULTS" in result.output
    assert "[Rank 1]" in result.output
    assert "Forward:" in result.output
    assert "Reverse:" in result.output
    assert "Off-Targets=0" in result.output
    assert "Variant Penalty=0.0" in result.output


def test_cli_design_with_specificity(mock_fasta: str, mock_vcf: str) -> None:
    """Tests the full CLI design command with pangenome specificity and VCF variant checks active."""
    runner = CliRunner()
    target_seq = (
        "CACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGCATGGAGTCCT"
        "GTGGCATCCACGAAACTACCTTCAACTCCATCATGAAGTGTGACGTGGACATCCGCAAAGACCTGTACGCC"
        "AACACAGTGCTGTCTGGCGGCACCACCATGTACCCTGGCATTGCTGACAGGATGCAGAAGGAGATCACTGC"
        "CCTGGCACCCAGCACAATGAAGATCAAGATCATTGCTCCTCCTGAGCGC"
    )

    result = runner.invoke(
        main,
        [
            "design",
            "--target",
            target_seq,
            "--pangenome",
            mock_fasta,
            "--vcf",
            mock_vcf,
            "--num-return",
            "5",
        ],
    )

    assert result.exit_code == 0
    assert "PRIMERFORGE OPTIMIZED DESIGN RESULTS" in result.output
    assert "Off-Targets=" in result.output
    # The VCF SNP pos 210 falls within the 3' extension region of candidate reverse primers,
    # meaning some candidates will be penalized and fail the 3' SNP validation check.
    assert "Variant Penalty=" in result.output


def test_cli_design_fasta_input_file(mock_fasta: str) -> None:
    """Tests the CLI design command when target template is passed as a FASTA file path."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "design",
            "--target",
            mock_fasta,
            "--num-return",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "PRIMERFORGE OPTIMIZED DESIGN RESULTS" in result.output
