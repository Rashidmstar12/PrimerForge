import pytest
try:
    import mappy
except ImportError:
    pytest.skip("mappy not available", allow_module_level=True)

"""Unit tests for the DataCurationPipeline real-data scraper and hybrid training in PrimerForge."""

import pytest
import os
import pandas as pd
import numpy as np
from primerforge.data_curation import (
    DataCurationPipeline,
    PubMedPMCXMLParser,
    PatentXMLParser,
    GelVisionAnalyzerStub,
    MeltCurveAnalyzerStub,
)


@pytest.fixture
def pipeline() -> DataCurationPipeline:
    """Fixture to initialize DataCurationPipeline."""
    return DataCurationPipeline(data_dir="data")


def test_scrape_and_curate_real_data(pipeline: DataCurationPipeline) -> None:
    """Verifies the schema, columns, and properties of the curated real data."""
    target_size = 15
    df = pipeline.scrape_and_curate_real_data(target_size=target_size)

    # 1. Check size
    assert len(df) == target_size

    # 2. Check Unified Schema columns
    expected_cols = [
        "species",
        "chromosome",
        "forward_seq",
        "reverse_seq",
        "target_id",
        "source_db",
        "pcr_type",
        "polymerase",
        "additive_dmso",
        "mg_conc_mm",
        "efficiency",
        "ct_value",
        "specificity",
        "success_idx",
        "success",
    ]
    for col in expected_cols:
        assert col in df.columns

    # 3. Check biophysical feature columns exist
    assert "f_tm" in df.columns
    assert "cross_dimer_dg" in df.columns
    assert "f_clamp_gc" in df.columns
    assert "f_poly_run" in df.columns

    # 4. Check data types are correct
    assert isinstance(df["forward_seq"].iloc[0], str)
    assert isinstance(df["success_idx"].iloc[0], (float, np.float32, np.float64))
    assert isinstance(df["efficiency"].iloc[0], (float, np.float32, np.float64))


def test_hybrid_dataset_splitting(pipeline: DataCurationPipeline) -> None:
    """Verifies that partition_and_save creates correctly sized training and testing splits."""
    real_df = pipeline.scrape_and_curate_real_data(target_size=10)
    synthetic_df = pipeline.generate_empirical_db(n_samples=10)

    hybrid_df = pd.concat([real_df, synthetic_df], ignore_index=True)

    X_train, y_train, X_test, y_test = pipeline.partition_and_save(hybrid_df)

    # Verify return dimensions
    assert X_train.shape[1] == 36
    assert X_test.shape[1] == 36
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == 20


def test_scrape_real_data_ultra(pipeline: DataCurationPipeline) -> None:
    """Verifies the schema, columns, and properties of the curated ultra-scale real data."""
    target_size = 15
    df = pipeline.scrape_real_data_ultra(target_size=target_size)

    # 1. Check size
    assert len(df) == target_size

    # 2. Check Unified Schema columns + extensions
    expected_cols = [
        "species",
        "chromosome",
        "forward_seq",
        "reverse_seq",
        "target_id",
        "source_db",
        "pcr_type",
        "polymerase",
        "polymerase_encoded",
        "additive_dmso",
        "mg_conc_mm",
        "efficiency",
        "ct_value",
        "specificity",
        "success_idx",
        "success",
        "salt_monovalent_mm",
        "salt_divalent_mm",
        "dntp_conc_mm",
        "uncertainty_interval",
    ]
    for col in expected_cols:
        assert col in df.columns

    # 3. Check biophysical feature columns exist
    assert "f_tm" in df.columns
    assert "cross_dimer_dg" in df.columns
    assert "f_clamp_gc" in df.columns

    # 4. Check data types are correct
    assert isinstance(df["forward_seq"].iloc[0], str)
    assert isinstance(df["success_idx"].iloc[0], (float, np.float32, np.float64))
    assert isinstance(df["salt_monovalent_mm"].iloc[0], (float, np.float32, np.float64))


def test_scrape_real_data_live_ultra(pipeline: DataCurationPipeline) -> None:
    """Verifies that scrape_real_data_live_ultra successfully curates the 1M unified schema."""
    target_size = 10
    df = pipeline.scrape_real_data_live_ultra(target_size=target_size)

    # 1. Check size
    assert len(df) == target_size

    # 2. Check Unified Schema columns + extensions
    expected_cols = [
        "species",
        "chromosome",
        "forward_seq",
        "reverse_seq",
        "target_id",
        "source_db",
        "pcr_type",
        "polymerase",
        "polymerase_encoded",
        "additive_dmso",
        "mg_conc_mm",
        "efficiency",
        "ct_value",
        "specificity",
        "success_idx",
        "success",
        "salt_monovalent_mm",
        "salt_divalent_mm",
        "dntp_conc_mm",
        "uncertainty_interval",
    ]
    for col in expected_cols:
        assert col in df.columns

    # 3. Check data types are correct
    assert isinstance(df["forward_seq"].iloc[0], str)
    assert isinstance(df["success_idx"].iloc[0], (float, np.float32, np.float64))


def test_multi_modal_parsers() -> None:
    """Verifies the parsing logic of PubMed PMC XML, Patents XML, and vision/melt stubs."""
    # 1. PMC XML Parser
    mock_pmc_xml = "<article><front><p>forward primer>GCACTCTTCCAGCCTTCCTT<</p><p>reverse primer>CTGTGTTGGCGTACAGGTCT<</p></front></article>"
    pmc_records = PubMedPMCXMLParser.parse_article_xml(mock_pmc_xml)
    assert len(pmc_records) == 1
    assert pmc_records[0]["source_db"] == "pubmed_pmc"
    assert len(pmc_records[0]["forward_seq"]) == 20

    # 2. Patent XML Parser
    mock_patent_xml = "<patent-document><forward_seq>GCACTCTTCCAGCCTTCCTTG</forward_seq><reverse_seq>CTGTGTTGGCGTACAGGTCTA</reverse_seq></patent-document>"
    patent_records = PatentXMLParser.parse_patent_xml(mock_patent_xml)
    assert len(patent_records) == 1
    assert patent_records[0]["source_db"] == "patents"
    assert len(patent_records[0]["forward_seq"]) == 21

    # 3. Gel Vision Analyzer
    mock_gel_image = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR..."
    gel_res = GelVisionAnalyzerStub.analyze_gel_image(mock_gel_image)
    assert "specificity_score" in gel_res
    assert gel_res["gel_outcome"] in ["Single_Peak", "Primer_Dimer", "Multi_Peak"]

    # 4. Melt Curve Telemetry Analyzer
    # Single Peak
    single_peak_telemetry = [0.01, 0.02, 0.05, 0.10, 0.20, 0.85, 0.20, 0.08, 0.02, 0.01]
    melt_res = MeltCurveAnalyzerStub.analyze_melt_curve(single_peak_telemetry)
    assert melt_res["single_peak"] is True
    assert melt_res["specificity_index"] == 0.99

    # Double Peak
    double_peak_telemetry = [0.01, 0.02, 0.35, 0.10, 0.05, 0.20, 0.65, 0.20, 0.05, 0.01]
    melt_res_double = MeltCurveAnalyzerStub.analyze_melt_curve(double_peak_telemetry)
    assert melt_res_double["single_peak"] is False
    assert melt_res_double["specificity_index"] == 0.30
