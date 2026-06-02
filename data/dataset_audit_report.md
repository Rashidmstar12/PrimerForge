# PrimerForge Training Dataset Audit Report

**Date Generated**: 2026-06-02

This document contains the publication-ready data validation and quality control audit for `data/master_training_db_v2.csv`.

---
## Section 1 — Dataset Overview

| Parameter | Category | Count | Percentage |
| :--- | :--- | :---: | :---: |
| **Total Primer Pairs** | All | 3,380 | 100.00% |
| **Source DB** | primerbank | 2720 | 80.47% |
|  | synthetic | 450 | 13.31% |
|  | origene | 112 | 3.31% |
|  | artic | 95 | 2.81% |
|  | rtprimerdb | 3 | 0.09% |
| **Label** | label=1 (Functional) | 2929 | 86.66% |
| | label=0 (Non-Functional) | 451 | 13.34% |
| **Confidence** | High | 209 | 6.18% |
| | Medium | 3171 | 93.82% |
| | Low | 0 | 0.00% |
| **Negative Type** | Synthetic Negatives | 450 | 13.31% |

## Section 2 — Sequence Statistics

| Metric | Mean ± Std | Min | Max |
| :--- | :---: | :---: | :---: |
| **Forward Length (nt)** | 21.21 ± 1.82 | 19.0 | 34.0 |
| **Reverse Length (nt)** | 21.37 ± 1.88 | 19.0 | 33.0 |
| **GC Content (%)** | 50.45% ± 7.68% | 10.0% | 73.7% |
| **Melting Temperature (°C)** | 57.30°C ± 2.44°C | 40.9°C | 69.2°C |
| **Amplicon Size (bp)** | 146.90 ± 69.02 | 60.0 - 481.0 |

*Generated plots saved to:* `plots/gc_distribution.png`, `plots/tm_distribution.png`, and `plots/amplicon_size_distribution.png`.

## Section 3 — Label Quality Assessment

- **qPCR Amplification Efficiency (label=1)**: 1.940 ± 0.028 (N=2)
- **Flagged Near-Duplicate Pairs**: 2

### Source Database × Label Confidence Crosstabulation
| source_db   |   high |   medium |
|:------------|-------:|---------:|
| PrimerBank  |      0 |     2720 |
| artic       |     95 |        0 |
| origene     |    112 |        0 |
| rtprimerdb  |      2 |        1 |
| synthetic   |      0 |      450 |

### Biophysical Features Column Completeness
| Column Name | Non-NaN Completeness (%) |
| :--- | :---: |
| `f_tm` | 100.0% |
| `r_tm` | 100.0% |
| `tm_diff` | 100.0% |
| `f_gc` | 100.0% |
| `r_gc` | 100.0% |
| `f_hairpin_dg` | 100.0% |
| `r_hairpin_dg` | 100.0% |
| `cross_dimer_dg` | 100.0% |
| `f_len` | 100.0% |
| `r_len` | 100.0% |
| `f_clamp_gc` | 100.0% |
| `r_clamp_gc` | 100.0% |
| `f_poly_run` | 100.0% |
| `r_poly_run` | 100.0% |
| `target_gc` | 100.0% |
| `target_len` | 100.0% |

## Section 4 — Organism & Gene Diversity

- **Unique Genes Represented**: 1214
- **Organism Breakdown**:
  - *human*: 3282 pairs (97.10%)
  - *sars-cov-2*: 95 pairs (2.81%)
  - *homo sapiens*: 2 pairs (0.06%)
  - *mus musculus*: 1 pairs (0.03%)

*Top 20 genes plot saved to:* `plots/gene_diversity.png`.

## Section 5 — Dataset Readiness Checklist

- [x] **PASS**: Minimum 800 total pairs (for Bioinformatics submission minimum)
- [x] **PASS**: At least 200 label=0 entries (sufficient negatives)
- [x] **PASS**: label=0 entries ≤ 40% of total (not over-represented)
- [x] **PASS**: All biophysical features computed for ≥ 90% of rows
- [x] **PASS**: No exact duplicate sequences
- [x] **PASS**: At least 3 different organisms represented
- [x] **PASS**: All rows have paper_doi or source_db populated

## Section 6 — CD-HIT Clustering Preview (80% Threshold)

- **Total Clusters**: 3038
- **Average Cluster Size**: 1.11 entries
- **Largest Cluster Size**: 4 entries
- **Singletons Count**: 2715 clusters

*No homology leakage risk zones found (all clusters contain <= 15 entries).*

---
## Final Verdict
**DATASET READY FOR TRAINING**
