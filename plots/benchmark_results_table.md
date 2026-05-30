# PrimerForge External Validation Benchmark Results

Generated over N=1,000 designed primer pairs on completely held-out external validation sequences.

## 1. Classification Metrics

| Method | ROC AUC | PR AUC | Sensitivity | Specificity | F1-Score | Brier Score | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PrimerForge | 0.9288 | 0.9924 | 0.9522 | 0.0297 | 0.9239 | 0.0465 | 0.2131 |
| Primer3 | 0.5000 | 0.9495 | 1.0000 | 0.0000 | 0.9468 | 0.0491 | 0.0910 |
| NCBI Primer-BLAST | 0.4911 | 0.9462 | 1.0000 | 0.0000 | 0.9468 | 0.0459 | 0.0936 |
| PrimerAST (2026) | 0.5051 | 0.9267 | 1.0000 | 0.0000 | 0.9468 | 0.0475 | 0.0838 |
| ThermoPlex | 0.7792 | 0.9564 | 1.0000 | 0.4950 | 0.9724 | 0.0635 | 0.2496 |

## 2. Uncertainty Calibration

- **PrimerForge nominal 95% interval coverage:** 89.30%
- **PrimerForge average interval width:** 0.5452
