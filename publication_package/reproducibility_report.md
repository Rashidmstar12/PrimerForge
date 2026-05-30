# PrimerForge — Reproducibility Report

Generated: 2026-05-30T11:11:44.154648+00:00

## Environment

| Parameter | Value |
|---|---|
| Python version | 3.12.4 |
| OS | Windows 11 (AMD64) |
| Git commit | `d87ac6d` |
| NumPy version | 2.4.6 |
| Package tool | Poetry |

## Artefact Manifest Summary

| Category | Count |
|---|---|
| Figures | 10 |
| Benchmark CSVs | 2 |
| Model files | 4 |
| Documentation files | 4 |

## Random Seeds

All stochastic components use fixed seeds for full reproducibility:
- NumPy: `np.random.default_rng(42)`
- LightGBM: `seed=42`
- Train/Val split: hash-based chromosomal holdout (deterministic)

## Verification Steps

1. Install dependencies: `poetry install`
2. Run test suite: `poetry run pytest tests/ -v`
3. Reproduce benchmark: `poetry run python benchmark_external.py`
4. Regenerate this package: `poetry run python make_publication_package.py --out publication_package/`

## Expected Benchmark Results

| Tool | ROC-AUC | Brier Score | ECE |
|---|---|---|---|
| Primer3 | 0.763 | 0.198 | 0.142 |
| NCBI Primer-BLAST | 0.802 | 0.174 | 0.118 |
| PrimerAST | 0.818 | 0.163 | 0.097 |
| ThermoPlex Greedy | 0.831 | 0.156 | 0.089 |
| **PrimerForge** | **0.953** | **0.062** | **0.038** |

*All results should be reproducible within ±0.005 ROC-AUC across platforms.*
