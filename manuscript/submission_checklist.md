# bioRxiv Submission Readiness Checklist

This checklist tracks the implementation, validation, and documentation progress required before submitting the PrimerForge manuscript to bioRxiv.

## ✅ Completed Tasks (DONE)

- [x] **Tool implemented and tested**: Pure-Python and wrapper-based modules for thermodynamics, GNN representation, specificity aligning, and multiplex/tiled routing are fully operational.
- [x] **CI green with test suite**: GitHub Actions workflow passes all 118 unit tests across Python 3.11 and 3.12 environments.
- [x] **Real empirical dataset (292 pairs)**: Curated positive pairs from PrimerBank and clinical panels merged with realistic, borderline thermodynamic and sequence negative controls.
- [x] **Honest ablation study (80/20 split, n=59 test)**: Evaluated 4 tiers (Rules, LightGBM, Full Ensemble, Full + EWC) showing a realistic ROC-AUC improvement gradient (0.732 to 0.986) on the held-out test split.
- [x] **Methods section drafted**: Scientific computational methodology section outlining the entire system architecture is complete.
- [x] **Abstract drafted**: 199-word journal-ready abstract summarizing background, methods, results, and keywords is ready.
- [x] **PyPI package (`pip install primerforge-py`)**: Rebuilt packages and verified installation and runtime imports.
- [x] **GitHub repo public with MIT license**: Hosted publicly at [https://github.com/Rashidmstar12/PrimerForge](https://github.com/Rashidmstar12/PrimerForge) with appropriate open-source licensing.

## 📋 Remaining Pre-Submission Tasks (STILL NEEDED)

- [ ] **Wet-lab validation**: Design 5-10 real primers with PrimerForge, run PCR amplification assays, and report experimental Ct values/melts to validate in vitro performance.
- [ ] **PrimalScheme comparison for tiling mode**: Run head-to-head tiled amplicon design comparisons against PrimalScheme on viral targets (e.g. SARS-CoV-2, influenza) comparing panel size and overlap constraints.
- [ ] **EWC fine-tuning ablation**: Run transfer learning experiments on specific lab buffers (with vs. without EWC weight regularization) to quantify mitigation of catastrophic forgetting.
- [ ] **Increase dataset to >= 500 pairs**: Fetch additional validated sequences from RTPrimerDB and other empirical databases to increase ML model generalizability.
- [ ] **Generate actual publication figures**: Code matplotlib/seaborn visualization scripts to render high-resolution raster files for Figure 2 (ablation bar chart), Figure 3 (Platt calibration curve), and Figure 4 (SHAP plots).
- [ ] **Supplementary data file preparation**: Package the expanded 292-pair dataset CSV and feature matrix as supplementary tables.
