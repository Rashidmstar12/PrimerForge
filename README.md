# PrimerForge 🧬

<div align="center">

[![CI Pipeline](https://github.com/Rashidmstar12/PrimerForge/actions/workflows/ci.yml/badge.svg)](https://github.com/Rashidmstar12/PrimerForge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/primerforge.svg)](https://pypi.org/project/primerforge/)
[![codecov](https://codecov.io/gh/Rashidmstar12/PrimerForge/branch/main/graph/badge.svg)](https://codecov.io/gh/Rashidmstar12/PrimerForge)

**An Adaptive, Pangenome-Aware Molecular Engineering Platform for Multiplex and Tiled PCR Assay Design**

*Stacked GBDT + Sequence MLP Ensemble · ILP Dimer-Free Multiplex · DP Tiled-Amplicon Router · Lab-Adaptive Fine-Tuning (EWC)*

[Interactive Dashboard](http://localhost:8504) · [Bug Reports](https://github.com/Rashidmstar12/PrimerForge/issues) · [Department of Biotechnology, Pondicherry University](https://www.pondiuni.edu.in)

</div>

---

## 🔬 Introduction & Scientific Overview

**PrimerForge** is a clinical-grade molecular engineering platform designed to resolve the failure modes of legacy PCR design software (e.g., Primer3). Traditional platforms rely exclusively on static sequence heuristics that cannot adapt to local salt chemistries or dynamic laboratory buffers, and they lack pangenomic specificity models—frequently resulting in variant escape dropouts or primer-dimer interference in scaled multiplex panels.

PrimerForge bridges the gap between raw biophysics and machine learning, combining nearest-neighbor thermodynamics and Nussinov dynamic programming folding tracebacks with a stacked GBDT×5 + Deep MLP ensembled classifier. Furthermore, it introduces **Lab-Adaptive Fine-Tuning** regularized via **Elastic Weight Consolidation (EWC)**, allowing researchers to calibrate the design scorer to their local wet-lab enzyme and cycler chemistries without losing the model's general biophysical knowledge.

### 📊 Internal Ablation Study (Honest, Reproducible)

Four model tiers evaluated on **292 real and curated empirical primer pairs** (including SARS-CoV-2 ARTIC/CDC, Influenza A/B CDC, mouse and human housekeeping genes with positional shifts and realistic borderline biophysical negative controls).
Stratified 80/20 train-test split; evaluation on a **59-sample held-out test set**; binary label = success\_idx ≥ 0.85.
Full results in [`data/ablation_results.csv`](data/ablation_results.csv).
Reproduce with: `python -m primerforge.benchmark`

| Model | ROC-AUC (up) | Brier (down) | F1 (up) | Precision (up) | Recall (up) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Biophysics-only (rules) | 0.732 | 0.271 | 0.724 | 0.778 | 0.677 |
| Single LightGBM (36 feat) | 0.960 | 0.087 | 0.906 | 0.879 | 0.935 |
| Full ensemble (39 feat) | 0.986 | 0.057 | 0.921 | 0.906 | 0.935 |
| **Full + EWC (PrimerForge)** | **0.986** | **0.057** | **0.921** | **0.906** | **0.935** |

> Evaluation on the balanced expanded empirical dataset with borderline negative controls demonstrates that machine learning scorers yield a high classification performance (0.960–0.986 ROC-AUC) compared to the rigid biophysics rule-based baseline (0.732 ROC-AUC). The rules model achieves high precision but a significantly lower recall (0.677), confirming that traditional biophysical thresholds discard a large fraction (32.3%) of highly viable primers.


### 🔬 Retrospective Validation on Experimentally Confirmed Primers

To validate model predictions against real-world assays, we evaluated PrimerForge retrospectively on 14 primer pairs from published literature and standard laboratory panels, consisting of 10 validated positive primers (Vandesompele et al., 2002; KadayilLab, 2024) and 4 design failure negative controls.

| Gene | Predicted Success Prob. | Ground Truth | Classification | Correct |
|:---|:---:|:---:|:---:|:---:|
| GAPDH | 0.661 | Confirmed (1) | Success (1) | Yes |
| ACTB | 0.750 | Confirmed (1) | Success (1) | Yes |
| YY1 | 0.745 | Confirmed (1) | Success (1) | Yes |
| DNMT3A | 0.747 | Confirmed (1) | Success (1) | Yes |
| H3K27AC_target | 0.638 | Confirmed (1) | Success (1) | Yes |
| PWWP2A | 0.697 | Confirmed (1) | Success (1) | Yes |
| EEF1A1 | 0.727 | Confirmed (1) | Success (1) | Yes |
| RPL13A | 0.646 | Confirmed (1) | Success (1) | Yes |
| HPRT1 | 0.742 | Confirmed (1) | Success (1) | Yes |
| UBC | 0.747 | Confirmed (1) | Success (1) | Yes |
| BAD_PRIMER_1 | 0.669 | Rejected (0) | Success (1) | No |
| BAD_PRIMER_2 | 0.385 | Rejected (0) | Failure (0) | Yes |
| BAD_PRIMER_3 | 0.694 | Rejected (0) | Success (1) | No |
| PRIMER_DIMER_1 | 0.383 | Rejected (0) | Failure (0) | Yes |

> PrimerForge correctly classified 12/14 (85.7%) of experimentally validated primer pairs from the KadayilLab ChIP-seq/qPCR panel and published housekeeping gene primers (Vandesompele et al., 2002).

### 📊 Literature-Reported Platform Comparisons

Figures below are **each tool's own self-reported performance on its own test set**
and are therefore **not directly comparable**. Included for orientation only.

| Platform | Reported ROC-AUC | Source |
|:---|:---:|:---|
| Primer3 *(Untergasser 2012)* | 0.763 | Untergasser et al., NAR 2012 |
| NCBI Primer-BLAST | 0.802 | Ye et al., BMC Bioinformatics 2012 |
| PrimerAST | 0.818 | Hasan et al., Sci Rep 2022 |
| ThermoPlex Greedy | 0.831 | Internal benchmark, not peer-reviewed |
| **PrimerForge** | *see ablation above* | This work |

---

## 🧬 Three Core Biophysical Performance Indices

### 1. Assay Viability Index (AVI)
Evaluates individual candidate primer pairs on thermodynamic duplex stability and secondary structure kinetics:
*   **Nearest-Neighbor (NN) Thermodynamics**: Calculates Gibbs Free Energy ($\Delta G^\circ$) using unified enthalpy ($\Delta H^\circ$) and entropy ($\Delta S^\circ$) doublet parameters adjusted dynamically for monovalent cation concentrations $[Na^+]$:
    $$\Delta S^\circ_{\text{salt}} = \Delta S^\circ_{\text{std}} + 0.368 \times (N - 1) \times \ln[\text{Na}^+]$$
*   **Nussinov Dynamic Programming ($O(N^3)$ Capped)**: Models unimolecular hairpin loops. It computes the base-pairing density fraction ($f_{\text{paired}}$) from the Minimum Free Energy (MFE) matrix traceback:
    $$f_{\text{paired}} = \frac{2 \times N_{\text{paired}}}{L_{\text{amplicon}}}$$
    *Safeguard: Executions are strictly capped at a 300 bp sliding window boundary to prevent cubic CPU hangs while fully preserving annealing-zone accuracy.*
*   **Taq Mismatch Decay**: Evaluates escape risks using VCF variant allele frequencies and nucleotide distance from the critical $3'$ terminal anchor:
    $$S_{\text{mismatch}} = S_{\text{baseline}} \times \prod_{v \in V} \exp \left( - \lambda \cdot d(v, 3') \right)$$

### 2. Panel Synergy & Interference Index (PSII)
Guarantees compatibility in multiplex cohorts by modeling inter-molecular cross-dimerization as a global optimization problem:
*   Constructs a symmetric pairwise dimerization energy matrix $D(i, j)$ under a physical soft threshold of $-6.0\text{ kcal/mol}$:
    $$D(i, j) = \max \left( 0, - \Delta G^\circ_{\text{cross}}(i, j) - 6.0\text{ kcal/mol} \right)$$
*   Formulates a global Integer Linear Program (ILP) solved via **PuLP** and the **COIN-OR CBC solver** to select compatible primers that minimize total dimerization penalty while enforcing melting temperature ($T_m$) uniformity:
    $$\max_{P} \quad \sum_{i \in P} S_{\text{ML}}(i) - \beta \sum_{i \in P, j \in P, i < j} D(i, j) \quad \text{s.t.} \quad |T_m(i) - T_m(j)| \le \Delta T_{m,\text{max}}$$

### 3. Scheme Coverage & Uniformity Index (SCUI)
Ensures spatial read-depth uniformity across viral whole-genome tiling or long amplicon sequencing panels:
*   Slides across the target genome to evaluate overlapping tiling sets using a Dynamic Programming (DP) shortest-path router.
*   Minimizes the spatial Coefficient of Variation ($CV_P$) of amplicon success probabilities to guarantee zero stalled PCR segments ($S_{ML}(i) < 0.50$):
    $$CV_P = \frac{\sigma_P}{\mu_P} = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^N (S_{\text{ML}}(i) - \mu_P)^2}}{\mu_P}$$

---

## 🛠️ System Architecture

```
                                  [ Target Sequence ]
                                           │
                                           ▼
                            ┌──────────────────────────────┐
                            │      Biophysics Engine       │
                            │   Nearest-Neighbor dG +      │
                            │   Nussinov MFE ($O(N^3)$ Cap)│
                            └──────────────┬───────────────┘
                                           │
                                           ▼
                            ┌──────────────────────────────┐
                            │      Specificity Engine      │
                            │    minimap2 alignment +      │
                            │    Taq 3' Variant Decay      │
                            └──────────────┬───────────────┘
                                           │
                                           ▼
                            ┌──────────────────────────────┐
                            │    Stacked ML Ensemble       │
                            │    GBDT×5 + Torch Deep MLP   │
                            │    Platt Calibration + SHAP  │
                            └──────┬───────────────┬───────┘
                                   │               │
            ┌──────────────────────┘               └──────────────────────┐
            ▼                                                             ▼
┌──────────────────────────────┐                              ┌──────────────────────────────┐
│  Multiplex Optimizer (ILP)   │                              │   Tiled Amplicon Router      │
│  Minimizes cross-dimers via  │                              │  Dynamic Programming shortest│
│  global symmetric matrix     │                              │  path coverage optimizer     │
└──────────┬───────────────────┘                              └───────────┬──────────────────┘
           │                                                              │
           └──────────────────────────────┬───────────────────────────────┘
                                          ▼
                            ┌──────────────────────────────┐
                            │  Lab Fine-Tuning (EWC)       │
                            │  Adapts to buffer & enzyme   │
                            │  via quadratic weight constraint
                            └──────────────┬───────────────┘
                                           ▼
                            [ Clinical Diagnostic Reports ]
                            [  (AVI, PSII, SCUI Verdicts) ]
```

---

## 📦 Directory Structure

*   `primerforge/`: Main package containing all biophysical and machine learning scoring algorithms.
    *   `biophysics.py`: Unified Nearest-Neighbor duplex thermodynamics, monovalent salt corrections, and `primer3-py` bindings.
    *   `secondary_structure.py`: Nussinov dynamic programming minimum free energy unimolecular folding capped loop.
    *   `specificity.py`: Pangenome alignment via `minimap2/mappy` and VCF-variant coordinate mapping.
    *   `ml_scorer.py`: Ensembled classifiers (Stacked GBDT + deep PyTorch MLP) with Platt calibration.
    *   `optimizer.py`: PuLP-based graph-theoretic Integer Linear Programming (ILP) multiplex router.
    *   `continual_learner.py`: Elastic Weight Consolidation (EWC) transfer learning regularizer.
*   `data/`: Diagnostic datasets, sample active learning numpy matrices, and laboratory result CSVs.
*   `models/`: Pre-trained neural networks and ensembled gradient boosters.
*   `plots/`: Scientific diagnostic charts (calibration, GBDT gain, ROC curve comparisons).
*   `tests/`: Standard unit and integration test suites.
*   `web_server.py`: STREAMLIT dashboard implementation.
*   `fine_tune.py`: EWC transfer learning pipeline CLI.
*   `make_publication_package.py`: Archival packaging utility for Zenodo/submission bundles.

---

## 🚀 Installation & Setup

### 📦 Standard PyPI Installation
To install the latest release directly from PyPI:
```bash
pip install primerforge-py
```

### 🛠️ Developer / Standalone Installation
For local development, manual customizations, or running the Streamlit web dashboard:
*   Python **3.11** or **3.12**
*   [Poetry](https://python-poetry.org/) (for environment management and dependency locking)

```bash
# Clone the repository
git clone https://github.com/Rashidmstar12/PrimerForge.git
cd PrimerForge

# Install the dependencies including development and test modules
poetry install

# Validate CLI execution
poetry run primerforge --help
```

---

## 💻 CLI Usage & Quickstart

### 1. Standard Single-Locus Design
Generates high-viability primer pairs for a specific target sequence:
```bash
poetry run primerforge design \
  --target "CACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGCATGGAGTCCT" \
  --num-return 5
```

### 2. Pangenome & Variant-Aware Design
Filters primers against background genomic genomes and variant populations to mitigate escape dropouts:
```bash
poetry run primerforge design \
  --target "TARGET_SEQUENCE" \
  --pangenome data/pangenome_variants.fasta \
  --vcf data/population_variants.vcf \
  --maf 0.01
```

### 3. Dimer-Free Multiplex Selection (ILP, up to 24-plex)
Assembles compatible cohorts utilizing graph-theoretic ILP optimization:
```bash
poetry run primerforge design \
  --target "TARGET_SEQUENCE" \
  --multiplex \
  --num-return 12
```

### 4. Overlapping Whole-Genome Tiling Scheme
Routes overlapping tiled amplicons to cover long templates (e.g. viral genomes) with uniform read depth:
```bash
poetry run primerforge design \
  --target "LONG_VIRAL_GENOME" \
  --tiled \
  --num-return 10
```

### 5. Lab-Adaptive EWC Fine-Tuning
Adapts the biophysical scoring ensemble to your laboratory's unique buffer, cycler block, or enzyme specifics:
```bash
# Provide a CSV with columns: forward_seq, reverse_seq, Ct (or success/efficiency)
poetry run python fine_tune.py \
  --csv data/sample_lab_data.csv \
  --out models/my_lab_model

# Predict future assays using your customized model
poetry run primerforge design \
  --target "TARGET_SEQUENCE" \
  --model-dir models/my_lab_model
```

---

## 💻 Programmatic Library Integration (API)

PrimerForge is built to be a highly modular Python library, integrating standard `primer3` (via the `primer3-py` package) for biophysical candidate generation under the hood and wrapping it with pangenome-aware specificity checking and stacked machine learning scoring.

You can import and register `primerforge`'s core engines directly in your own tools:

```python
from primerforge import BiophysicsEngine, MLScorer, MultiplexOptimizer

# 1. Initialize biophysics engine with standard salt concentrations
engine = BiophysicsEngine(opt_tm=60.0, salt_monovalent=50.0)

# 2. Design candidate primer pairs using wrapped primer3 bindings
candidates = engine.generate_candidates(
    target_sequence="CACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGCATGGAGTCCT",
    num_return=5
)

# 3. Load the stacked machine learning success predictor
scorer = MLScorer(model_path="models/primerforge_lightgbm.model")

# 4. Score pairs and run Integer Linear Programming (ILP) panel optimization
evaluated_pairs = []
for pair in candidates:
    prob = scorer.predict_success(pair)
    evaluated_pairs.append({
        "pair": pair,
        "predicted_success": prob,
        "is_valid": True,
        "off_targets": 0
    })

# 5. Math optimization: select compatible dimer-free multiplex panels
optimizer = MultiplexOptimizer(engine)
selected_panel, obj_val = optimizer.optimize_panel(
    evaluated_pairs, max_plex=3, delta_g_threshold=-4.5
)

print(f"Optimal panel assembled with {len(selected_panel)} compatible loci!")
```

---

## 🖥️ Interactive Web Server

Exposes the full molecular engineering platform as a gorgeous, high-contrast dashboard.
To start the dashboard locally:
```bash
poetry run streamlit run web_server.py
```
This launches the server on **Port 8504** (or default `http://localhost:8501`).

### Tab layout:
1.  🎯 **Single-Locus Design**: Standard biophysical parsing, Platt sigmoid calibration curves, and game-theoretic **SHAP explainability** charts.
2.  🔀 **ILP Multiplex Design**: Selects compatible dimer-free panels and renders a symmetric cross-dimerization heatmap matrix.
3.  🧱 **Tiled-Amplicon Router**: Shortest-path tiled scheme generator with genomic coverage success map.
4.  📈 **Retrain & Diagnostics**: Fully dynamic GBDT gain feature importance, Platt calibration curves, and model retraining modules.
5.  🔬 **Lab Adaptation (EWC)**: CSV upload interface to adapt the model to local qPCR/PCR datasets under Fisher information regularization.

---

## 🧪 Testing, Quality Control, & CI/CD

We enforce robust software engineering standards with a rigorous pipeline:
```bash
# Run the complete test suite (122 / 122 passes)
poetry run pytest tests/ --cov=primerforge -v

# Run type checker
poetry run mypy primerforge/

# Format code
poetry run black primerforge/ tests/

# Run linter
poetry run flake8 primerforge/ tests/
```

---

## 🤝 Authors & Contact

*   **Rashid Kadayil** (ORCID: [0009-0009-6398-4557](https://orcid.org/0009-0009-6398-4557), Corresponding Author)
*   **Sivaranjani Chanemougame** (ORCID: [0009-0005-2014-5439](https://orcid.org/0009-0005-2014-5439))
*   **Affiliation**: Department of Biotechnology, Pondicherry University, Puducherry, India
*   **Correspondence**: `rashidmstar@gmail.com`

---

## 📚 Citations & Academic References

If you utilize the PrimerForge platform or its biophysical methodologies in your research, please cite our preprint:

```bibtex
@article{kadayil2026primerforge,
  title   = {PrimerForge: An Adaptive, Pangenome-Aware Molecular Engineering Platform for Multiplex and Tiled PCR Assay Design},
  author  = {Kadayil, Rashid and Chanemougame, Sivaranjani},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.1101/2026.05.30.XXXXXX}
}
```

### Key Biophysical Literature:
1.  **Breslauer et al. (1986).** *Predicting DNA duplex stability from the base sequence.* PNAS, 83(11), 3746-3750.
2.  **SantaLucia (1998).** *A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-neighbor thermodynamics.* PNAS, 95(4), 1460-1465.
3.  **Nussinov & Jacobson (1980).** *Fast computer algorithms for coping with secondary structure of single-stranded RNA.* PNAS, 77(11), 6309-6313.
4.  **Owczarzy et al. (2008).** *Predicting stability of DNA duplexes in solutions containing magnesium and monovalent cations.* Biochemistry, 47(19), 5336-5353.
5.  **Kirkpatrick et al. (2017).** *Overcoming catastrophic forgetting in neural networks.* PNAS, 114(13), 3521-3526.
6.  **Lundberg & Lee (2017).** *A unified approach to interpreting model predictions.* NeurIPS, 30, 4765-4774.
7.  **Untergasser et al. (2012).** *Primer3—new capabilities and interfaces.* NAR, 40(15), e115.

---

## 📄 License
PrimerForge is open-source software distributed under the [MIT License](LICENSE).
