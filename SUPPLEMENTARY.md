# PrimerForge — Supplementary Methods

> **Associated Manuscript**:  
> *PrimerForge: A Hybrid Thermodynamic and Machine Learning Platform for  
> Pangenome-Aware PCR Primer Design*  
> Submitted to *Nucleic Acids Research* (Web Server Issue) / *Bioinformatics* (Applications Note)

---

## Table of Contents

1. [Thermodynamic Nearest-Neighbour Engine](#s1)
2. [Pangenome Specificity & Variant-Aware Filtering](#s2)
3. [Empirical Database Construction (PrimerForge-Empirical-DB)](#s3)
4. [Stacked Ensemble ML Architecture](#s4)
5. [Platt Calibration & Uncertainty Quantification](#s5)
6. [ILP Multiplex Optimizer](#s6)
7. [Dynamic Programming Tiled-Amplicon Router](#s7)
8. [Lab-Adaptive Fine-Tuning Module](#s8)
9. [External Validation & Benchmarking Protocol](#s9)
10. [Reproducibility Checklist](#s10)
11. [Software & Dependency Versions](#s11)

---

<a name="s1"></a>
## S1 — Thermodynamic Nearest-Neighbour Engine

### S1.1 Melting Temperature

Melting temperature $T_m$ is calculated using the **SantaLucia (1998)** unified nearest-neighbour parameters:

$$T_m = \frac{\Delta H^\circ}{\Delta S^\circ + R \ln(C_T/4)} - 273.15 \quad (°\text{C})$$

where $\Delta H^\circ$ (kcal mol⁻¹) and $\Delta S^\circ$ (cal mol⁻¹ K⁻¹) are summed over all dinucleotide stacks plus initiation terms, $R = 1.987\ \text{cal mol}^{-1}\text{K}^{-1}$, and $C_T = 250\ \text{nM}$ (total strand concentration, typical qPCR).  
Salt correction uses the **Owczarzy (2004)** empirical formula for monovalent cations at $[\text{Na}^+] = 50\ \text{mM}$.

### S1.2 Secondary Structure Free Energies

Hairpin and self/cross-dimer formation energies are computed via `primer3-py` (wrapping the C library `primer3`):

$$\Delta G_{37} = \Delta H - 310.15 \cdot \Delta S$$

Candidates with $\Delta G_{\text{hairpin}} < -2.0\ \text{kcal mol}^{-1}$ or $\Delta G_{\text{self-dimer}} < -6.0\ \text{kcal mol}^{-1}$ are flagged.  Cross-dimer threshold for multiplex rejection: $\Delta G_{\text{cross}} < \theta_{\Delta G}$ (user-configurable, default $-4.5\ \text{kcal mol}^{-1}$).

### S1.3 Candidate Generation Algorithm

```
Algorithm 1: BiophysicsEngine.generate_candidates
─────────────────────────────────────────────────
Input:  target sequence T (length n), opt_tm, size range [l_min, l_max]
Output: ranked list of PrimerPair objects

1. For each forward start position s_f ∈ [0, n − l_min]:
   a. For each primer length l ∈ [l_min, l_max]:
      i.  Extract forward candidate f = T[s_f : s_f+l]
      ii. Compute Tm(f), ΔG_hairpin(f), GC(f)
      iii. Skip if |Tm(f) − opt_tm| > 8°C or GC ∉ [0.35, 0.70]
2. Generate reverse complement candidates analogously from 3′ end
3. For each valid (f, r) pair:
   a. Compute product size P = r_start − f_start + l_r
   b. Skip if P ∉ [50, 1500] bp
   c. Compute ΔG_cross(f, r)
   d. Score pair: base_penalty = −ΔG_cross
4. Return top-k pairs sorted by ascending base_penalty
```

---

<a name="s2"></a>
## S2 — Pangenome Specificity & Variant-Aware Filtering

### S2.1 Multi-genome Alignment

Primer candidates are aligned to the reference pangenome using `mappy` (Python bindings to `minimap2`, Li 2018) with preset `sr` (short-read) and secondary alignment reporting enabled:

```python
aligner = mappy.Aligner(pangenome_path, preset="sr", best_n=10)
hits = list(aligner.map(primer_seq))
off_target_count = sum(1 for h in hits if h.mapq >= 20 and not is_on_target(h))
```

Primers with `off_target_count > 0` receive a 50 % penalty in the base score; those with `off_target_count > 2` are discarded.

### S2.2 Variant-Aware 3′ Anchor Filtering

VCF records are parsed with `cyvcf2` (O'Leary et al. 2017). The 3′ anchor region is defined as the last 5 bp of each primer. Any variant with:
- Genomic position overlapping the anchor AND  
- Minor Allele Frequency (MAF) ≥ threshold (default 0.01)

causes rejection (hard filter).  Variants outside the anchor but within the primer body (≥ MAF) add a weighted penalty:

$$\text{penalty}_{\text{variant}} = \text{MAF} \times \max(0,\ 5 - d_{\text{3′}})$$

where $d_{\text{3′}}$ is the distance from the variant to the 3′ end in bp.

---

<a name="s3"></a>
## S3 — Empirical Database Construction (PrimerForge-Empirical-DB)

### S3.1 Sources

| Source | Pairs (approx.) | Label type |
|---|---|---|
| PubMed PMC open-access full-text XML (primer table extraction) | ~180 000 | Binary success/failure |
| Patent sequence databases (US/EP/WO) | ~95 000 | Binary |
| Internal synthetic simulation (SantaLucia thermodynamic + Monte Carlo amplification model) | ~500 000 | Continuous $P_{\text{success}}$ |
| User fine-tune (lab-specific, opt-in) | Variable | qPCR Ct / gel band |

### S3.2 Positive vs. Negative Construction

**Positives** ($y = 1$): Published primer pairs with reported successful PCR amplification (gel validation or qPCR Ct ≤ 35).

**Hard Negatives** ($y = 0$): Synthetically perturbed versions of positives with controlled failure modes:
- 3′ terminal G/C → A/T substitution ($\Delta G_{\text{3′}}$ destabilized)
- Homopolymer runs of length ≥ 6 introduced
- Cross-dimer partner introduced ($\Delta G_{\text{cross}} < -6\ \text{kcal mol}^{-1}$)
- SNP introduced at position −2 from 3′ with MAF = 0.05

### S3.3 Train / Val / Test Split (Chromosomal Holdout)

To prevent sequence leakage:
- **Training** (80 %): Human chromosomes 1–18 + pathogen loci
- **Validation** (10 %): Human chromosomes 19–22
- **Held-out test** (10 %): Human chromosomes X, Y + independent viral genomes (SARS-CoV-2, Influenza A, HIV-1)

---

<a name="s4"></a>
## S4 — Stacked Ensemble ML Architecture

### S4.1 Feature Vector (36 dimensions)

| # | Feature | Description |
|---|---|---|
| 1 | `tm_f` | Forward primer $T_m$ (°C) |
| 2 | `tm_r` | Reverse primer $T_m$ (°C) |
| 3 | `tm_diff` | $|T_{m,f} - T_{m,r}|$ |
| 4 | `gc_f` | Forward GC content |
| 5 | `gc_r` | Reverse GC content |
| 6 | `hairpin_dg_f` | Forward hairpin ΔG |
| 7 | `hairpin_dg_r` | Reverse hairpin ΔG |
| 8 | `self_dimer_dg_f` | Forward self-dimer ΔG |
| 9 | `self_dimer_dg_r` | Reverse self-dimer ΔG |
| 10 | `cross_dimer_dg` | Cross-dimer ΔG (main discriminator) |
| 11 | `product_size` | Amplicon length (bp) |
| 12 | `f_homopolymer` | Max homopolymer run in forward |
| 13 | `r_homopolymer` | Max homopolymer run in reverse |
| 14 | `f_3prime_gc` | 3′ GC clamp (last 5 bp) |
| 15 | `r_3prime_gc` | 3′ GC clamp (last 5 bp) |
| 16 | `f_off_targets` | Off-target alignment count |
| 17 | `r_off_targets` | Off-target alignment count |
| 18 | `f_var_dist` | Distance to nearest 3′ variant (bp) |
| 19 | `r_var_dist` | Distance to nearest 3′ variant (bp) |
| 20 | `f_var_maf` | MAF of nearest variant |
| 21 | `r_var_maf` | MAF of nearest variant |
| 22–27 | `f_dinuc_[AC/AG/AT/CG/CT/GT]` | Forward dinucleotide frequency |
| 28–33 | `r_dinuc_[AC/AG/AT/CG/CT/GT]` | Reverse dinucleotide frequency |
| 34 | `mlp_embed_f` | Forward sequence MLP embedding scalar |
| 35 | `mlp_embed_r` | Reverse sequence MLP embedding scalar |
| 36 | `mlp_embed_cross` | Cross-sequence MLP interaction term |

### S4.2 NumPy MLP Sequence Embedder

A two-layer fully connected network implemented in pure NumPy for zero-dependency portability:

$$h = \text{ReLU}(W_1 x + b_1), \quad \hat{y} = W_2 h + b_2$$

where $x \in \mathbb{R}^{24}$ is a one-hot + k-mer composition encoding of the primer sequence, $W_1 \in \mathbb{R}^{64 \times 24}$, $W_2 \in \mathbb{R}^{1 \times 64}$.

Training uses mini-batch SGD with momentum (lr = 0.01, momentum = 0.9, 300 epochs, batch size 128).

### S4.3 GBDT Ensemble

Five LightGBM boosters are trained with different random seeds on the same feature matrix:

```
n_estimators = 400, learning_rate = 0.05
max_depth = 7, num_leaves = 63
min_child_samples = 20
reg_alpha = 0.1, reg_lambda = 0.1
subsample = 0.8, colsample_bytree = 0.8
```

Final ensemble prediction: $\hat{p} = \frac{1}{5}\sum_{k=1}^{5} \sigma(f_k(x))$ where $\sigma$ is the Platt sigmoid.

---

<a name="s5"></a>
## S5 — Platt Calibration & Uncertainty Quantification

### S5.1 Platt Scaling

Platt calibration maps raw GBDT outputs $s$ to calibrated probabilities:

$$P(\text{success}) = \sigma(A \cdot s + B) = \frac{1}{1 + e^{-(As + B)}}$$

Parameters $A$ and $B$ are fitted by gradient ascent on held-out validation log-likelihood:

$$\ell(A, B) = \sum_i \left[ y_i \ln \hat{p}_i + (1-y_i) \ln(1-\hat{p}_i) \right]$$

$$A \leftarrow A + \eta \frac{\partial \ell}{\partial A}, \quad B \leftarrow B + \eta \frac{\partial \ell}{\partial B}$$

### S5.2 Uncertainty Quantification

**Epistemic uncertainty** (model uncertainty): standard deviation across the 5 GBDT ensemble members' predictions.

**95 % Confidence Interval**:
$$\text{CI}_{95} = [\hat{p} - 1.96\,\sigma_{\text{epistemic}},\ \hat{p} + 1.96\,\sigma_{\text{epistemic}}]$$

**Aleatoric uncertainty** (data noise): estimated via GBDT quantile regression at $\alpha \in \{0.025, 0.975\}$ using `objective = "quantile"`.

---

<a name="s6"></a>
## S6 — ILP Multiplex Optimizer

### S6.1 Problem Formulation

Given a set of $N$ candidate primer pairs $\mathcal{P} = \{p_1, \dots, p_N\}$ with predicted success scores $w_i = P_{\text{success}}(p_i)$, the multiplex panel assembly is cast as a **Maximum-Weight Independent Set** (MWIS) problem on a conflict graph $G = (\mathcal{P}, E)$:

$$E = \{(i, j) : \Delta G_{\text{cross}}(p_i, p_j) < \theta_{\Delta G}\}$$

**ILP formulation**:

$$\max_{x \in \{0,1\}^N} \sum_{i=1}^{N} w_i x_i$$

subject to:

$$x_i + x_j \leq 1 \quad \forall\, (i,j) \in E \quad \text{(dimer conflict)}$$
$$\sum_{i=1}^{N} x_i \leq M \quad \text{(max plex constraint)}$$

Solved via PuLP (CBC solver) in polynomial time for $N \leq 500$, $M \leq 24$.

### S6.2 Panel Coverage Constraint

For multi-locus panels, an additional constraint ensures at least one representative primer per requested locus:

$$\sum_{i \in \mathcal{P}_k} x_i \geq 1 \quad \forall\, k \in \{1, \dots, K\}$$

where $\mathcal{P}_k$ is the set of candidate pairs for locus $k$.

---

<a name="s7"></a>
## S7 — Dynamic Programming Tiled-Amplicon Router

### S7.1 Tile Graph Construction

Given a reference sequence $G$ of length $L$, candidate tiles are generated at positions $\{0, s, 2s, \ldots\}$ with step $s = \text{tile\_size} - \text{overlap}$.  Each tile $i$ starting at position $a_i$ with size $T$ generates primer pair $p_i$ via `BiophysicsEngine`.

### S7.2 DP Recurrence

Let $f(i)$ = maximum total success score for a tiling that ends at tile $i$:

$$f(i) = \max_{j : a_j + T_j \leq a_i + \delta} \left[ f(j) + P_{\text{success}}(p_i) \right]$$

where $\delta$ is the maximum allowed gap (default 0 bp; overlaps encouraged).  
Base case: $f(0) = P_{\text{success}}(p_0)$.

Optimal tile set recovered by backtracking the parent pointers.

### S7.3 Complexity

Time: $O(K^2)$ where $K = \lceil (L - T) / s \rceil$ is the number of candidate tiles.  
Space: $O(K)$ for DP table and parent array.  
Typical SARS-CoV-2 genome ($L = 30\,000$ bp, $T = 400$, $s = 350$): $K \approx 85$, runtime $< 0.5\ \text{s}$.

---

<a name="s8"></a>
## S8 — Lab-Adaptive Fine-Tuning Module

### S8.1 GBDT Continual Learning

Each LightGBM booster is extended with $M_{\text{new}} = 10$ additional trees at learning rate $\eta = 0.01$ and regularization $\lambda = 5.0$, trained on a rehearsal dataset:

$$\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{user}} \cup \mathcal{D}_{\text{anchor}}$$

where $\mathcal{D}_{\text{anchor}}$ contains $N_{\text{anchor}} = 200$ synthetically generated primer pairs drawn from the original training distribution to prevent catastrophic forgetting.

### S8.2 MLP Elastic Weight Consolidation

The sequence MLP's first layer $(W_1, b_1)$ is frozen.  Only the output head $(W_2, b_2)$ is fine-tuned with an L2 anchor penalty:

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{EWC}} \|W_2 - W_2^{(0)}\|_2^2$$

Gradient update:

$$\Delta W_2 = \frac{\partial \mathcal{L}_{\text{MSE}}}{\partial W_2} + 2\lambda_{\text{EWC}}(W_2 - W_2^{(0)})$$

Default: $\lambda_{\text{EWC}} = 0.1$, 50 fine-tuning epochs, lr = 0.001.

### S8.3 Label Normalization

| Input format | Normalization |
|---|---|
| `success = 1 / "positive"` | $y = 0.95$ |
| `success = 0 / "negative"` | $y = 0.05$ |
| Ct value | $y = 1 - \min(1, \text{Ct}/40)$ |
| Efficiency (0–1) | $y = \text{efficiency}$ |
| Efficiency (0–100 %) | $y = \text{efficiency}/100$ |

---

<a name="s9"></a>
## S9 — External Validation & Benchmarking Protocol

### S9.1 External Test Set

Independent validation was performed on $N = 1\,000$ primer pairs from:
- 200 clinical BRCA1/2 diagnostic assays (Myriad Genetics disclosure data)
- 200 SARS-CoV-2 tiling panel pairs (ARTIC v4 + v4.1)
- 200 metagenomic universal primer pairs (16S/ITS rDNA databases)
- 200 somatic mutation detection pairs (TCGA exome panel)
- 200 hard edge cases (GC-rich > 70 %, repetitive elements, VNTR-adjacent)

### S9.2 Baselines

| Tool | Version | Method |
|---|---|---|
| Primer3 | 2.6.1 | Thermodynamics only |
| NCBI Primer-BLAST | API (2024-04) | Thermodynamics + BLAST specificity |
| PrimerAST | 1.2.0 | Rule-based + mfold |
| ThermoPlex Greedy | Reimplemented | Greedy dimer-avoidance |
| **PrimerForge** | 0.3.0 | This work |

### S9.3 Evaluation Metrics

$$\text{ROC-AUC} = \int_0^1 \text{TPR}(t)\, d\text{FPR}(t)$$

$$\text{Brier Score} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2 \quad \text{(lower = better)}$$

$$\text{ECE} = \sum_{b=1}^{B} \frac{|I_b|}{N} \left| \overline{y}_b - \overline{\hat{p}}_b \right| \quad (B=10\ \text{bins})$$

---

<a name="s10"></a>
## S10 — Reproducibility Checklist

- [ ] All random seeds fixed (`numpy.random.seed(42)`, `lightgbm` `seed=42`)  
- [ ] Data splits are deterministic (hash-based chromosomal assignment)  
- [ ] All intermediate artefacts logged to `models/` with SHA-256 checksums  
- [ ] `poetry.lock` pinned to exact dependency versions  
- [ ] Docker image available: `docker pull primerforge/primerforge:0.3.0`  
- [ ] Zenodo archive DOI: `10.5281/zenodo.XXXXXXX` (populated at submission)  
- [ ] GitHub Actions CI passes on Python 3.11 and 3.12 (Linux + macOS)  
- [ ] All plots generated by `make_publication_package.py` are bit-identical across runs  

---

<a name="s11"></a>
## S11 — Software & Dependency Versions

| Package | Version | Role |
|---|---|---|
| Python | 3.11 / 3.12 | Runtime |
| primer3-py | ≥ 2.0.2 | Thermodynamic NN calculations |
| mappy | ≥ 2.26 | minimap2 pangenome alignment |
| lightgbm | ≥ 4.3.0 | GBDT ensemble |
| pulp | ≥ 2.8.0 | ILP solver |
| numpy | ≥ 1.26 | Numerical computations |
| pandas | ≥ 2.2 | Data handling |
| matplotlib | ≥ 3.8 | Plotting |
| streamlit | ≥ 1.35 | Web server |
| click | ≥ 8.1 | CLI |
| pytest | ≥ 8.2 | Test suite |

---

*End of Supplementary Methods*
