# PrimerForge Biophysical & Thermodynamic Diagnostic Suite: Scientific Methodology Reference

This document defines the mathematical, physical, and thermodynamic formulations utilized in the PrimerForge diagnostic suite. To ensure clinical-grade assay engineering, PrimerForge evaluates designed primer pairs and multiplex panels across three distinct performance indices: **Assay Viability Index (AVI)**, **Panel Synergy & Interference Index (PSII)**, and **Scheme Coverage & Uniformity Index (SCUI)**.

---

## 1. Single-Locus: Assay Viability Index (AVI)

The AVI assesses the physical viability, secondary structure obstructions, and mutation resilience of a single primer pair. It is divided into four thermodynamic and kinetic metrics:

### 1.1 Duplex Anchoring Stability ($\Delta G^\circ_{3'\text{-terminal}}$)
The stability of the $3'$ terminal region of each primer determines the efficiency and specificity of polymerase initiation. Stability is calculated using the Nearest-Neighbor (NN) unified thermodynamic parameters (SantaLucia 1998):

$$\Delta G^\circ(T) = \Delta H^\circ - T \Delta S^\circ + \Delta G^\circ_{\text{init}} + \Delta G^\circ_{\text{salt}}$$

Where:
*   $\Delta H^\circ$ and $\Delta S^\circ$ are the sum of enthalpy and entropy changes of the nearest-neighbor doublets in the terminal 5-mer of the $3'$ region.
*   $\Delta G^\circ_{\text{init}}$ is the initiation free energy ($+0.98 \text{ kcal/mol}$ for terminal G-C, $+1.03 \text{ kcal/mol}$ for terminal A-T).
*   $\Delta G^\circ_{\text{salt}}$ is the salt correction term adjusting for monovalent cation concentrations ($[\text{Na}^+]$):

$$\Delta S^\circ_{\text{salt}} = \Delta S^\circ_{\text{std}} + 0.368 \times (N - 1) \times \ln[\text{Na}^+]$$

#### Anchoring Rubric:
*   **Optimal ($[-5.0, -9.0] \text{ kcal/mol}$)**: Provides sufficient anchoring stability to initiate polymerase extension while preventing non-specific priming. GC clamp at the $3'$ end must contain 1-2 G/C bases.
*   **Under-clamped ($> -5.0 \text{ kcal/mol}$ or 0 G/C)**: Low hybridization affinity at the critical $3'$ terminus, leading to enzymatic initiation dropouts.
*   **Over-clamped ($< -9.0 \text{ kcal/mol}$ or $> 2$ G/C)**: Excessively stable binding at the $3'$ terminus, significantly increasing the probability of mispriming and non-specific amplification in complex genomes.

---

### 1.2 Self-Annealing & Secondary Structure Blockers
Competing intra-molecular and inter-molecular hybridization reactions deplete active primer concentration and obstruct elongation:

#### 1.2.1 Hairpin Loop Stability ($\Delta G^\circ_{\text{hairpin}}$)
Intra-molecular hairpin loops are modeled using Nussinov MFE dynamic programming folded structures, with loop-size and mismatch penalties:

$$\Delta G^\circ_{\text{hairpin}} \ge -4.0 \text{ kcal/mol} \quad (\text{Optimal})$$

#### 1.2.2 Dimerization Stability ($\Delta G^\circ_{\text{homodimer}}$ and $\Delta G^\circ_{\text{cross-dimer}}$)
Inter-molecular self-annealing (homodimers) and primer-pair binding (heterodimers) consume active reagents:

$$\Delta G^\circ_{\text{homodimer}} \ge -5.0 \text{ kcal/mol}, \quad \Delta G^\circ_{\text{cross-dimer}} \ge -5.0 \text{ kcal/mol} \quad (\text{Optimal})$$

#### 1.2.3 Target Amplicon Secondary Folding Obstruction ($f_{\text{paired}}$)
The secondary structure of the single-stranded amplicon sequence during the extension phase can physically block or stall Taq polymerase. The base-pairing density fraction ($f_{\text{paired}}$) is extracted from the Nussinov MFE secondary fold:

$$f_{\text{paired}} = \frac{2 \times N_{\text{paired}}}{L_{\text{amplicon}}}$$

#### Folding Rubric:
*   **Optimal ($f_{\text{paired}} \le 0.45$ or $\text{MFE} \ge -12.0 \text{ kcal/mol}$)**: Minimal target secondary folds, allowing rapid, continuous polymerase elongation.
*   **Obstructed ($f_{\text{paired}} > 0.45$ or $\text{MFE} < -12.0 \text{ kcal/mol}$)**: Stable structural folds (e.g. stem-loops) physically block extension, requiring PCR additives (DMSO/betaine) to destabilize.

---

### 1.3 Variant Mismatch Resilience ($S_{\text{mismatch}}$)
Polymorphic variants (SNPs/indels) overlapping the primer binding site weaken hybridization. The mismatch resilience model implements a Taq-weighted exponential decay:

$$S_{\text{mismatch}} = S_{\text{baseline}} \times \prod_{v \in V} \exp \left( - \lambda \cdot d(v, 3') \right)$$

Where:
*   $S_{\text{baseline}}$ is the baseline ML predicted success probability.
*   $v \in V$ is the set of polymorphic variants located in the primer hybridization zone.
*   $d(v, 3')$ is the physical distance (nucleotides) from the $3'$ terminal extension site. Mismatches near the $3'$ end severely disrupt polymerase anchoring.
*   $\lambda$ is the decay sensitivity coefficient ($\lambda \approx 0.15$).

---

## 2. Multiplex Panel: Panel Synergy & Interference Index (PSII)

Multiplex PCR requires simultaneous amplification of multiple targets in a single reaction vessel, making competitive kinetics and cross-reactivity critical.

### 2.1 Thermal Cohort Uniformity ($\Delta T_{m,\text{max}}$)
To ensure uniform amplification rates across all loci under identical cycling conditions:

$$\Delta T_{m,\text{max}} = \max_{p \in P} T_m(p) - \min_{p \in P} T_m(p)$$

*   **Optimal ($\le 2.0^\circ\text{C}$)**: Synchronic annealing kinetics. All primer pairs bind to their targets with identical thermodynamic rates, eliminating competitive amplification bias.
*   **Dropout Risk ($> 4.0^\circ\text{C}$)**: Competitive dropouts. Targets with higher $T_m$ values will preferentially amplify, consuming dNTPs and primers, while low-$T_m$ targets fail to amplify.

### 2.2 Global Dimerization Penalty ($\Phi(P)$)
The dimerization penalty matrix evaluates all pairwise inter-molecular interactions in the pool:

$$D(i, j) = \max \left( 0, - \Delta G^\circ_{\text{cross}}(i, j) - 6.0 \text{ kcal/mol} \right)$$

$$\Phi(P) = \sum_{i \in P, j \in P, i < j} D(i, j)$$

The Integer Linear Programming (ILP) optimizer selects a subset of loci that minimizes $\Phi(P)$ subject to target plex constraints. A global penalty of $\Phi(P) = 0.000$ represents a completely dimer-free panel.

---

## 3. Tiled Genome: Scheme Coverage & Uniformity Index (SCUI)

Tiled amplicon sequencing (e.g. for viral genomes like SARS-CoV-2) utilizes overlapping amplicons to capture a whole genome.

### 3.1 Scheme Uniformity ($CV_P$)
The Coefficient of Variation ($CV_P$) of the ML predicted success probability measures the uniformity of library preparation and sequencing depth:

$$CV_P = \frac{\sigma_P}{\mu_P} = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^N (S_{\text{ML}}(i) - \mu_P)^2}}{\mu_P}$$

Where:
*   $S_{\text{ML}}(i)$ is the predicted success probability of tile $i$.
*   $\mu_P$ is the average predicted success across all tiles.
*   $\sigma_P$ is the standard deviation.

#### Uniformity Rubric:
*   **Excellent ($CV_P \le 0.10$)**: Pristine coverage depth. Minimal sequencing bias between target amplicons.
*   **Suboptimal ($0.10 < CV_P \le 0.20$)**: Minor read-depth variance. Fully compensated by standard depth normalization.
*   **Low Uniformity ($CV_P > 0.20$)**: Extreme coverage bias. High-success tiles will consume sequencing reads, leaving low-success tiles with insufficient coverage depth.

### 3.2 Regional Amplification Bottlenecks ($N_{\text{stalled}}$)
Bottleneck zones are regions where amplification efficiency is severely compromised:

$$N_{\text{stalled}} = \sum_{i=1}^N \mathbb{I}(S_{\text{ML}}(i) < 0.50)$$

Where $\mathbb{I}$ is the indicator function. Stalled segments occur in zones of extreme GC content or heavy variant density.

---

## 4. Academic References

1.  **SantaLucia, J. (1998).** *A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-neighbor thermodynamics.* Proceedings of the National Academy of Sciences, 95(4), 1460-1465.
2.  **Nussinov, R., & Jacobson, A. B. (1980).** *Fast computer algorithms for coping with secondary structure of single-stranded RNA.* Proceedings of the National Academy of Sciences, 77(11), 6309-6313.
3.  **Owczarzy, R., et al. (2008).** *Predicting stability of DNA duplexes in amine buffers.* Biochemistry, 47(19), 5336-5353.
4.  **Breslauer, K. J., et al. (1986).** *Predicting DNA duplex stability from the base sequence.* Proceedings of the National Academy of Sciences, 83(11), 3746-3750.
