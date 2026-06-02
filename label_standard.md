# PrimerForge PCR Primer Dataset Curation and Labeling Standard

This document defines the strict, production-ready dataset labeling standard for building and curating the PrimerForge PCR primer training dataset. Following this standard guarantees that independent curators and automated pipelines achieve $\ge 95\%$ labeling agreement.

---

## Section 1: Binary Label Definitions

Curated primer pairs must be assigned one of three labels based on their experimental validation outcomes.

### 1. Label = 1 (Functional)
A primer pair is labeled as **1 (Functional)** if it meets at least one of the following criteria:
- **Gel Electrophoresis**: Yields a single, clean, distinct gel band at the expected amplicon size, with no visible secondary bands or primer-dimers.
- **qPCR Cycle Threshold**: Achieves a cycle threshold ($C_t$) value $\le 35$ cycles under standard reaction conditions (defined as a $60^\circ\text{C}$ annealing temperature and $2.0\,\text{mM}$ $\text{MgCl}_2$).
- **Amplification Efficiency**: Exhibits a qPCR amplification efficiency $E \ge 1.80$ (where $E = 10^{-1/\text{slope}} - 1$, corresponding to $90\text{--}110\%$) with a linear regression coefficient of determination $R^2 \ge 0.98$ across a minimum 4-log serial dilution.
- **Clinical Validation**: Included in a validated, active diagnostic panel officially released by the WHO, CDC, or the ARTIC network.

### 2. Label = 0 (Non-Functional)
A primer pair is labeled as **0 (Non-Functional)** if it exhibits any of the following failure modes:
- **No Amplification**: Shows no target-specific product band on a gel or no qPCR signal after 35 cycles at any annealing temperature.
- **Non-Specific Amplification**: Yields multiple non-specific bands, smear patterns, or primer-dimers on a gel with no clear dominant band at the expected product size.
- **Poor Efficiency**: Displays a qPCR amplification efficiency $E < 1.60$ (below $80\%$), indicating poor annealing or primer interference.
- **Explicit Literature Classification**: Described explicitly in the source publication text or supplementary data tables as "failed," "non-specific," "poorly performing," or "unusable."

### 3. Label = DISCARD (Exclude)
A primer pair must be **discarded** and excluded from the database if it falls into any of the following categories:
- **Borderline Efficiency**: Exhibits a qPCR amplification efficiency $E$ between $1.60$ and $1.79$ (where performance is highly protocol-dependent).
- **Prokaryotic Targets**: Validated exclusively on bacterial, archaeal, or environmental metagenomic targets (due to different genomic complexities).
- **Degenerate Primers**: Contains any IUPAC degeneracy codes (e.g., R, Y, S, W, K, M, B, D, H, V, N) rather than standard ATGC nucleotides.
- **Non-Standard Length**: Has a primer length strictly $< 15\,\text{nt}$ or $> 35\,\text{nt}$.

---

## Section 2: Source Trust Hierarchy

When different databases or publications report conflicting validation labels for the exact same primer pair sequence, the conflict must be resolved using the following priority hierarchy (1 is highest trust, 4 is lowest trust):

1. **RTPrimerDB Efficiency Value (Highest Trust)**: Continuous experimental metrics (amplification efficiency $E$ and $R^2$ values) override any qualitative descriptions.
2. **ARTIC/CDC/WHO Clinical Panel Inclusion**: Consensus clinical diagnostic validation overrides general academic database entries.
3. **PrimerBank Database Entry**: Cell-line validated assays (lacking raw efficiency measurements) override general literature reviews.
4. **PMC Paper Supplementary Table (Lowest Trust)**: High-throughput screening lists (where success is often inferred from gene expression levels rather than direct qPCR validation) have the lowest priority.

---

## Section 3: Inclusion Boundaries

A primer pair is eligible for inclusion in the database if and only if **all** of the following boundary conditions are met:

- **GC Content**: Both the forward and reverse primers must have a GC content between **30% and 75%**.
- **Melting Temperature ($T_m$)**: Both primers must have a melting temperature between **$50^\circ\text{C}$ and $72^\circ\text{C}$**, calculated using the SantaLucia nearest-neighbor thermodynamic method under standardized conditions ($50\,\text{mM}$ monovalent salt, $200\,\text{nM}$ primer concentration).
- **Amplicon Size**: The expected PCR product length must be between **$60\,\text{bp}$ and $600\,\text{bp}$** (excluding long-range PCR assays).
- **Organism Constraint**: The targets must be **eukaryotic (human or mouse) or viral** genomes only. Bacterial and archaeal targets are strictly excluded.
- **Sequence Constraints**: Sequences must consist of uppercase **A, T, G, and C** characters only. No lowercase letters, spaces, numbers, or ambiguity codes are permitted.

---

## Section 4: Edge Case Rules

To maintain high curation consistency, the following rules must be applied to standard edge cases:

1. **Species-Specific Discrepancies**: If a primer pair is reported to work successfully in mouse tissue but fails in human tissue, it must be split into two separate database records if the target sequences differ. If the target sequence is identical, the pair must be labeled as **0 (Non-Functional)** to remain conservative.
2. **Unreported Amplicon Size**: If a publication fails to report the amplicon size, the curator must retrieve the target GenBank accession and run an in-silico PCR to determine the product length. If the in-silico product is outside the $60\text{--}600\,\text{bp}$ range or cannot be uniquely resolved, the pair must be labeled as **DISCARD**.
3. **Qualitative Band Reporting**: If a paper reports a "positive band of the expected size" in the text but fails to provide a gel image or raw image data, the pair is labeled as **1 (Functional)** only if the paper has undergone peer review and explicitly details the product size validation. Otherwise, it must be labeled as **DISCARD**.
4. **SYBR Green vs. Probe-Based qPCR**: If a primer pair is validated using both SYBR Green and TaqMan probe assays with conflicting efficiency results, the SYBR Green efficiency must be used for standard classification because probe assays can mask poor primer performance (e.g., primer-dimers).
5. **Nested PCR Assays**: For nested PCR assays, the outer and inner primer pairs must be treated as independent pairs. The outer pair is labeled based on the first-round amplification, and the inner pair is labeled based on the second-round amplification.
6. **Accession Mismatches**: If the primer sequence reported in the text of a publication does not match the reference sequence listed in the cited GenBank accession (due to transcription errors or genome assembly updates), the GenBank accession sequence takes priority, and the published text sequence is labeled as **DISCARD**.
7. **Non-Standard Reaction Conditions**: Primers validated under non-standard chemistry (e.g., $> 5.0\,\text{mM}$ $\text{MgCl}_2$, custom additives like betaine, or extreme annealing temperatures outside the $50\text{--}72^\circ\text{C}$ range) must be labeled as **DISCARD** unless they are fine-tuning samples explicitly targeted for Task B.
8. **Multi-Locus Binding (Off-Target)**: If in-silico PCR reveals that a primer pair binds to multiple loci in the target genome and yields multiple expected amplicons of similar sizes, the pair must be labeled as **0 (Non-Functional)** due to lack of specificity, regardless of gel reporting.

---

## Section 5: Audit Trail Requirement

Every record imported into the PrimerForge database must contain a complete, non-null audit trail consisting of the following metadata fields:

- `source_db` or `paper_doi`: Must contain either a valid database identifier (e.g., `rtprimerdb`, `primerbank`) or a verified DOI string.
- `accession_id`: The target NCBI GenBank accession number (e.g., `NM_002046.7`, `MN908947.3`) where the template sequence is located (use `N/A` only if targeting an artificial construct).
- `date_collected`: The timestamp when the record was curated and parsed, formatted in ISO 8601 format (`YYYY-MM-DD`).

---

## Section 6: Labeling Standard Summary Table

| Label | Designation | Primary Criteria | Action |
| :--- | :--- | :--- | :--- |
| **1** | Functional | Single gel band at expected size, OR qPCR $C_t \le 35$ (at $60^\circ\text{C}$ anneal, $2\,\text{mM}$ $\text{MgCl}_2$), OR qPCR efficiency $E \ge 1.80$ ($R^2 \ge 0.98$), OR WHO/CDC/ARTIC panel inclusion. | **Include** in training and testing datasets. |
| **0** | Non-Functional | No PCR band after 35 cycles, OR multiple non-specific bands/primer-dimers, OR qPCR efficiency $E < 1.60$, OR explicitly described as failed/non-specific in source paper. | **Include** as negative samples in datasets. |
| **DISCARD** | Exclude | qPCR efficiency $E \in [1.60, 1.79]$, OR prokaryotic target, OR contains degenerate IUPAC codes, OR primer length $<15\,\text{nt}$ or $>35\,\text{nt}$. | **Exclude** from all machine learning training runs. |
