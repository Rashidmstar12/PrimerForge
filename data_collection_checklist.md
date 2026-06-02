# PrimerForge Data Collection Quality Control Checklist

This checklist defines the strict quality control constraints that must be run on every primer pair record before appending it to the master training database file `data/master_training_db_v2.csv`.

---

## Tier 1 — Sequence Validity Checks (4 checks)
*Failing any check in this tier results in immediate **REJECTION** of the entry. These checks detect critical structural anomalies or data corruption.*

### 1. Check 1 — Base Alphabet Validity
- **What it detects**: Degenerate nucleotides (e.g., N, R, Y, S) or non-standard characters (whitespace, numbers, dashes) in primer sequences.
- **Pass Condition**: Both `sequence_fwd` and `sequence_rev` contain only uppercase `A`, `T`, `G`, and `C` characters.
- **Fail Action**: **REJECT**
- **Failing Example**:
  - `sequence_fwd`: `GGAGCGAGATCCCTCCANAT` (contains N)

### 2. Check 2 — Nucleotide Length Boundaries
- **What it detects**: Primers that are too short (insufficient specificity) or too long (slow annealing, non-standard synthesis).
- **Pass Condition**: Both `sequence_fwd` and `sequence_rev` must have a length $\ge 15$ and $\le 35$ nucleotides.
- **Fail Action**: **REJECT**
- **Failing Example**:
  - `sequence_fwd`: `GGAGCGAGATCC` (12 nt - too short)

### 3. Check 3 — Reverse Complement Duplication (Self-Annealing)
- **What it detects**: Primers that are identical to their own reverse complements, which would fold and self-anneal perfectly, preventing target binding.
- **Pass Condition**: Neither `sequence_fwd` nor `sequence_rev` is identical to its own reverse complement.
- **Fail Action**: **REJECT**
- **Failing Example**:
  - `sequence_fwd`: `GCATGC` (reverse complement is `GCATGC` - self-anneals perfectly)

### 4. Check 4 — Sequence Identity (Copy-Paste Duplication)
- **What it detects**: Database entry copy-paste errors where the same sequence is assigned to both the forward and reverse primers.
- **Pass Condition**: `sequence_fwd` is NOT identical to `sequence_rev`.
- **Fail Action**: **REJECT**
- **Failing Example**:
  - `sequence_fwd`: `GGAGCGAGATCCCTCCAAAT`
  - `sequence_rev`: `GGAGCGAGATCCCTCCAAAT` (identical to fwd)

---

## Tier 2 — Biophysical Boundary Checks (4 checks)
*Failing any check in this tier results in the entry being **DISCARDED** from standard training datasets. These checks verify that the primers follow standard PCR design rules.*

### 5. Check 5 — GC Content Boundaries
- **What it detects**: Primers with extreme GC content that would lead to unstable annealing or structural problems.
- **Pass Condition**: The calculated GC content percentage for both primers is $\ge 30\%$ and $\le 75\%$.
- **Fail Action**: **DISCARD**
- **Failing Example**:
  - `sequence_fwd`: `ATATATATATATATATATAT` (0% GC content)

### 6. Check 6 — Melting Temperature ($T_m$) Boundaries
- **What it detects**: Primers with melting temperatures outside functional diagnostic parameters.
- **Pass Condition**: The calculated $T_m$ for both primers is $\ge 50^\circ\text{C}$ and $\le 72^\circ\text{C}$ (using SantaLucia 1998 nearest-neighbor parameters with $50\,\text{mM}$ monovalent salt and $200\,\text{nM}$ primer concentration).
- **Fail Action**: **DISCARD**
- **Failing Example**:
  - `sequence_fwd`: `CGGCGGCCGCGGCC` ($T_m > 75^\circ\text{C}$)

### 7. Check 7 — Amplicon Size Range
- **What it detects**: Long-range PCR entries or extremely short targets that are outside typical qPCR/multiplex panel standards.
- **Pass Condition**: The expected `amplicon_size_bp` is $\ge 60\,\text{bp}$ and $\le 600\,\text{bp}$.
- **Fail Action**: **DISCARD**
- **Failing Example**:
  - `amplicon_size_bp`: `1200` (long-range PCR)

### 8. Check 8 — Homopolymer Runs
- **What it detects**: Long homopolymer runs (poly-runs) that cause polymerase slippage or target secondary structure issues.
- **Pass Condition**: Neither `sequence_fwd` nor `sequence_rev` contains a run of more than 5 consecutive identical nucleotides (e.g., AAAAAA).
- **Fail Action**: **DISCARD**
- **Failing Example**:
  - `sequence_fwd`: `GGAGCGAGAAAAAAATCCCT` (contains run of 7 As)

---

## Tier 3 — Label and Provenance Checks (4 checks)
*Failing any check in this tier results in the entry being **FLAGGED** for manual curator review. These checks preserve metadata audit logs.*

### 9. Check 9 — Label Integrity
- **What it detects**: Blank, corrupted, fractional, or non-binary class labels.
- **Pass Condition**: The `label` column is exactly `0` or `1` (as integers or floats representing exact integers).
- **Fail Action**: **FLAG**
- **Failing Example**:
  - `label`: `0.5` (fractional score not permitted in classification)

### 10. Check 10 — Label Confidence Validation
- **What it detects**: Unrecognized or missing confidence classifications.
- **Pass Condition**: The `label_confidence` field is exactly one of: `high`, `medium`, `low` (case-insensitive).
- **Fail Action**: **FLAG**
- **Failing Example**:
  - `label_confidence`: `unverified`

### 11. Check 11 — Provenance Attribution
- **What it detects**: Orphan rows in the database that lack trace academic or database origin references.
- **Pass Condition**: At least one of `paper_doi` or `source_db` must be non-null, non-empty, and not equal to a blank/NaN value.
- **Fail Action**: **FLAG**
- **Failing Example**:
  - `paper_doi`: `""`
  - `source_db`: `""`

### 12. Check 12 — Unique Identifier format and Integrity
- **What it detects**: Violations of the database identity format or duplicate records.
- **Pass Condition**: `primer_id` follows the naming convention `SOURCEDB_GENENAME_INDEX` (verified via regex pattern) and is not already present in the existing database keys.
- **Fail Action**: **FLAG**
- **Failing Example**:
  - `primer_id`: `GAPDH-fwd-primer` (does not match naming convention)

---

## Programmatic Quality Control Validation Script

The following Python function implements all 12 validation checks. It can be imported directly into any data collection script or ingestion pipeline.

```python
import re
import math
from typing import Tuple, List, Dict, Any

def calculate_gc(seq: str) -> float:
    """Calculates GC percentage of a sequence."""
    if not seq:
        return 0.0
    return sum(1 for b in seq.upper() if b in "GC") / len(seq) * 100.0

def calculate_santa_lucia_tm(seq: str) -> float:
    """
    Approximates melting temperature using SantaLucia 1998 nearest-neighbor parameters.
    Parameters are calibrated for 50 mM NaCl and 200 nM primer concentration.
    """
    seq = seq.upper()
    # nearest-neighbor interactions dH (kcal/mol), dS (cal/mol*K)
    nn_table = {
        'AA': (-7.9, -22.2), 'TT': (-7.9, -22.2),
        'AT': (-7.2, -20.4), 'TA': (-7.2, -21.3),
        'CA': (-8.5, -22.7), 'TG': (-8.5, -22.7),
        'GT': (-8.4, -22.4), 'AC': (-8.4, -22.4),
        'CT': (-7.8, -21.0), 'AG': (-7.8, -21.0),
        'GA': (-8.2, -22.2), 'TC': (-8.2, -22.2),
        'CG': (-10.6, -27.2), 'GC': (-9.8, -24.4),
        'GG': (-8.0, -19.9), 'CC': (-8.0, -19.9)
    }
    
    # Initiation parameters
    init_h, init_s = 0.0, 0.0
    # End terminal values
    ends = [seq[0], seq[-1]]
    for end in ends:
        if end in ['G', 'C']:
            init_h += 0.1
            init_s += -2.8
        elif end in ['A', 'T']:
            init_h += 2.3
            init_s += 4.1
            
    # Accumulate nearest-neighbor values
    total_h = init_h
    total_s = init_s
    for i in range(len(seq) - 1):
        pair = seq[i:i+2]
        if pair in nn_table:
            total_h += nn_table[pair][0]
            total_s += nn_table[pair][1]
            
    # Monovalent salt correction for entropy (dS): 50mM = 0.05M
    # dS_corr = dS + 0.368 * (N - 1) * ln(salt)
    salt_monovalent = 0.05
    total_s += 0.368 * (len(seq) - 1) * math.log(salt_monovalent)
    
    # concentration correction for non-self-complementary primers: R * ln(Ct/4)
    # Ct = 200 nM = 2e-7 M. R = 1.9872 cal/mol*K
    conc_corr = 1.9872 * math.log(2e-7 / 4.0)
    
    # convert dH from kcal to cal: total_h * 1000
    tm = (total_h * 1000.0) / (total_s + conc_corr) - 273.15
    return round(tm, 2)

def get_reverse_complement(seq: str) -> str:
    """Computes the reverse complement of a DNA sequence."""
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return "".join(comp[b] for b in reversed(seq.upper()))

def has_poly_run(seq: str, max_len: int = 5) -> bool:
    """Checks if a sequence has any homopolymer run strictly greater than max_len."""
    if len(seq) <= max_len:
        return False
    # Regex matching consecutive character runs
    pattern = re.compile(r"([A-Z])\1{" + str(max_len) + r",}")
    return bool(pattern.search(seq.upper()))

def validate_primer_entry(row: Dict[str, Any], existing_ids: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Performs quality control validation checking 12 parameters on a primer row dictionary.
    Returns (True, []) if all pass, or (False, [failed_checks]) if any fail.
    """
    failed_checks = []
    existing_ids = existing_ids or []
    
    f_seq = str(row.get('sequence_fwd', '')).strip().upper()
    r_seq = str(row.get('sequence_rev', '')).strip().upper()
    
    # --- TIER 1: Sequence Validity Checks ---
    
    # Check 1: Base Alphabet Validity
    if not (f_seq and r_seq) or not (re.match(r"^[ATGC]+$", f_seq) and re.match(r"^[ATGC]+$", r_seq)):
        failed_checks.append("CHECK_1_INVALID_ALPHABET")
        
    # Check 2: Length Boundaries
    if not (15 <= len(f_seq) <= 35) or not (15 <= len(r_seq) <= 35):
        failed_checks.append("CHECK_2_INVALID_LENGTH")
        
    # Check 3: Self-Annealing Reverse Complement
    if (f_seq and f_seq == get_reverse_complement(f_seq)) or (r_seq and r_seq == get_reverse_complement(r_seq)):
        failed_checks.append("CHECK_3_SELF_ANNEALING")
        
    # Check 4: Sequence Identity Duplication
    if f_seq and f_seq == r_seq:
        failed_checks.append("CHECK_4_DUPLICATE_SEQUENCES")
        
    # --- TIER 2: Biophysical Boundary Checks ---
    # Only calculate if sequence alphabet checks pass to prevent logic errors
    if "CHECK_1_INVALID_ALPHABET" not in failed_checks:
        # Check 5: GC content boundaries
        f_gc = calculate_gc(f_seq)
        r_gc = calculate_gc(r_seq)
        if not (30.0 <= f_gc <= 75.0) or not (30.0 <= r_gc <= 75.0):
            failed_checks.append("CHECK_5_EXTREME_GC")
            
        # Check 6: Melting Temperature boundaries
        try:
            f_tm = calculate_santa_lucia_tm(f_seq)
            r_tm = calculate_santa_lucia_tm(r_seq)
            if not (50.0 <= f_tm <= 72.0) or not (50.0 <= r_tm <= 72.0):
                failed_checks.append("CHECK_6_EXTREME_TM")
        except Exception:
            failed_checks.append("CHECK_6_EXTREME_TM")
            
        # Check 8: Homopolymer Runs
        if has_poly_run(f_seq, 5) or has_poly_run(r_seq, 5):
            failed_checks.append("CHECK_8_HOMOPOLYMER_RUN")
            
    # Check 7: Amplicon size range
    amp_size = row.get('amplicon_size_bp')
    if amp_size is not None:
        try:
            amp_val = int(amp_size)
            if not (60 <= amp_val <= 600):
                failed_checks.append("CHECK_7_INVALID_AMPLICON_SIZE")
        except ValueError:
            failed_checks.append("CHECK_7_INVALID_AMPLICON_SIZE")
            
    # --- TIER 3: Label and Provenance Checks ---
    
    # Check 9: Label Integrity
    label = row.get('label')
    if label not in [0, 1]:
        failed_checks.append("CHECK_9_INVALID_LABEL")
        
    # Check 10: Label Confidence validation
    conf = str(row.get('label_confidence', '')).strip().lower()
    if conf not in ['high', 'medium', 'low']:
        failed_checks.append("CHECK_10_INVALID_CONFIDENCE")
        
    # Check 11: Provenance Attribution
    doi = str(row.get('paper_doi', '')).strip()
    db = str(row.get('source_db', '')).strip()
    if (not doi or doi.upper() in ['', 'NAN', 'NONE', 'NULL']) and (not db or db.upper() in ['', 'NAN', 'NONE', 'NULL']):
        failed_checks.append("CHECK_11_MISSING_PROVENANCE")
        
    # Check 12: Unique Identifier structure & Duplication
    p_id = str(row.get('primer_id', '')).strip().upper()
    id_match = re.match(r"^[A-Z0-9]+_[A-Z0-9]+_[0-9]{3}$", p_id)
    if not id_match:
        failed_checks.append("CHECK_12_INVALID_ID_FORMAT")
    elif p_id in existing_ids:
        failed_checks.append("CHECK_12_DUPLICATE_ID")
        
    if failed_checks:
        return False, failed_checks
    return True, []
```
