# PrimerForge Master Training Database Column Schema Specification

This document defines the strict schema, validation rules, and structural constraints for the master training database CSV file `data/master_training_db_v2.csv` used by the PrimerForge platform.

---

## Column Definitions by Group

### Group 1 — Identity (5 columns)

| Column Name | Data Type | Allowed Range / Values | Required / Optional | Example Value | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `primer_id` | string | Pattern: `^[A-Z0-9]+_[A-Z0-9]+_[0-9]{3}$` | Required | `RTPDB_GAPDH_001` | Unique alphanumeric identifier generated using the derivation rules below. |
| `gene_name` | string | Non-empty alphanumeric string | Required | `GAPDH` | HGNC symbol or gene abbreviation targeted by the primer pair. |
| `organism` | string | `human`, `mouse`, or `sars-cov-2` | Required | `human` | Target species genome. |
| `source_db` | string | `rtprimerdb`, `primerbank`, `artic`, `cdc`, `who`, `literature` | Required | `rtprimerdb` | Database or source repository name from which the entry was collected. |
| `paper_doi` | string | Valid DOI pattern or `N/A` | Required | `10.1093/nar/gks1219` | The digital object identifier of the validating publication. |

### Group 2 — Sequences (2 columns)

| Column Name | Data Type | Allowed Range / Values | Required / Optional | Example Value | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sequence_fwd` | string | Uppercase `[A, T, G, C]` only, length 15–35 nt | Required | `GGAGCGAGATCCCTCCAAAT` | 5'-to-3' forward primer sequence. No degeneracies allowed. |
| `sequence_rev` | string | Uppercase `[A, T, G, C]` only, length 15–35 nt | Required | `GGCTGTTGTCATACTTCTCATGG` | 5'-to-3' reverse primer sequence. No degeneracies allowed. |

### Group 3 — Labels (4 columns)

| Column Name | Data Type | Allowed Range / Values | Required / Optional | Example Value | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `label` | int | `0`, `1` | Required | `1` | Binary class label. `1` = functional, `0` = non-functional (according to `label_standard.md`). |
| `label_confidence` | string | `high`, `medium`, `low` | Required | `high` | Curation confidence score based on validation data type. |
| `label_source` | string | `rtprimerdb`, `primerbank`, `artic`, `pmc`, `synthetic` | Required | `rtprimerdb` | Database origin of the annotation label. |
| `label_notes` | string | Free text | Optional | `DMSO added to reaction mix` | Explanatory notes regarding unusual testing setups or protocols. |

### Group 4 — Experimental Metadata (7 columns)

| Column Name | Data Type | Allowed Range / Values | Required / Optional | Example Value | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cell_line_tested` | string | Non-empty string or empty | Optional | `HEK293` | Cell line or tissue type used to extract validation template. |
| `annealing_tm_used_C` | float | `45.0` to `75.0` | Optional | `60.0` | Actual annealing temperature used during validation runs. |
| `amplicon_size_bp` | int | `60` to `600` | Required | `226` | Observed or expected size of the PCR product. |
| `qpcr_efficiency` | float | `1.0` to `2.5` | Optional | `1.95` | Raw amplification efficiency value (where $E=2.0$ is perfect duplication). |
| `qpcr_r2` | float | `0.0` to `1.0` | Optional | `0.992` | $R^2$ coefficient of standard qPCR standard curves. |
| `gel_band_confirmed` | bool | `True`, `False` | Required | `True` | Indicates if a gel band of the expected size was confirmed. |
| `date_collected` | string | Date pattern: `YYYY-MM-DD` | Required | `2026-06-02` | Timestamp indicating when the entry was parsed. |

### Group 5 — Biophysical Features (16 columns — computed by BiophysicsEngine)

| Column Name | Data Type | Allowed Range / Values | Required / Optional | Example Value | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `f_tm` | float | `50.0` to `72.0` | Required | `59.2` | Calculated melting temperature of forward primer ($^\circ\text{C}$). |
| `r_tm` | float | `50.0` to `72.0` | Required | `58.9` | Calculated melting temperature of reverse primer ($^\circ\text{C}$). |
| `tm_diff` | float | `0.0` to `15.0` | Required | `0.3` | Absolute melting temperature difference ($\|T_{m,\text{fwd}} - T_{m,\text{rev}}\|$). |
| `f_gc` | float | `30.0` to `75.0` | Required | `50.0` | GC content percentage of the forward primer. |
| `r_gc` | float | `30.0` to `75.0` | Required | `47.8` | GC content percentage of the reverse primer. |
| `f_hairpin_dg` | float | `-20.0` to `0.0` | Required | `-0.2` | Predicted hairpin formation free energy ($\Delta G$ in kcal/mol). |
| `r_hairpin_dg` | float | `-20.0` to `0.0` | Required | `0.0` | Predicted hairpin formation free energy ($\Delta G$ in kcal/mol). |
| `cross_dimer_dg` | float | `-30.0` to `0.0` | Required | `-3.4` | Predicted cross-dimerization free energy ($\Delta G$ in kcal/mol). |
| `f_len` | int | `15` to `35` | Required | `20` | Length of the forward primer sequence in nucleotides. |
| `r_len` | int | `15` to `35` | Required | `23` | Length of the reverse primer sequence in nucleotides. |
| `f_clamp_gc` | int | `0` to `5` | Required | `2` | Number of G/C bases in the 3' terminal 5 nucleotides of the fwd primer. |
| `r_clamp_gc` | int | `0` to `5` | Required | `3` | Number of G/C bases in the 3' terminal 5 nucleotides of the rev primer. |
| `f_poly_run` | int | `1` to `15` | Required | `2` | Maximum homopolymer run length in the forward primer. |
| `r_poly_run` | int | `1` to `15` | Required | `1` | Maximum homopolymer run length in the reverse primer. |
| `target_gc` | float | `10.0` to `90.0` | Required | `48.2` | Average GC content percentage of the target amplicon region. |
| `target_len` | int | `60` to `600` | Required | `226` | Target amplicon sequence length in base pairs. |

### Group 6 — Dataset Flags (3 columns)

| Column Name | Data Type | Allowed Range / Values | Required / Optional | Example Value | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `is_synthetic_negative` | bool | `True`, `False` | Required | `False` | Flag identifying manually injected negative controls. |
| `is_prospective_validation` | bool | `True`, `False` | Required | `False` | Flag identifying wet-lab validated prospective designs. |
| `is_held_out_test` | bool | `True`, `False` | Required | `False` | Flag locking the row to the held-out test splits. |

---

## Column Derivation Rules

### 1. Unique Identifiers (`primer_id`)
To prevent duplicate records and maintain database key integrity, the `primer_id` field must follow a strict, uppercase alphanumeric prefix string matching the schema `SOURCEDB_GENENAME_INDEX`:
- **SOURCEDB**: Short capitalized abbreviation mapping to `source_db` (e.g., `RTPDB` for RTPrimerDB, `PMB` for PrimerBank, `ARTIC` for ARTIC, `LIT` for academic literature).
- **GENENAME**: Normalized HGNC identifier or virus gene code (uppercase, alphanumeric, e.g., `GAPDH`, `ACTB`, `ORF1AB`).
- **INDEX**: A 3-digit zero-padded index (e.g., `001`, `002`) to handle cases where multiple different primers target the same gene.
- **Example**: `RTPDB_GAPDH_001` or `LIT_SARSCOV2_023`.

### 2. Missing Optional Fields Handling
- **Empty Strings**: For string columns (`cell_line_tested`, `label_notes`), missing values must be represented as a blank empty string (`""`) in the CSV cell.
- **Numerical NaN Values**: For optional numerical float or int columns (`annealing_tm_used_C`, `qpcr_efficiency`, `qpcr_r2`), missing records must be written as an empty cell (which parses as `NaN` or `None` in Pandas), rather than default values (like `0` or `-1.0`), to prevent skewing statistical features.

---

## Validation Script Pseudocode

Before any row is appended to `data/master_training_db_v2.csv`, the ingestion pipeline must run the following ten checks. If a row fails any check, the record must be rejected and logged to a validation error file:

```python
def validate_primer_record(row, biophysics_engine):
    # Check 1: Verify Sequence format (uppercase, letters only)
    if not (row['sequence_fwd'].isupper() and row['sequence_fwd'].isalpha()):
        return False, "Forward sequence contains non-alphabetical or lowercase letters."
    if not (row['sequence_rev'].isupper() and row['sequence_rev'].isalpha()):
        return False, "Reverse sequence contains non-alphabetical or lowercase letters."
        
    # Check 2: Verify only valid ATGC characters exist
    valid_nucleotides = set("ATGC")
    if not set(row['sequence_fwd']).issubset(valid_nucleotides):
        return False, "Forward sequence contains degenerate nucleotides or invalid bases."
    if not set(row['sequence_rev']).issubset(valid_nucleotides):
        return False, "Reverse sequence contains degenerate nucleotides or invalid bases."
        
    # Check 3: Verify sequence length is inside range
    if not (15 <= len(row['sequence_fwd']) <= 35):
        return False, f"Forward sequence length is {len(row['sequence_fwd'])}, must be 15-35 nt."
    if not (15 <= len(row['sequence_rev']) <= 35):
        return False, f"Reverse sequence length is {len(row['sequence_rev'])}, must be 15-35 nt."
        
    # Check 4: Check if binary label is either 0 or 1
    if row['label'] not in [0, 1]:
        return False, f"Invalid label '{row['label']}', must be 0 or 1."
        
    # Check 5: Verify required metadata date format
    import re
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", row['date_collected']):
        return False, "date_collected must match YYYY-MM-DD ISO 8601 format."
        
    # Check 6: Check for correct primer_id naming convention
    id_pattern = r"^[A-Z0-9]+_[A-Z0-9]+_\d{3}$"
    if not re.match(id_pattern, row['primer_id']):
        return False, f"primer_id '{row['primer_id']}' violates SOURCEDB_GENENAME_INDEX structure."
        
    # Check 7: Recalculate and assert Biophysical GC limits
    f_gc = sum(1 for b in row['sequence_fwd'] if b in "GC") / len(row['sequence_fwd']) * 100.0
    r_gc = sum(1 for b in row['sequence_rev'] if b in "GC") / len(row['sequence_rev']) * 100.0
    if not (30.0 <= f_gc <= 75.0) or not (30.0 <= r_gc <= 75.0):
        return False, f"Calculated GC content (Fwd: {f_gc:.1f}%, Rev: {r_gc:.1f}%) is outside the 30-75% limits."
        
    # Check 8: Verify calculated Tm limits via Nearest-Neighbor method
    f_tm = biophysics_engine.calculate_tm(row['sequence_fwd'])
    r_tm = biophysics_engine.calculate_tm(row['sequence_rev'])
    if not (50.0 <= f_tm <= 72.0) or not (50.0 <= r_tm <= 72.0):
        return False, f"Calculated Tm (Fwd: {f_tm:.1f}°C, Rev: {r_tm:.1f}°C) is outside the 50-72°C limits."
        
    # Check 9: Validate amplicon size bounds
    if not (60 <= row['amplicon_size_bp'] <= 600):
        return False, f"amplicon_size_bp '{row['amplicon_size_bp']}' is outside eukaryotic/diagnostic bounds (60-600 bp)."
        
    # Check 10: Validate required non-empty target features
    if row['organism'].lower() not in ['human', 'mouse', 'sars-cov-2']:
        return False, f"Organism '{row['organism']}' is not supported (only eukaryotic/viral genomes permitted)."
        
    return True, "Passed all validation assertions."
```

---

## Master Database Example Rows

Here is an example containing 5 valid, comma-separated rows. This dataset includes a validated human target (GAPDH), a mouse housekeeper (ACTB), a clinical SARS-CoV-2 target, a synthetically injected negative control, and a borderline entry showing handled fields:

```csv
primer_id,gene_name,organism,source_db,paper_doi,sequence_fwd,sequence_rev,label,label_confidence,label_source,label_notes,cell_line_tested,annealing_tm_used_C,amplicon_size_bp,qpcr_efficiency,qpcr_r2,gel_band_confirmed,date_collected,f_tm,r_tm,tm_diff,f_gc,r_gc,f_hairpin_dg,r_hairpin_dg,cross_dimer_dg,f_len,r_len,f_clamp_gc,r_clamp_gc,f_poly_run,r_poly_run,target_gc,target_len,is_synthetic_negative,is_prospective_validation,is_held_out_test
PMB_GAPDH_001,GAPDH,human,primerbank,10.1093/nar/gks1219,GGAGCGAGATCCCTCCAAAT,GGCTGTTGTCATACTTCTCATGG,1,high,primerbank,Validated in primary tissue,HEK293,60.0,226,1.96,0.995,True,2026-06-02,59.2,60.1,0.9,50.0,52.2,-0.1,-0.3,-2.4,20,23,2,3,2,1,48.2,226,False,False,False
RTPDB_ACTB_001,ACTB,mouse,rtprimerdb,10.1186/gb-2004-5-10-r80,AAGACCTGTACGCCAACACA,GCGCTCAGGAGGAGCAATGA,1,high,rtprimerdb,Standard housekeeping pair,,60.0,154,1.92,0.991,True,2026-06-02,58.8,59.5,0.7,50.0,60.0,0.0,-0.1,-1.8,20,20,2,3,1,2,51.3,154,False,False,False
ARTIC_SARSCOV2_001,ORF1AB,sars-cov-2,artic,10.1038/s41587-020-0581-y,AACTACAAACTAAATGTTGGCAT,ACACTACTGATGTACTTCAAA,1,high,artic,ARTIC V4 scheme pool 1,,55.0,400,,,True,2026-06-02,55.0,52.2,2.8,30.4,33.3,-0.1,0.0,-1.5,23,21,3,1,3,3,31.8,400,False,False,False
SYN_NEG_001,NONE,human,literature,N/A,ATGCATGCATGCATGCATGC,CGATCGATCGATCGATCGAT,0,medium,synthetic,Injected synthetic negative,Hela,,180,,,False,2026-06-02,54.2,55.1,0.9,50.0,50.0,-5.4,-4.8,-8.2,20,20,2,2,1,1,45.0,180,True,False,False
LIT_GAPDH_002,GAPDH,human,literature,10.1016/j.ygeno.2014.07.004,GGAGCGAGATCCCTCCAAAT,GGCTGTTGTCATACTTCTCACAA,1,medium,pmc,Borderline efficiency profile,HEK293,58.0,226,1.81,0.981,True,2026-06-02,59.2,58.1,1.1,50.0,47.8,-0.1,0.0,-2.2,20,23,2,1,2,2,48.2,226,False,False,True
```
