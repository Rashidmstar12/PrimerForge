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
    return "".join(comp.get(b, b) for b in reversed(seq.upper()))

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
    if amp_size is not None and not (isinstance(amp_size, float) and math.isnan(amp_size)):
        try:
            amp_val = int(amp_size)
            if not (60 <= amp_val <= 600):
                failed_checks.append("CHECK_7_INVALID_AMPLICON_SIZE")
        except (ValueError, TypeError):
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
