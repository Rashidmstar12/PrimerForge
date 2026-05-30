"""Specificity and genetic variation check engine for PrimerForge.

Integrates mappy (minimap2) for fast pangenome alignment and implements
VariantAwareFilter to parse VCF coordinates and penalize or reject candidate
primers containing SNPs/indels in their critical 3' terminal anchor region.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from primerforge.utils import setup_logger

logger = setup_logger("primerforge.specificity")

# Gracefully handle mappy missing on platforms without build environments (e.g. Windows without zlib)
try:
    import mappy as mp
    MAPPY_AVAILABLE = True
except ImportError:
    mp = None
    MAPPY_AVAILABLE = False
    logger.warning(
        "mappy (minimap2 Python bindings) is not installed or could not be loaded. "
        "PrimerForge will operate with a pure-Python fallback alignment engine. "
        "For publication-grade performance and high-throughput pangenome mapping, "
        "install zlib and a C++ compiler, and rebuild mappy."
    )


@dataclass(frozen=True)
class AlignmentHit:
    """Represents a specific primer off-target or target alignment hit."""

    contig: str
    start: int
    end: int
    strand: int  # +1 for forward, -1 for reverse complement
    mismatches: int
    is_primary: bool


class SpecificityEngine:
    """Core specificity engine handling pangenome indexing, mapping, and variant filtering."""

    def __init__(self) -> None:
        """Initializes the SpecificityEngine."""
        self.aligner: Any = None
        self.fasta_path: str | None = None
        self._fallback_db: Dict[str, str] = {}  # Pure-Python fallback target sequence database

    def index_pangenome(self, fasta_path: str) -> None:
        """Indexes the target pangenome or reference genome.

        Args:
            fasta_path: Path to the FASTA file containing reference genomes.
        """
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found at: {fasta_path}")

        self.fasta_path = fasta_path

        if MAPPY_AVAILABLE and mp is not None:
            logger.info(f"Indexing pangenome using mappy/minimap2: {fasta_path}...")
            try:
                # Use preset='sr' (short-read mode) for short primer sequences
                self.aligner = mp.Aligner(fasta_path, preset="sr")
                if not self.aligner:
                    raise RuntimeError("mappy failed to initialize Aligner.")
                logger.info("mappy pangenome index built successfully.")
            except Exception as e:
                logger.error(f"mappy index compilation failed: {e}. Falling back to pure-Python.")
                self.aligner = None
        
        # Load into fallback memory database (used always if mappy fails/is absent)
        if not self.aligner:
            logger.info(f"Loading reference into memory for fallback alignment: {fasta_path}...")
            self._load_fallback_db(fasta_path)

    def _load_fallback_db(self, fasta_path: str) -> None:
        """Loads a FASTA file into a simple in-memory key-value dictionary for fallback mapping."""
        self._fallback_db = {}
        current_header = ""
        current_seq: List[str] = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_header:
                        self._fallback_db[current_header] = "".join(current_seq)
                    current_header = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line.upper())
            if current_header:
                self._fallback_db[current_header] = "".join(current_seq)
        logger.info(f"Fallback database loaded with {len(self._fallback_db)} contigs.")

    def reverse_complement(self, seq: str) -> str:
        """Generates the reverse complement of a DNA sequence.

        Args:
            seq: Nucleotide sequence.

        Returns:
            str: Reverse complement sequence.
        """
        complement = {
            "A": "T", "C": "G", "G": "C", "T": "A", "N": "N",
            "a": "t", "c": "g", "g": "c", "t": "a", "n": "n"
        }
        return "".join(complement.get(base, base) for base in reversed(seq))

    def check_specificity(self, primer_sequence: str, max_mismatches: int = 3) -> List[AlignmentHit]:
        """Scans the indexed reference genome/pangenome for potential primer hybridization sites.

        Args:
            primer_sequence: Primer sequence to query.
            max_mismatches: Maximum allowable mismatch threshold.

        Returns:
            List[AlignmentHit]: A list of alignment hits matching target criteria.
        """
        if not self.fasta_path:
            raise RuntimeError("Pangenome index is not loaded. Call index_pangenome first.")

        primer_seq_upper = primer_sequence.upper()

        if MAPPY_AVAILABLE and self.aligner is not None:
            hits = []
            try:
                for hit in self.aligner.map(primer_seq_upper):
                    # Filter out very poor alignments that exceed our mismatch limit
                    if hit.NM > max_mismatches:
                        continue
                    strand = 1 if hit.strand == 1 else -1
                    hits.append(
                        AlignmentHit(
                            contig=hit.ctg,
                            start=hit.r_st,
                            end=hit.r_en,
                            strand=strand,
                            mismatches=hit.NM,
                            is_primary=hit.is_primary,
                        )
                    )
                return hits
            except Exception as e:
                logger.error(f"mappy alignment failed: {e}. Running fallback alignment.")

        # Pure-Python sliding window fallback aligner
        return self._fallback_align(primer_seq_upper, max_mismatches)

    def _fallback_align(self, sequence: str, max_mismatches: int) -> List[AlignmentHit]:
        """Implements a sliding-window Hamming distance alignment fallback for Windows/systems without mappy."""
        hits: List[AlignmentHit] = []
        seq_len = len(sequence)
        rev_sequence = self.reverse_complement(sequence)

        for contig, ref_seq in self._fallback_db.items():
            ref_len = len(ref_seq)
            if ref_len < seq_len:
                continue

            # Slide window across reference sequence
            for i in range(ref_len - seq_len + 1):
                window = ref_seq[i : i + seq_len]
                
                # Check forward strand
                mismatches_f = sum(1 for a, b in zip(sequence, window) if a != b)
                if mismatches_f <= max_mismatches:
                    hits.append(
                        AlignmentHit(
                            contig=contig,
                            start=i,
                            end=i + seq_len,
                            strand=1,
                            mismatches=mismatches_f,
                            is_primary=True if not hits else False,
                        )
                    )

                # Check reverse strand
                mismatches_r = sum(1 for a, b in zip(rev_sequence, window) if a != b)
                if mismatches_r <= max_mismatches:
                    hits.append(
                        AlignmentHit(
                            contig=contig,
                            start=i,
                            end=i + seq_len,
                            strand=-1,
                            mismatches=mismatches_r,
                            is_primary=True if not hits else False,
                        )
                    )
        return hits


class VariantAwareFilter:
    """Parses genomic variation data (VCF) and flags/penalizes primers overlapping variable positions."""

    @dataclass(frozen=True)
    class Variant:
        """Data model for parsed VCF genomic variants."""

        chrom: str
        pos: int
        ref: str
        alt: str
        maf: float

    def __init__(self) -> None:
        """Initializes the VariantAwareFilter."""
        self.variants: List[VariantAwareFilter.Variant] = []

    def load_variants(self, vcf_path: str) -> None:
        """Parses a standard VCF (or mock tab-delimited VCF) file containing variants.

        Args:
            vcf_path: Path to the VCF file.
        """
        if not os.path.exists(vcf_path):
            raise FileNotFoundError(f"VCF file not found at: {vcf_path}")

        self.variants = []
        logger.info(f"Parsing variant file: {vcf_path}...")

        with open(vcf_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) < 5:
                    continue

                chrom = parts[0]
                try:
                    pos = int(parts[1])
                except ValueError:
                    continue
                ref = parts[3]
                alt = parts[4]

                # Extract Minor Allele Frequency (MAF / AF) from INFO field
                maf = 0.0
                if len(parts) >= 8:
                    info = parts[7]
                    # Look for standard AF or MAF fields: e.g. AF=0.05 or MAF=0.02
                    maf_match = re.search(r"\b(?:AF|MAF)=([0-9.]+)\b", info)
                    if maf_match:
                        try:
                            maf = float(maf_match.group(1))
                        except ValueError:
                            maf = 0.0

                self.variants.append(
                    VariantAwareFilter.Variant(
                        chrom=chrom, pos=pos, ref=ref, alt=alt, maf=maf
                    )
                )
        logger.info(f"Loaded {len(self.variants)} genomic variants into filter memory.")

    def evaluate_primer(
        self, primer_seq: str, start_pos: int, strand: int, maf_threshold: float = 0.01
    ) -> Tuple[float, bool]:
        """Evaluates a single primer sequence against loaded variants.

        Computes a variant penalty score and checks for variants in the critical 3' end.

        Args:
            primer_seq: Sequence of the primer.
            start_pos: Start genomic coordinate of the binding site on the reference.
            strand: Strand orientation (+1 for forward, -1 for reverse complement).
            maf_threshold: Minimum Minor Allele Frequency (MAF) to trigger penalty/rejection.

        Returns:
            Tuple[float, bool]: (penalty_score, is_valid)
                penalty_score: Combined thermodynamic/kinetic mismatch penalty.
                is_valid: False if a variant falls inside the critical 3' terminal 5 bp.
        """
        primer_len = len(primer_seq)
        end_pos = start_pos + primer_len

        penalty = 0.0
        is_valid = True

        for var in self.variants:
            # Check if variant falls within primer coordinates
            if start_pos <= var.pos < end_pos:
                if var.maf < maf_threshold:
                    continue

                # Determine relative distance to the critical 3' extension end
                # Forward primer extends from start to end (3' end is at the right/high coordinate)
                # Reverse primer extends from end to start (3' end is at the left/low coordinate)
                if strand == 1:
                    dist_to_3_prime = (end_pos - 1) - var.pos
                else:
                    dist_to_3_prime = var.pos - start_pos

                # 3' Anchor violation: Any polymorphism in the last 5bp is catastrophic
                if 0 <= dist_to_3_prime <= 5:
                    logger.warning(
                        f"Critical 3' anchor violation: Variant at pos {var.pos} (MAF={var.maf}) "
                        f"is {dist_to_3_prime}bp from the 3' end on strand {strand}."
                    )
                    is_valid = False
                    penalty += 100.0  # Apply maximum penalty
                else:
                    # Non-critical overlap: Apply scalar penalty depending on proximity to 3' end
                    proximity_weight = (primer_len - dist_to_3_prime) / primer_len
                    penalty += 20.0 * proximity_weight * var.maf

        return penalty, is_valid
