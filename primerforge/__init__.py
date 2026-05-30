"""PrimerForge: A Hybrid Thermodynamic and Machine Learning Platform for Pangenome-Aware PCR Primer Design.

Exposes the primary biophysical engine, machine learning scorer, pangenome specificity checks,
and integer linear programming optimizer.
"""

__version__ = "0.1.0"
__author__ = "PrimerForge Contributors"

from primerforge.biophysics import BiophysicsEngine, PrimerPair, PrimerSequence
from primerforge.specificity import SpecificityEngine, AlignmentHit, VariantAwareFilter
from primerforge.ml_scorer import MLScorer
from primerforge.optimizer import MultiplexOptimizer, TiledAmpliconRouter
from primerforge.data_curation import DataCurationPipeline

__all__ = [
    "BiophysicsEngine",
    "PrimerPair",
    "PrimerSequence",
    "SpecificityEngine",
    "AlignmentHit",
    "VariantAwareFilter",
    "MLScorer",
    "MultiplexOptimizer",
    "TiledAmpliconRouter",
    "DataCurationPipeline",
]

