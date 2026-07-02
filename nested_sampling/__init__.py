"""Nested sampling and evidence calculation module"""

from .evidence_calculator import EvidenceCalculator, compute_bayes_factor

__all__ = ["EvidenceCalculator", "compute_bayes_factor"]
