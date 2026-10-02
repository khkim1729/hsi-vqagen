"""Deterministic diagnostics and human-review preparation."""

from .ablation import compare_modality_ablation
from .checks import evaluate_deterministic
from .tables import build_manual_review_sheet

__all__ = [
    "build_manual_review_sheet",
    "compare_modality_ablation",
    "evaluate_deterministic",
]
