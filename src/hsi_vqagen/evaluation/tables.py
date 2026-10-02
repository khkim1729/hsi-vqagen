"""Deterministic blind worksheet construction for human review."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import pandas as pd


RATING_COLUMNS = (
    "correctness", "grounding", "hallucination", "diversity",
    "visual_dependence", "usefulness", "specificity",
)


def build_manual_review_sheet(cases: Iterable[dict[str, Any]]) -> pd.DataFrame:
    prepared = []
    for row in cases:
        identity = f"{row.get('config_id')}:{row.get('sample_id')}:{row.get('pair_index')}"
        prepared.append({
            "blind_id": hashlib.sha256(identity.encode()).hexdigest()[:12],
            "sample_id": row.get("sample_id"),
            "pair_index": row.get("pair_index"),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "evidence": row.get("evidence", ""),
            **{column: "" for column in RATING_COLUMNS},
            "reviewer_notes": "",
        })
    return pd.DataFrame(prepared).sort_values("blind_id", ignore_index=True)
