"""Exact alignment for weight-controlled modality comparisons."""

from __future__ import annotations

import re
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict


REGISTERED_PAIRS = {"C01": "C10", "C02": "C11", "C03": "C09", "C04": "C06"}


class AblationComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    pair_index: int
    multimodal_question: str
    text_only_question: str
    multimodal_answer: str
    text_only_answer: str
    question_token_jaccard: float
    answer_length_delta: int
    description_absent_visual_detail: str = ""
    paraphrase_tendency: str = ""
    hallucination_direction: str = ""
    diversity_change: str = ""
    specificity_change: str = ""


class AblationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    multimodal_config_id: str
    text_only_config_id: str
    comparisons: tuple[AblationComparison, ...]


def _config_id(rows: list[dict[str, Any]]) -> str:
    values = {str(row.get("config_id")) for row in rows}
    if len(values) != 1:
        raise ValueError("each ablation side must contain one configuration")
    return values.pop()


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[\w가-힣]+", value.casefold()))


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def compare_modality_ablation(
    multimodal: Iterable[dict[str, Any]], text_only: Iterable[dict[str, Any]]
) -> AblationReport:
    multimodal_rows, text_rows = list(multimodal), list(text_only)
    multimodal_id, text_id = _config_id(multimodal_rows), _config_id(text_rows)
    if REGISTERED_PAIRS.get(multimodal_id) != text_id:
        raise ValueError("inputs are not a registered modality pair")
    mm = {(str(row["sample_id"]), int(row["pair_index"])): row for row in multimodal_rows}
    tx = {(str(row["sample_id"]), int(row["pair_index"])): row for row in text_rows}
    if set(mm) != set(tx) or len(mm) != len(multimodal_rows) or len(tx) != len(text_rows):
        raise ValueError("ablation pairing mismatch by sample_id/pair_index")
    comparisons = []
    for key in sorted(mm):
        left, right = mm[key], tx[key]
        left_q, right_q = str(left["question"]), str(right["question"])
        left_a, right_a = str(left["answer"]), str(right["answer"])
        comparisons.append(AblationComparison(
            sample_id=key[0], pair_index=key[1],
            multimodal_question=left_q, text_only_question=right_q,
            multimodal_answer=left_a, text_only_answer=right_a,
            question_token_jaccard=_jaccard(left_q, right_q),
            answer_length_delta=len(left_a) - len(right_a),
        ))
    return AblationReport(
        multimodal_config_id=multimodal_id,
        text_only_config_id=text_id,
        comparisons=tuple(comparisons),
    )
