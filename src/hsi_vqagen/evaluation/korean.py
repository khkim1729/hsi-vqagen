"""Reproducible Korean-language and source-grounding proxy metrics."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from hsi_vqagen.evaluation.checks import SPECTRAL_TERMS, VISUAL_SOURCE_PHRASES


HANGUL_RE = re.compile(r"[가-힣]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


class KoreanMetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_id: str
    case_count: int
    pair_count: int
    korean_complete_pair_rate: float = Field(ge=0, le=1)
    mean_source_description_cosine: float = Field(ge=0, le=1)
    mean_question_diversity: float = Field(ge=0, le=1)
    direct_visual_access_warning_count: int = Field(ge=0)
    unsupported_spectral_claim_count: int = Field(ge=0)
    mean_answer_characters: float = Field(ge=0)
    mean_latency_seconds: float = Field(ge=0)


def has_korean_text(value: str) -> bool:
    return bool(HANGUL_RE.search(value))


def _is_korean_task_answer(value: str) -> bool:
    """Accept Korean prose or a language-neutral numeric/symbolic short answer."""

    return has_korean_text(value) or (
        bool(re.search(r"\d", value)) and not bool(re.search(r"[A-Za-z]", value))
    )


def _character_ngrams(value: str, minimum: int = 2, maximum: int = 4) -> Counter[str]:
    normalized = re.sub(r"[^0-9A-Za-z가-힣]", "", value.casefold())
    grams: Counter[str] = Counter()
    for size in range(minimum, maximum + 1):
        grams.update(normalized[index : index + size] for index in range(len(normalized) - size + 1))
    return grams


def _tfidf_vectors(documents: list[str]) -> list[dict[str, float]]:
    counts = [_character_ngrams(document) for document in documents]
    document_frequency: Counter[str] = Counter()
    for count in counts:
        document_frequency.update(count.keys())
    size = len(documents)
    vectors = []
    for count in counts:
        total = sum(count.values()) or 1
        vectors.append({
            term: (frequency / total) * (math.log((1 + size) / (1 + document_frequency[term])) + 1)
            for term, frequency in count.items()
        })
    return vectors


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def source_description_cosine(candidate: str, description: str) -> float:
    """Maximum char 2--4 gram TF-IDF cosine against a source sentence."""

    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(description) if part.strip()]
    if not sentences:
        sentences = [description]
    vectors = _tfidf_vectors([candidate, *sentences])
    return max((_cosine(vectors[0], vector) for vector in vectors[1:]), default=0.0)


def _question_diversity(questions: list[str]) -> float:
    similarities = []
    for left, right in combinations(questions, 2):
        vectors = _tfidf_vectors([left, right])
        similarities.append(_cosine(vectors[0], vectors[1]))
    return 1.0 - (sum(similarities) / len(similarities)) if similarities else 0.0


def evaluate_korean_outputs(rows: Iterable[dict[str, Any]]) -> list[KoreanMetricSummary]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)

    summaries = []
    for config_id, config_rows in sorted(grouped.items()):
        complete = sum(
            has_korean_text(str(row.get("question", "")))
            and _is_korean_task_answer(str(row.get("answer", "")))
            and has_korean_text(str(row.get("evidence", "")))
            for row in config_rows
        )
        grounding = [
            source_description_cosine(
                f"{row.get('answer', '')} {row.get('evidence', '')}",
                str(row.get("provenance", {}).get("description", "")),
            )
            for row in config_rows
        ]
        by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in config_rows:
            by_case[str(row["sample_id"])].append(row)
        diversity = [
            _question_diversity([str(row.get("question", "")) for row in case_rows])
            for case_rows in by_case.values()
        ]
        visual_warnings = sum(
            row.get("input_condition") == "text_only"
            and any(
                phrase in " ".join(str(row.get(field, "")) for field in ("question", "answer", "evidence")).casefold()
                for phrase in VISUAL_SOURCE_PHRASES
            )
            for row in config_rows
        )
        spectral_warnings = 0
        for row in config_rows:
            combined = " ".join(str(row.get(field, "")) for field in ("question", "answer", "evidence")).casefold()
            description = str(row.get("provenance", {}).get("description", "")).casefold()
            spectral_warnings += int(any(term in combined and term not in description for term in SPECTRAL_TERMS))
        case_latencies = [float(case_rows[0].get("latency_seconds", 0.0)) for case_rows in by_case.values()]
        summaries.append(KoreanMetricSummary(
            config_id=config_id,
            case_count=len(by_case),
            pair_count=len(config_rows),
            korean_complete_pair_rate=complete / len(config_rows) if config_rows else 0.0,
            mean_source_description_cosine=sum(grounding) / len(grounding) if grounding else 0.0,
            mean_question_diversity=sum(diversity) / len(diversity) if diversity else 0.0,
            direct_visual_access_warning_count=visual_warnings,
            unsupported_spectral_claim_count=spectral_warnings,
            mean_answer_characters=(
                sum(len(str(row.get("answer", ""))) for row in config_rows) / len(config_rows)
                if config_rows else 0.0
            ),
            mean_latency_seconds=sum(case_latencies) / len(case_latencies) if case_latencies else 0.0,
        ))
    return summaries
