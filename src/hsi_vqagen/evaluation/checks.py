"""Observable diagnostics that never invent subjective quality scores."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict


SPECTRAL_TERMS = {
    "ndvi", "ndwi", "nir", "swir", "red edge", "spectral signature",
    "식생지수", "정규식생지수", "근적외선", "단파적외선", "스펙트럼",
}
VISUAL_SOURCE_PHRASES = {
    "image shows", "visible in the image", "from the image", "in the rgb",
    "이미지에서", "영상에서 보", "rgb에서 보", "그림에서 보",
}


class EvaluationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: Literal["error", "warning"]
    message: str
    config_id: str | None = None
    sample_id: str | None = None
    pair_index: int | None = None


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    row_count: int
    configuration_count: int
    sample_count: int
    issues: tuple[EvaluationIssue, ...]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[\w가-힣]+", value.casefold()))


def _similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evaluate_deterministic(
    cases: Iterable[dict[str, Any]],
    *,
    require_configs: Iterable[str] | None = None,
    require_samples: Iterable[str] | None = None,
    require_pairs: int = 4,
) -> EvaluationSummary:
    rows = list(cases)
    issues: list[EvaluationIssue] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        config_id = str(row.get("config_id", ""))
        sample_id = str(row.get("sample_id", ""))
        pair_index = row.get("pair_index")
        grouped[(config_id, sample_id)].append(row)
        for field in ("question", "answer", "evidence", "question_type"):
            if not str(row.get(field, "")).strip():
                issues.append(EvaluationIssue(
                    code="empty_field", severity="error",
                    message=f"{field} is empty", config_id=config_id,
                    sample_id=sample_id, pair_index=pair_index,
                ))
        combined = " ".join(str(row.get(field, "")) for field in ("question", "answer", "evidence")).casefold()
        description = str(row.get("provenance", {}).get("description", "")).casefold()
        unsupported = sorted(term for term in SPECTRAL_TERMS if term in combined and term not in description)
        if unsupported:
            issues.append(EvaluationIssue(
                code="unsupported_spectral_claim", severity="warning",
                message="spectral terms absent from description: " + ", ".join(unsupported),
                config_id=config_id, sample_id=sample_id, pair_index=pair_index,
            ))
        if row.get("input_condition") == "text_only" and any(
            phrase in combined for phrase in VISUAL_SOURCE_PHRASES
        ):
            issues.append(EvaluationIssue(
                code="text_only_visual_source_leakage", severity="warning",
                message="text-only output claims direct visual access",
                config_id=config_id, sample_id=sample_id, pair_index=pair_index,
            ))

    for (config_id, sample_id), group in grouped.items():
        if len(group) != require_pairs or {row.get("pair_index") for row in group} != set(range(1, require_pairs + 1)):
            issues.append(EvaluationIssue(
                code="pair_count_mismatch", severity="error",
                message=f"expected pair indices 1..{require_pairs}",
                config_id=config_id, sample_id=sample_id,
            ))
        for left_index, left in enumerate(group):
            for right in group[left_index + 1:]:
                left_q, right_q = str(left.get("question", "")), str(right.get("question", ""))
                if left_q.strip() and left_q.strip().casefold() == right_q.strip().casefold():
                    issues.append(EvaluationIssue(
                        code="duplicate_question", severity="warning",
                        message="two questions are identical", config_id=config_id,
                        sample_id=sample_id, pair_index=right.get("pair_index"),
                    ))
                if _similarity(left_q, right_q) >= 0.8:
                    issues.append(EvaluationIssue(
                        code="paraphrase_question", severity="warning",
                        message="two questions have high token overlap", config_id=config_id,
                        sample_id=sample_id, pair_index=right.get("pair_index"),
                    ))

    configs = list(require_configs) if require_configs is not None else sorted({key[0] for key in grouped})
    samples = list(require_samples) if require_samples is not None else sorted({key[1] for key in grouped})
    for config_id in configs:
        for sample_id in samples:
            if (config_id, sample_id) not in grouped:
                issues.append(EvaluationIssue(
                    code="missing_matrix_cell", severity="error",
                    message="required configuration/sample cell is missing",
                    config_id=config_id, sample_id=sample_id,
                ))
    return EvaluationSummary(
        row_count=len(rows),
        configuration_count=len({row.get("config_id") for row in rows}),
        sample_count=len({row.get("sample_id") for row in rows}),
        issues=tuple(issues),
    )


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/normalized.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--require-configs", nargs="*")
    parser.add_argument("--require-samples", nargs="*")
    parser.add_argument("--require-pairs", type=int, default=4)
    args = parser.parse_args()
    rows = _load_rows(args.outputs)
    required_configs = (
        [item for value in args.require_configs for item in value.split(",") if item]
        if args.require_configs else None
    )
    required_samples = args.require_samples
    count_error = None
    if required_samples and len(required_samples) == 1 and required_samples[0].isdigit():
        expected_count = int(required_samples[0])
        observed = sorted({str(row.get("sample_id")) for row in rows})
        if len(observed) != expected_count:
            count_error = f"expected {expected_count} unique samples, found {len(observed)}"
        required_samples = observed
    summary = evaluate_deterministic(
        rows, require_configs=required_configs,
        require_samples=required_samples, require_pairs=args.require_pairs,
    )
    print(summary.model_dump_json(indent=2))
    if count_error:
        print(count_error)
    raise SystemExit(0 if summary.passed and count_error is None else 1)


if __name__ == "__main__":
    main()
