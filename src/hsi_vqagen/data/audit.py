"""Dataset-wide validation and summary statistics."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from .records import DatasetIssue, scan_dataset


class LengthStatistics(BaseModel):
    minimum: int = 0
    median: float = 0
    mean: float = 0
    p95: int = 0
    maximum: int = 0


class DatasetAudit(BaseModel):
    shard_count: int
    total_description_rows: int
    total_sft_rows: int
    unique_sample_count: int
    valid_sample_count: int
    fully_paired_count: int
    duplicate_sample_ids: list[str] = Field(default_factory=list)
    missing_description_count: int = 0
    missing_sft_count: int = 0
    missing_rgb_count: int = 0
    missing_meta_count: int = 0
    unreadable_rgb_count: int = 0
    malformed_json_count: int = 0
    description_disagreement_count: int = 0
    image_resolutions: dict[str, int] = Field(default_factory=dict)
    hsi_shapes: dict[str, int] = Field(default_factory=dict)
    description_lengths: LengthStatistics = Field(default_factory=LengthStatistics)
    file_formats: dict[str, int] = Field(default_factory=dict)
    errors: list[DatasetIssue] = Field(default_factory=list)


def _length_statistics(values: list[int]) -> LengthStatistics:
    if not values:
        return LengthStatistics()
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return LengthStatistics(
        minimum=ordered[0],
        median=statistics.median(ordered),
        mean=round(statistics.fmean(ordered), 4),
        p95=ordered[p95_index],
        maximum=ordered[-1],
    )


def _file_formats(root: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in root.rglob("*"):
        if path.is_file():
            counts[path.suffix.lower() or "<no_suffix>"] += 1
    return dict(sorted(counts.items()))


def audit_dataset(root: Path) -> DatasetAudit:
    root = Path(root)
    scan = scan_dataset(root)
    issue_counts = Counter(issue.code for issue in scan.errors)
    return DatasetAudit(
        shard_count=len(scan.shard_names),
        total_description_rows=scan.total_description_rows,
        total_sft_rows=scan.total_sft_rows,
        unique_sample_count=len(scan.sample_ids),
        valid_sample_count=len(scan.records),
        fully_paired_count=len(scan.records),
        duplicate_sample_ids=sorted(scan.duplicate_sample_ids),
        missing_description_count=issue_counts["missing_description"],
        missing_sft_count=issue_counts["missing_sft"],
        missing_rgb_count=issue_counts["missing_rgb"],
        missing_meta_count=issue_counts["missing_meta"],
        unreadable_rgb_count=issue_counts["unreadable_rgb"],
        malformed_json_count=issue_counts["malformed_json"],
        description_disagreement_count=issue_counts["description_disagreement"],
        image_resolutions=dict(sorted(scan.image_resolutions.items())),
        hsi_shapes=dict(sorted(scan.hsi_shapes.items())),
        description_lengths=_length_statistics(scan.description_lengths),
        file_formats=_file_formats(root),
        errors=scan.errors,
    )
