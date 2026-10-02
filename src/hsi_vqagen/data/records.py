"""Typed loading of the realized description/SFT shard format."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from PIL import Image
from pydantic import BaseModel, ConfigDict


class DatasetRecord(BaseModel):
    """One fully paired description and HSI-derived RGB record."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    shard: str
    description: str
    rgb_path: Path
    meta_path: Path
    hsi_shape: tuple[int, int, int]
    rgb_size: tuple[int, int]
    prompt_version: str | None = None
    prompt_hash: str | None = None


class DatasetIssue(BaseModel):
    code: str
    message: str
    shard: str | None = None
    sample_id: str | None = None
    path: str | None = None
    line: int | None = None


@dataclass
class DatasetScan:
    shard_names: list[str] = field(default_factory=list)
    records: list[DatasetRecord] = field(default_factory=list)
    errors: list[DatasetIssue] = field(default_factory=list)
    total_description_rows: int = 0
    total_sft_rows: int = 0
    sample_ids: set[str] = field(default_factory=set)
    duplicate_sample_ids: set[str] = field(default_factory=set)
    description_lengths: list[int] = field(default_factory=list)
    image_resolutions: Counter[str] = field(default_factory=Counter)
    hsi_shapes: Counter[str] = field(default_factory=Counter)


def _read_jsonl(
    path: Path, shard: str, kind: str, scan: DatasetScan
) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        scan.errors.append(
            DatasetIssue(
                code=f"missing_{kind}_file",
                message=f"required {kind} JSONL is missing",
                shard=shard,
                path=str(path),
            )
        )
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                scan.errors.append(
                    DatasetIssue(
                        code="malformed_json",
                        message=f"{kind} JSON decode error: {exc.msg}",
                        shard=shard,
                        path=str(path),
                        line=line_number,
                    )
                )
                continue
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
                scan.errors.append(
                    DatasetIssue(
                        code="invalid_row",
                        message=f"{kind} row lacks a string sample_id",
                        shard=shard,
                        path=str(path),
                        line=line_number,
                    )
                )
                continue
            yield row


def _sample_issue(
    scan: DatasetScan,
    code: str,
    message: str,
    shard: str,
    sample_id: str,
    path: Path | None = None,
) -> None:
    scan.errors.append(
        DatasetIssue(
            code=code,
            message=message,
            shard=shard,
            sample_id=sample_id,
            path=str(path) if path else None,
        )
    )


def scan_dataset(root: Path) -> DatasetScan:
    """Scan shards without mutating them, retaining every detected issue."""

    root = Path(root)
    scan = DatasetScan()
    candidate_records: list[DatasetRecord] = []
    seen_locations: dict[str, str] = {}

    shards = sorted(
        path
        for path in root.glob("s*")
        if path.is_dir()
        and ((path / "descriptions.jsonl").exists() or (path / "sft.jsonl").exists())
    )
    scan.shard_names = [path.name for path in shards]

    for shard_path in shards:
        shard = shard_path.name
        description_rows = list(
            _read_jsonl(shard_path / "descriptions.jsonl", shard, "description", scan)
        )
        sft_rows = list(_read_jsonl(shard_path / "sft.jsonl", shard, "sft", scan))
        scan.total_description_rows += len(description_rows)
        scan.total_sft_rows += len(sft_rows)

        descriptions: dict[str, dict[str, Any]] = {}
        sft: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for row in description_rows:
            sample_id = row["sample_id"]
            if sample_id in descriptions:
                scan.duplicate_sample_ids.add(sample_id)
            else:
                descriptions[sample_id] = row
                order.append(sample_id)
            value = row.get("final_description")
            if isinstance(value, str):
                scan.description_lengths.append(len(value))
        for row in sft_rows:
            sample_id = row["sample_id"]
            if sample_id in sft:
                scan.duplicate_sample_ids.add(sample_id)
            else:
                sft[sample_id] = row
            if sample_id not in descriptions:
                order.append(sample_id)

        for sample_id in order:
            scan.sample_ids.add(sample_id)
            if sample_id in seen_locations:
                scan.duplicate_sample_ids.add(sample_id)
                _sample_issue(
                    scan,
                    "duplicate_sample_id",
                    f"sample also appears in {seen_locations[sample_id]}",
                    shard,
                    sample_id,
                )
                continue
            seen_locations[sample_id] = shard
            description_row = descriptions.get(sample_id)
            sft_row = sft.get(sample_id)
            invalid = False
            if description_row is None:
                _sample_issue(
                    scan, "missing_description", "no descriptions.jsonl row", shard, sample_id
                )
                invalid = True
            if sft_row is None:
                _sample_issue(scan, "missing_sft", "no sft.jsonl row", shard, sample_id)
                invalid = True
            if invalid:
                continue

            description = description_row.get("final_description")
            if not isinstance(description, str) or not description.strip():
                _sample_issue(
                    scan, "missing_description", "description is empty", shard, sample_id
                )
                invalid = True
                description = ""
            sft_description = (
                sft_row.get("structured_answer", {}).get("final_description")
                if isinstance(sft_row.get("structured_answer"), dict)
                else None
            )
            if sft_description != description:
                _sample_issue(
                    scan,
                    "description_disagreement",
                    "descriptions.jsonl and sft.jsonl text differ",
                    shard,
                    sample_id,
                )
                invalid = True

            rgb_value = (
                sft_row.get("images", {}).get("rgb")
                if isinstance(sft_row.get("images"), dict)
                else None
            )
            rgb_path = Path(rgb_value) if isinstance(rgb_value, str) else Path()
            if isinstance(rgb_value, str) and not rgb_path.is_absolute():
                rgb_path = shard_path / rgb_path
            rgb_size: tuple[int, int] | None = None
            if not isinstance(rgb_value, str) or not rgb_path.is_file():
                _sample_issue(
                    scan, "missing_rgb", "referenced RGB file is missing", shard, sample_id, rgb_path
                )
                invalid = True
            else:
                try:
                    with Image.open(rgb_path) as image:
                        image.verify()
                    with Image.open(rgb_path) as image:
                        rgb_size = image.size
                    scan.image_resolutions[f"{rgb_size[0]}x{rgb_size[1]}"] += 1
                except Exception as exc:  # Pillow exposes multiple corruption errors.
                    _sample_issue(
                        scan, "unreadable_rgb", f"cannot read RGB: {exc}", shard, sample_id, rgb_path
                    )
                    invalid = True

            meta_path = shard_path / "patches" / sample_id / "meta.json"
            hsi_shape: tuple[int, int, int] | None = None
            if not meta_path.is_file():
                _sample_issue(
                    scan, "missing_meta", "patch metadata is missing", shard, sample_id, meta_path
                )
                invalid = True
            else:
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    shape = meta.get("quality", {}).get("cube_shape")
                    if not (
                        isinstance(shape, list)
                        and len(shape) == 3
                        and all(isinstance(value, int) for value in shape)
                    ):
                        raise ValueError("quality.cube_shape is not a three-integer list")
                    hsi_shape = tuple(shape)
                    scan.hsi_shapes["x".join(map(str, hsi_shape))] += 1
                except (json.JSONDecodeError, ValueError, AttributeError) as exc:
                    _sample_issue(
                        scan, "invalid_meta", f"cannot read HSI shape: {exc}", shard, sample_id, meta_path
                    )
                    invalid = True

            if invalid or rgb_size is None or hsi_shape is None:
                continue
            prompt_set = description_row.get("prompt_set", {})
            candidate_records.append(
                DatasetRecord(
                    sample_id=sample_id,
                    shard=shard,
                    description=description,
                    rgb_path=rgb_path,
                    meta_path=meta_path,
                    hsi_shape=hsi_shape,
                    rgb_size=rgb_size,
                    prompt_version=prompt_set.get("version"),
                    prompt_hash=prompt_set.get("hash"),
                )
            )

    for sample_id in sorted(scan.duplicate_sample_ids):
        if not any(
            issue.code == "duplicate_sample_id" and issue.sample_id == sample_id
            for issue in scan.errors
        ):
            _sample_issue(
                scan,
                "duplicate_sample_id",
                "sample ID is repeated within a shard file",
                seen_locations.get(sample_id, "unknown"),
                sample_id,
            )
    scan.records = [
        record
        for record in candidate_records
        if record.sample_id not in scan.duplicate_sample_ids
    ]
    return scan


def load_dataset_records(root: Path) -> list[DatasetRecord]:
    """Return only complete, consistent, uniquely identified records."""

    return scan_dataset(root).records
