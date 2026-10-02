"""Locked feasibility-sample manifest validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image
import yaml

from .records import DatasetRecord


APPROVED_SAMPLE_IDS = (
    "NEON_D01_BART_DP3_312000_4875000_bidirectional_reflectance-468-532X660-724",
    "NEON_D01_HARV_DP3_725000_4701000_bidirectional_reflectance-404-468X788-852",
    "NEON_D06_KONZ_DP3_708000_4336000_bidirectional_reflectance-788-852X212-276",
    "NEON_D14_SRER_DP3_506000_3520000_bidirectional_reflectance-788-852X532-596",
    "NEON_D19_DEJU_DP3_566000_7088000_bidirectional_reflectance-532-596X724-788",
)


def sample_manifest_hash(sample_ids: tuple[str, ...] | list[str]) -> str:
    payload = "\n".join(sample_ids) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_fixed_samples(
    records: list[DatasetRecord], manifest_path: Path
) -> list[DatasetRecord]:
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    entries = manifest.get("samples") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise ValueError("sample manifest must contain a samples list")
    sample_ids = [entry.get("sample_id") for entry in entries]
    if sample_ids != list(APPROVED_SAMPLE_IDS):
        raise ValueError("sample manifest does not match approved order")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample manifest contains duplicate IDs")
    expected_hash = sample_manifest_hash(sample_ids)
    if manifest.get("manifest_sha256") != expected_hash:
        raise ValueError("sample manifest hash does not match its ordered IDs")

    by_id = {record.sample_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("dataset records contain duplicate IDs")
    selected: list[DatasetRecord] = []
    for sample_id in sample_ids:
        record = by_id.get(sample_id)
        if record is None:
            raise ValueError(f"missing approved sample: {sample_id}")
        if not record.description.strip():
            raise ValueError(f"approved sample has empty description: {sample_id}")
        try:
            with Image.open(record.rgb_path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"approved sample has unreadable RGB: {sample_id}") from exc
        selected.append(record)
    return selected
