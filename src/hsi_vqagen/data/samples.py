"""Locked feasibility-sample manifest validation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
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

PAPER50_CONTENT_SHA256 = "fb2764269685ac84347f43c15823bcf3565351034ea8ea3a2b0aa0d41922791f"
ROBUSTNESS20_CONTENT_SHA256 = "5c2ef2dc06c0f5cc3c929ac9a448c42f1a8a9b633e4814b9ae19a80d95442ce0"
FROZEN_STUDY_CONTENT = {
    "hsi-vqagen-icce2027-paper50-v1": PAPER50_CONTENT_SHA256,
    "hsi-vqagen-icce2027-rgb-robustness-v1": ROBUSTNESS20_CONTENT_SHA256,
}


def study_content_hash(entries: list[dict]) -> str:
    fields = ("sample_id", "shard", "description_sha256", "rgb_sha256")
    rows = [{key: entry.get(key) for key in fields} for entry in entries]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _site(sample_id: str) -> str:
    parts = sample_id.split("_")
    if len(parts) < 4 or parts[0] != "NEON":
        raise ValueError(f"sample ID does not identify a NEON site: {sample_id}")
    return parts[2]


def select_stratified_records(
    records: list[DatasetRecord], *, count: int, seed: str = "icce-2027-v1"
) -> list[DatasetRecord]:
    """Select reproducible site-by-description-length strata, favoring distinct shards."""

    if count <= 0 or count > len(records):
        raise ValueError("sample count must be between one and dataset size")
    if len({row.sample_id for row in records}) != len(records):
        raise ValueError("dataset records contain duplicate sample IDs")
    lengths = sorted(len(row.description) for row in records)
    lower, upper = lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]
    strata: dict[tuple[str, int], list[DatasetRecord]] = defaultdict(list)
    for row in records:
        length = len(row.description)
        band = 0 if length < lower else 1 if length < upper else 2
        strata[(_site(row.sample_id), band)].append(row)
    for group in strata.values():
        group.sort(key=lambda row: (
            hashlib.sha256(f"{seed}:{row.sample_id}".encode()).hexdigest(), row.sample_id
        ))

    selected: list[DatasetRecord] = []
    used_shards: set[str] = set()
    sites = sorted({site for site, _ in strata})
    round_index = 0
    while len(selected) < count:
        progressed = False
        for site_index, site in enumerate(sites):
            if len(selected) == count:
                break
            band_order = [(site_index + round_index + offset) % 3 for offset in range(3)]
            available = [strata[(site, band)] for band in band_order if strata.get((site, band))]
            if not available:
                continue
            distinct = next(
                ((group, index) for group in available for index, row in enumerate(group)
                 if row.shard not in used_shards),
                None,
            )
            group, candidate_index = distinct if distinct is not None else (available[0], 0)
            row = group.pop(candidate_index)
            selected.append(row)
            used_shards.add(row.shard)
            progressed = True
        if not progressed:
            raise ValueError("not enough records to fill requested cohort")
        round_index += 1
    return selected


def load_study_samples(
    records: list[DatasetRecord],
    manifest_path: Path,
    *,
    expected_count: int,
    expected_hash: str,
) -> list[DatasetRecord]:
    """Load an independently frozen paper cohort without changing the five-sample gate."""

    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    entries = manifest.get("samples") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError("study manifest sample count mismatch")
    sample_ids = [entry.get("sample_id") for entry in entries]
    if any(not isinstance(sample_id, str) for sample_id in sample_ids):
        raise ValueError("study manifest has an invalid sample ID")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("study manifest contains duplicate sample IDs")
    actual_hash = sample_manifest_hash(sample_ids)
    if actual_hash != expected_hash or manifest.get("manifest_sha256") != expected_hash:
        raise ValueError("study manifest hash or order differs from frozen cohort")
    content_hash = study_content_hash(entries)
    if (manifest.get("content_sha256") != content_hash
            or (manifest.get("experiment_id") in FROZEN_STUDY_CONTENT
                and content_hash != FROZEN_STUDY_CONTENT[manifest["experiment_id"]])):
        raise ValueError("study manifest content digest commitment mismatch")
    by_id = {row.sample_id: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("dataset records contain duplicate sample IDs")
    selected = []
    for entry, sample_id in zip(entries, sample_ids):
        row = by_id.get(sample_id)
        if row is None or not row.description.strip():
            raise ValueError(f"missing or empty study sample: {sample_id}")
        if entry.get("shard") is not None and entry["shard"] != row.shard:
            raise ValueError(f"study sample shard mismatch: {sample_id}")
        description_digest = hashlib.sha256(row.description.encode("utf-8")).hexdigest()
        if entry.get("description_sha256") != description_digest:
            raise ValueError(f"study sample description digest mismatch: {sample_id}")
        try:
            rgb_digest = hashlib.sha256(row.rgb_path.read_bytes()).hexdigest()
            with Image.open(row.rgb_path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"unreadable study RGB: {sample_id}") from exc
        if entry.get("rgb_sha256") != rgb_digest:
            raise ValueError(f"study sample RGB digest mismatch: {sample_id}")
        selected.append(row)
    return selected


def rgb_mapping_hash(pairs: list[tuple[str, str]]) -> str:
    payload = "\n".join(f"{sample_id}->{image_id}" for sample_id, image_id in pairs) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_rgb_swap_manifest(
    records: list[DatasetRecord],
    manifest_path: Path,
    *,
    expected_count: int,
    expected_sample_hash: str,
    expected_mapping_hash: str,
) -> tuple[list[DatasetRecord], dict[str, DatasetRecord]]:
    """Validate an immutable within-site image derangement for a robustness run."""

    selected = load_study_samples(
        records, manifest_path,
        expected_count=expected_count,
        expected_hash=expected_sample_hash,
    )
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    pairs = [
        (entry["sample_id"], entry.get("shuffled_rgb_sample_id"))
        for entry in manifest["samples"]
    ]
    if (
        any(not isinstance(image_id, str) for _, image_id in pairs)
        or rgb_mapping_hash(pairs) != expected_mapping_hash
        or manifest.get("image_mapping_sha256") != expected_mapping_hash
    ):
        raise ValueError("RGB image mapping hash differs from frozen mapping")
    by_id = {row.sample_id: row for row in selected}
    if set(image_id for _, image_id in pairs) != set(by_id):
        raise ValueError("RGB image mapping must be a cohort derangement")
    if any(sample_id == image_id or _site(sample_id) != _site(image_id) for sample_id, image_id in pairs):
        raise ValueError("RGB image mapping must swap distinct scenes within each site")
    return selected, {sample_id: by_id[image_id] for sample_id, image_id in pairs}
