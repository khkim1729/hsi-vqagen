import hashlib
from pathlib import Path

import pytest
import yaml
from PIL import Image

from hsi_vqagen.data.records import DatasetRecord
from hsi_vqagen.data import samples as paper_samples


def _records(tmp_path: Path) -> list[DatasetRecord]:
    records = []
    for site in ("BART", "HARV", "KONZ"):
        for index in range(12):
            sample_id = f"NEON_D01_{site}_DP3_{index:04d}"
            rgb = tmp_path / f"{sample_id}.png"
            Image.new("RGB", (8, 8), color="green").save(rgb)
            records.append(DatasetRecord(
                sample_id=sample_id,
                shard=f"s{len(records):04d}",
                description="가" * (100 + (index % 3) * 300),
                rgb_path=rgb,
                meta_path=tmp_path / "meta.json",
                hsi_shape=(64, 64, 426),
                rgb_size=(384, 384),
            ))
    return records


def test_stratified_selection_is_order_independent_and_covers_sites_and_lengths(tmp_path: Path) -> None:
    """Catches sampling that changes with shard traversal or misses a site/length band."""
    records = _records(tmp_path)
    selected = paper_samples.select_stratified_records(records, count=18)
    reversed_selected = paper_samples.select_stratified_records(list(reversed(records)), count=18)

    assert [row.sample_id for row in selected] == [row.sample_id for row in reversed_selected]
    assert len({row.shard for row in selected}) == 18
    assert {row.sample_id.split("_")[2] for row in selected} == {"BART", "HARV", "KONZ"}
    assert {len(row.description) for row in selected} == {100, 400, 700}


def test_small_cohort_reaches_each_available_site_before_repeating(tmp_path: Path) -> None:
    """Catches site-by-band iteration spending the whole budget on early sites."""
    selected = paper_samples.select_stratified_records(_records(tmp_path), count=3)
    assert {row.sample_id.split("_")[2] for row in selected} == {
        "BART", "HARV", "KONZ"
    }


def test_study_manifest_rejects_order_change_even_with_recomputed_internal_hash(tmp_path: Path) -> None:
    """Catches a paper cohort silently changing after its expected hash is frozen."""
    records = _records(tmp_path)
    ids = [records[0].sample_id, records[1].sample_id]
    expected_hash = "f516fd0e08998744da3f717e0cf2fcf90c0e800742c7d9051210941d8d64efd5"
    manifest = tmp_path / "samples.yaml"
    payload = {
        "manifest_sha256": expected_hash,
        "samples": [{
            "sample_id": row.sample_id,
            "description_sha256": hashlib.sha256(row.description.encode()).hexdigest(),
            "rgb_sha256": hashlib.sha256(row.rgb_path.read_bytes()).hexdigest(),
        } for row in records[:2]],
    }
    payload["content_sha256"] = paper_samples.study_content_hash(payload["samples"])
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")

    assert [row.sample_id for row in paper_samples.load_study_samples(
        records, manifest, expected_count=2, expected_hash=expected_hash
    )] == ids

    changed = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    changed["samples"] = list(reversed(changed["samples"]))
    manifest.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="hash|order"):
        paper_samples.load_study_samples(records, manifest, expected_count=2, expected_hash=expected_hash)


def test_frozen_paper_manifest_covers_all_available_sites_and_excludes_pilot() -> None:
    """Catches accidental narrowing or leakage from the original pilot cohort."""
    payload = yaml.safe_load(Path("configs/paper_samples_50.yaml").read_text(encoding="utf-8"))
    entries = payload["samples"]
    ids = [entry["sample_id"] for entry in entries]

    assert len(ids) == len(set(ids)) == 50
    assert len({sample_id.split("_")[2] for sample_id in ids}) == 23
    assert not set(ids) & set(paper_samples.APPROVED_SAMPLE_IDS)
    assert len({entry["shard"] for entry in entries}) >= 35
    assert paper_samples.sample_manifest_hash(ids) == (
        "40d3f07e006cd255dee4245fec224b5da16ed958f8dc47b458d83048524a8ec2"
    )
    assert payload["manifest_sha256"] == paper_samples.sample_manifest_hash(ids)


def test_study_manifest_pins_description_and_rgb_bytes(tmp_path: Path) -> None:
    records = _records(tmp_path)
    row = records[0]
    manifest = tmp_path / "pinned.yaml"
    payload = {
        "manifest_sha256": paper_samples.sample_manifest_hash([row.sample_id]),
        "samples": [{
            "sample_id": row.sample_id,
            "description_sha256": hashlib.sha256(row.description.encode()).hexdigest(),
            "rgb_sha256": hashlib.sha256(row.rgb_path.read_bytes()).hexdigest(),
        }],
    }
    payload["content_sha256"] = paper_samples.study_content_hash(payload["samples"])
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    paper_samples.load_study_samples(records, manifest, expected_count=1,
                                     expected_hash=paper_samples.sample_manifest_hash([row.sample_id]))
    changed = row.model_copy(update={"description": row.description + "changed"})
    with pytest.raises(ValueError, match="description digest"):
        paper_samples.load_study_samples([changed], manifest, expected_count=1,
                                         expected_hash=paper_samples.sample_manifest_hash([row.sample_id]))
    Image.new("RGB", (8, 8), color="red").save(row.rgb_path)
    with pytest.raises(ValueError, match="RGB digest"):
        paper_samples.load_study_samples(records, manifest, expected_count=1,
                                         expected_hash=paper_samples.sample_manifest_hash([row.sample_id]))


def test_frozen_content_commitment_cannot_be_rewritten_with_same_ids() -> None:
    manifest = yaml.safe_load(Path("configs/paper_samples_50.yaml").read_text())
    assert manifest["content_sha256"] == paper_samples.PAPER50_CONTENT_SHA256
    assert paper_samples.study_content_hash(manifest["samples"]) == manifest["content_sha256"]
    altered = [dict(entry) for entry in manifest["samples"]]
    altered[0]["rgb_sha256"] = "0" * 64
    assert paper_samples.study_content_hash(altered) != paper_samples.PAPER50_CONTENT_SHA256


def test_missing_study_rgb_is_a_validation_error(tmp_path: Path) -> None:
    row = _records(tmp_path)[0]
    entry = {"sample_id": row.sample_id, "shard": row.shard,
             "description_sha256": hashlib.sha256(row.description.encode()).hexdigest(),
             "rgb_sha256": hashlib.sha256(row.rgb_path.read_bytes()).hexdigest()}
    payload = {"samples": [entry], "manifest_sha256": paper_samples.sample_manifest_hash([row.sample_id]),
               "content_sha256": paper_samples.study_content_hash([entry])}
    path = tmp_path / "sample.yaml"
    path.write_text(yaml.safe_dump(payload))
    row.rgb_path.unlink()
    with pytest.raises(ValueError, match="unreadable study RGB"):
        paper_samples.load_study_samples([row], path, expected_count=1,
                                         expected_hash=payload["manifest_sha256"])
