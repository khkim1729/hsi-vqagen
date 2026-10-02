from pathlib import Path

import pytest
import yaml

from hsi_vqagen.data.records import DatasetRecord
from hsi_vqagen.data.samples import APPROVED_SAMPLE_IDS, load_fixed_samples


def make_records(tmp_path: Path) -> list[DatasetRecord]:
    from PIL import Image

    records = []
    for index, sample_id in enumerate(APPROVED_SAMPLE_IDS):
        rgb = tmp_path / f"{sample_id}.png"
        Image.new("RGB", (8, 8), color=(index, index, index)).save(rgb)
        meta = tmp_path / f"{sample_id}.json"
        meta.write_text("{}", encoding="utf-8")
        records.append(
            DatasetRecord(
                sample_id=sample_id,
                shard=f"s{index:04d}",
                description=f"description {index}",
                rgb_path=rgb,
                meta_path=meta,
                hsi_shape=(64, 64, 426),
                rgb_size=(8, 8),
                prompt_version="fixture-v1",
                prompt_hash="fixture-hash",
            )
        )
    return records


def test_fixed_samples_load_in_approved_order(tmp_path: Path) -> None:
    """Catches per-configuration resampling or ordering drift."""
    manifest = Path("configs/feasibility_samples.yaml")
    selected = load_fixed_samples(make_records(tmp_path), manifest)
    assert [record.sample_id for record in selected] == list(APPROVED_SAMPLE_IDS)


def test_fixed_samples_reject_missing_record(tmp_path: Path) -> None:
    """Catches a feasibility run silently dropping an approved sample."""
    with pytest.raises(ValueError, match="missing approved sample"):
        load_fixed_samples(make_records(tmp_path)[:-1], Path("configs/feasibility_samples.yaml"))


def test_fixed_samples_reject_reordered_manifest(tmp_path: Path) -> None:
    """Catches a manifest edit that changes the preregistered sample order."""
    original = yaml.safe_load(Path("configs/feasibility_samples.yaml").read_text())
    original["samples"] = list(reversed(original["samples"]))
    manifest = tmp_path / "reordered.yaml"
    manifest.write_text(yaml.safe_dump(original), encoding="utf-8")

    with pytest.raises(ValueError, match="approved order"):
        load_fixed_samples(make_records(tmp_path), manifest)
