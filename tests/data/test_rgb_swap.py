import hashlib
from pathlib import Path

import pytest
import yaml

from hsi_vqagen.data import samples
from tests.data.test_paper_samples import _records


def _mapping_digest(pairs: list[tuple[str, str]]) -> str:
    text = "\n".join(f"{sample}->{image}" for sample, image in pairs) + "\n"
    return hashlib.sha256(text.encode()).hexdigest()


def test_rgb_swap_loader_requires_within_site_derangement(tmp_path: Path) -> None:
    """Catches self images, cross-site swaps, or a mutable mapping after freezing."""
    records = _records(tmp_path)
    selected = [records[0], records[1], records[12], records[13]]
    pairs = [
        (selected[0].sample_id, selected[1].sample_id),
        (selected[1].sample_id, selected[0].sample_id),
        (selected[2].sample_id, selected[3].sample_id),
        (selected[3].sample_id, selected[2].sample_id),
    ]
    sample_hash = samples.sample_manifest_hash([left for left, _ in pairs])
    mapping_hash = _mapping_digest(pairs)
    manifest = tmp_path / "swap.yaml"
    payload = {
        "manifest_sha256": sample_hash,
        "image_mapping_sha256": mapping_hash,
        "samples": [
            {"sample_id": left, "shuffled_rgb_sample_id": right,
             "description_sha256": hashlib.sha256(row.description.encode()).hexdigest(),
             "rgb_sha256": hashlib.sha256(row.rgb_path.read_bytes()).hexdigest()}
            for row, (left, right) in zip(selected, pairs)
        ],
    }
    payload["content_sha256"] = samples.study_content_hash(payload["samples"])
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cohort, mapping = samples.load_rgb_swap_manifest(
        records, manifest, expected_count=4,
        expected_sample_hash=sample_hash, expected_mapping_hash=mapping_hash,
    )
    assert [row.sample_id for row in cohort] == [left for left, _ in pairs]
    assert [mapping[row.sample_id].sample_id for row in cohort] == [right for _, right in pairs]

    altered = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    altered["samples"][0]["shuffled_rgb_sample_id"] = selected[0].sample_id
    altered["image_mapping_sha256"] = _mapping_digest([
        (entry["sample_id"], entry["shuffled_rgb_sample_id"])
        for entry in altered["samples"]
    ])
    manifest.write_text(yaml.safe_dump(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping|derangement"):
        samples.load_rgb_swap_manifest(
            records, manifest, expected_count=4,
            expected_sample_hash=sample_hash, expected_mapping_hash=mapping_hash,
        )


def test_frozen_robustness_manifest_is_a_subset_of_paper_cohort() -> None:
    """Catches image-swap subjects or mapping drifting outside the frozen paper cohort."""
    paper = yaml.safe_load(Path("configs/paper_samples_50.yaml").read_text())
    robustness = yaml.safe_load(Path("configs/robustness_samples_20.yaml").read_text())
    paper_ids = {entry["sample_id"] for entry in paper["samples"]}
    pairs = [(entry["sample_id"], entry["shuffled_rgb_sample_id"]) for entry in robustness["samples"]]

    assert len(pairs) == 20
    assert {left for left, _ in pairs} <= paper_ids
    assert samples.sample_manifest_hash([left for left, _ in pairs]) == (
        "636ce872445090612592511e4805eba08dbd4f42634e82eacb71532b28238d9b"
    )
    assert samples.rgb_mapping_hash(pairs) == (
        "3420119df2d0eed7b80b08907d88b26ed7d932a598828ac840df93138ff6fc10"
    )
    assert robustness["manifest_sha256"] == samples.sample_manifest_hash([left for left, _ in pairs])
    assert robustness["image_mapping_sha256"] == samples.rgb_mapping_hash(pairs)
