import json
from pathlib import Path

from hsi_vqagen.data.audit import audit_dataset
from hsi_vqagen.data.inspect_dataset import write_audit_outputs

from .conftest import add_sample, write_jsonl


def test_audit_reports_real_distributions(paired_shards: Path) -> None:
    """Catches audits that report row counts without inspecting paired artifacts."""
    audit = audit_dataset(paired_shards)

    assert audit.shard_count == 1
    assert audit.total_description_rows == 2
    assert audit.total_sft_rows == 2
    assert audit.unique_sample_count == 2
    assert audit.fully_paired_count == 2
    assert audit.image_resolutions == {"8x6": 2}
    assert audit.hsi_shapes == {"64x64x426": 2}
    assert audit.description_lengths.minimum == 17
    assert audit.description_lengths.maximum == 29
    assert audit.errors == []


def test_audit_reports_duplicates_malformed_missing_and_disagreement(tmp_path: Path) -> None:
    """Catches silent repair or omission of corrupt and inconsistent shard records."""
    root = tmp_path / "shards"
    shard1 = root / "s0001"
    shard2 = root / "s0002"
    good = add_sample(shard1, "duplicate", "first description")
    mismatch = add_sample(
        shard1,
        "mismatch",
        "description source",
        sft_description="different SFT text",
    )
    missing = add_sample(shard1, "missing", "missing artifacts")
    missing_rgb = Path(missing[1]["images"]["rgb"])
    missing_rgb.unlink()
    (shard1 / "patches" / "missing" / "meta.json").unlink()
    write_jsonl(shard1 / "descriptions.jsonl", [good[0], mismatch[0], missing[0]])
    write_jsonl(shard1 / "sft.jsonl", [good[1], mismatch[1], missing[1]])
    with (shard1 / "descriptions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{malformed\n")

    duplicate = add_sample(shard2, "duplicate", "second description")
    write_jsonl(shard2 / "descriptions.jsonl", [duplicate[0]])
    write_jsonl(shard2 / "sft.jsonl", [duplicate[1]])

    audit = audit_dataset(root)
    codes = [issue.code for issue in audit.errors]

    assert audit.duplicate_sample_ids == ["duplicate"]
    assert audit.malformed_json_count == 1
    assert audit.missing_rgb_count == 1
    assert audit.missing_meta_count == 1
    assert audit.description_disagreement_count == 1
    assert {
        "duplicate_sample_id",
        "malformed_json",
        "missing_rgb",
        "missing_meta",
        "description_disagreement",
    }.issubset(codes)


def test_audit_reports_missing_pair_row(tmp_path: Path) -> None:
    """Catches a description or SFT row disappearing without an audit error."""
    shard = tmp_path / "shards" / "s0001"
    description, _ = add_sample(shard, "description-only", "text")
    write_jsonl(shard / "descriptions.jsonl", [description])
    write_jsonl(shard / "sft.jsonl", [])

    audit = audit_dataset(tmp_path / "shards")

    assert audit.missing_sft_count == 1
    assert any(issue.code == "missing_sft" for issue in audit.errors)


def test_audit_outputs_are_machine_and_human_readable(
    paired_shards: Path, tmp_path: Path
) -> None:
    """Catches a CLI that computes an audit but fails to persist usable evidence."""
    audit = audit_dataset(paired_shards)
    json_path = tmp_path / "audit.json"
    csv_path = tmp_path / "audit.csv"
    markdown_path = tmp_path / "audit.md"

    write_audit_outputs(audit, json_path, csv_path, markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["fully_paired_count"] == 2
    assert "fully_paired_count,2" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Dataset Audit" in markdown
    assert "| Fully paired | 2 |" in markdown
