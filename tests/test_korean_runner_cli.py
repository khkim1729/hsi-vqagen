import hashlib
from pathlib import Path

import pytest
import yaml

from hsi_vqagen.data.samples import sample_manifest_hash, study_content_hash
from scripts import run_configuration
from tests.data.test_samples import make_records


def test_runner_accepts_explicit_korean_prompt_version() -> None:
    args = run_configuration.build_parser().parse_args(
        [
            "C01",
            "--base-url",
            "http://127.0.0.1:8201/v1",
            "--prompt-version",
            "vqa-generation-ko-v1",
        ]
    )

    assert args.prompt_version == "vqa-generation-ko-v1"


def test_explicit_study_manifest_selects_frozen_cohort_in_new_output_root(tmp_path: Path) -> None:
    """Catches custom runs silently falling back to the original five scenes."""
    records = make_records(tmp_path)
    ids = [row.sample_id for row in records[:2]]
    digest = sample_manifest_hash(ids)
    manifest = tmp_path / "paper.yaml"
    payload = {
        "experiment_id": "hsi-vqagen-test-study-v1",
        "manifest_sha256": digest,
        "samples": [{"sample_id": row.sample_id,
                     "description_sha256": hashlib.sha256(row.description.encode()).hexdigest(),
                     "rgb_sha256": hashlib.sha256(row.rgb_path.read_bytes()).hexdigest()}
                    for row in records[:2]],
    }
    payload["content_sha256"] = study_content_hash(payload["samples"])
    # This fixture is deliberately a tiny custom cohort, not the frozen paper50 study.
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    args = run_configuration.build_parser().parse_args([
        "C03", "--base-url", "http://127.0.0.1:8103/v1",
        "--sample-manifest", str(manifest),
        "--sample-manifest-sha256", digest,
        "--sample-count", "2",
        "--output-root", str(tmp_path / "paper_outputs"),
    ])

    selected, selected_hash = run_configuration.select_samples_for_run(records, args)

    assert [row.sample_id for row in selected] == ids
    assert selected_hash == digest


def test_custom_manifest_rejects_default_output_root(tmp_path: Path) -> None:
    """Catches a paper run writing into the original feasibility output tree."""
    args = run_configuration.build_parser().parse_args([
        "C03", "--base-url", "http://127.0.0.1:8103/v1",
        "--sample-manifest", str(tmp_path / "paper.yaml"),
        "--sample-manifest-sha256", "a" * 64,
        "--sample-count", "2",
    ])
    with pytest.raises(ValueError, match="output-root"):
        run_configuration.select_samples_for_run(make_records(tmp_path), args)


def test_robustness_cli_loads_separate_registry_and_mapping(tmp_path: Path) -> None:
    """Catches an R condition accidentally resolving through the C01-C11 registry."""
    args = run_configuration.build_parser().parse_args([
        "R02", "--base-url", "http://127.0.0.1:8102/v1",
        "--robustness-registry", "configs/robustness_experiments.yaml",
        "--sample-manifest", "configs/robustness_samples_20.yaml",
        "--sample-manifest-sha256",
        "636ce872445090612592511e4805eba08dbd4f42634e82eacb71532b28238d9b",
        "--sample-count", "20", "--output-root", str(tmp_path / "robustness"),
    ])

    registry = run_configuration.select_registry_for_run(args)

    assert registry.by_id["R02"].image_variant == "shuffled"
    assert registry.by_id["R02"].checkpoint == registry.by_id["R01"].checkpoint
    assert registry.image_mapping_sha256 == (
        "3420119df2d0eed7b80b08907d88b26ed7d932a598828ac840df93138ff6fc10"
    )


def test_runner_accepts_explicit_local_paths_for_shared_environment(tmp_path: Path) -> None:
    """Catches a main-worktree run requiring an untracked config to be copied."""
    args = run_configuration.build_parser().parse_args([
        "C03", "--base-url", "http://127.0.0.1:8103/v1",
        "--local-paths", str(tmp_path / "local_paths.yaml"),
    ])
    assert args.local_paths == tmp_path / "local_paths.yaml"
