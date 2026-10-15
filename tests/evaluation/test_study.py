import json
import hashlib
from pathlib import Path

import pytest

from hsi_vqagen import evaluation as study


def _write_condition(root: Path, config_id: str, completed: list[str], latency: float) -> None:
    folder = root / config_id
    folder.mkdir(parents=True)
    requested = ["s1", "s2", "s3"]
    manifest = {
        "config_id": config_id,
        "sample_manifest_sha256": "a" * 64,
        "prompt_version": "vqa-generation-ko-v1",
        "requested_sample_ids": requested,
        "completed_sample_ids": completed,
        "failed_sample_ids": [item for item in requested if item not in completed],
        "case_telemetry": {item: {"latency_seconds": latency} for item in requested},
    }
    (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for sample_id in completed:
        for index in range(1, 5):
            rows.append({
                "config_id": config_id, "sample_id": sample_id, "pair_index": index,
                "question_type": "attribute" if config_id == "A" and index % 2 else "spatial",
                "question": f"질문 {sample_id} {index}", "answer": "숲입니다.",
                "evidence": "수관이 있습니다.",
                "provenance": {"description": "수관이 있습니다."},
                "latency_seconds": latency,
            })
    (folder / "normalized.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_study_analysis_uses_case_level_pairs_and_manifest_failures(tmp_path: Path) -> None:
    """Catches pair-row pseudoreplication and valid-only denominators."""
    _write_condition(tmp_path, "A", ["s1", "s2"], 3.0)
    _write_condition(tmp_path, "B", ["s1", "s2", "s3"], 2.0)

    report = study.analyze_study_outputs(
        tmp_path, config_dirs={"A": "A", "B": "B"}, pairs=[("A", "B")],
        expected_samples=["s1", "s2", "s3"], bootstrap_reps=100, seed=7,
        allow_partial=True,
    )

    summaries = {item["config_id"]: item for item in report["configurations"]}
    assert summaries["A"]["validation_rate"] == pytest.approx(2 / 3)
    assert summaries["B"]["validation_rate"] == 1.0
    assert summaries["A"]["question_type_distribution"] == {"attribute": 4, "spatial": 4}
    assert len(report["case_metrics"]) == 5
    paired = report["comparisons"][0]
    assert paired["paired_scene_count"] == 2
    assert paired["metrics"]["question_type_count"]["mean_delta"] == 1.0
    assert paired["metrics"]["question_type_count"]["ci95"] is None
    assert paired["metrics"]["latency_seconds"]["ci95"] is None
    assert report["report_status"] == "exploratory_partial"
    assert "not correctness" in report["metric_note"]


def test_study_analysis_rejects_mixed_sample_manifests(tmp_path: Path) -> None:
    """Catches stale results from another cohort entering paired statistics."""
    _write_condition(tmp_path, "A", ["s1", "s2"], 3.0)
    _write_condition(tmp_path, "B", ["s1", "s2"], 2.0)
    path = tmp_path / "B" / "run_manifest.json"
    payload = json.loads(path.read_text())
    payload["sample_manifest_sha256"] = "b" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sample manifest"):
        study.analyze_study_outputs(
            tmp_path, config_dirs={"A": "A", "B": "B"},
            pairs=[("A", "B")], expected_samples=["s1", "s2", "s3"], allow_partial=True,
        )


def test_study_analysis_rejects_english_pilot_in_korean_paper(tmp_path: Path) -> None:
    """Catches compatible-looking English rows entering the Korean paper analysis."""
    _write_condition(tmp_path, "A", ["s1", "s2"], 3.0)
    path = tmp_path / "A" / "run_manifest.json"
    payload = json.loads(path.read_text())
    payload["prompt_version"] = "vqa-generation-v1"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Korean prompt"):
        study.analyze_study_outputs(
            tmp_path, config_dirs={"A": "A"}, pairs=[],
            expected_samples=["s1", "s2", "s3"], allow_partial=True,
        )


def test_study_artifact_writer_exports_reproducible_tables_and_figure(tmp_path: Path) -> None:
    """Catches an analysis that cannot be used to assemble the paper figures."""
    outputs = tmp_path / "outputs"
    _write_condition(outputs, "A", ["s1", "s2"], 3.0)
    _write_condition(outputs, "B", ["s1", "s2"], 2.0)
    report = study.analyze_study_outputs(
        outputs, config_dirs={"A": "A", "B": "B"},
        pairs=[("A", "B")], expected_samples=["s1", "s2", "s3"],
        bootstrap_reps=20, allow_partial=True,
    )

    study.write_study_artifacts(report, tmp_path / "artifacts" / "study")

    assert json.loads((tmp_path / "artifacts" / "study_metrics.json").read_text()) == report
    assert "config_id" in (tmp_path / "artifacts" / "study_cases.csv").read_text().splitlines()[0]
    assert "validation_rate" in (tmp_path / "artifacts" / "study_summary.csv").read_text().splitlines()[0]
    assert "question_type_count" in (tmp_path / "artifacts" / "study_comparisons.csv").read_text()
    assert (tmp_path / "artifacts" / "study_overview.png").stat().st_size > 1000
    assert (tmp_path / "artifacts" / "study_overview.pdf").stat().st_size > 1000


def test_mixed_precision_figure_keeps_only_eligible_ci_bars() -> None:
    from hsi_vqagen.evaluation.study import _eligible_ci_points

    eligible = _eligible_ci_points(
        ["high", "low"], [1.0, 2.0], [[0.5, 1.5], None]
    )
    assert eligible == [("high", 1.0, 0.5, 0.5)]


def test_final_report_refuses_incomplete_cohort(tmp_path: Path) -> None:
    _write_condition(tmp_path, "A", ["s1", "s2"], 3.0)
    with pytest.raises(ValueError, match="complete cohort"):
        study.analyze_study_outputs(
            tmp_path, config_dirs={"A": "A"}, pairs=[], expected_samples=["s1", "s2", "s3"]
        )


def test_partial_report_accepts_requested_subset_but_marks_exploratory(tmp_path: Path) -> None:
    _write_condition(tmp_path, "A", ["s1", "s2"], 3.0)
    path = tmp_path / "A" / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["requested_sample_ids"] = ["s1", "s2"]
    manifest["failed_sample_ids"] = []
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete cohort"):
        study.analyze_study_outputs(tmp_path, config_dirs={"A": "A"}, pairs=[],
                                    expected_samples=["s1", "s2", "s3"])
    report = study.analyze_study_outputs(tmp_path, config_dirs={"A": "A"}, pairs=[],
                                         expected_samples=["s1", "s2", "s3"], allow_partial=True)
    assert report["report_status"] == "exploratory_partial"
    assert report["configurations"][0]["validation_rate"] == pytest.approx(2 / 3)
    assert report["configurations"][0]["missing_sample_ids"] == ["s3"]


def test_registry_and_frozen_provenance_reject_mixed_rows(tmp_path: Path) -> None:
    from hsi_vqagen.config import load_robustness_registry
    from hsi_vqagen.inference.prompts import prompt_sha256

    config = load_robustness_registry(Path("configs/robustness_experiments.yaml")).by_id["R02"]
    description = "frozen scene description"
    digest = hashlib.sha256(description.encode()).hexdigest()
    source = "s1"
    image = "s2"
    provenance = {
        "dataset_shard": "s0001", "description": description,
        "description_sha256": digest, "rgb_sha256": "a" * 64,
        "image_variant": "shuffled", "image_source_sample_id": image,
        "image_rgb_sha256": "b" * 64,
    }
    expected = {
        source: {"shard": "s0001", "description_sha256": digest,
                 "rgb_sha256": "a" * 64, "shuffled_rgb_sample_id": image},
        image: {"shard": "s0002", "description_sha256": "c" * 64,
                "rgb_sha256": "b" * 64, "shuffled_rgb_sample_id": source},
    }
    folder = tmp_path / config.output_dir
    folder.mkdir()
    manifest = {
        "config_id": config.id, "model": config.checkpoint, "revision": config.revision,
        "backend": config.preferred_backend, "condition": config.condition,
        "generation_settings": config.generation.model_dump(mode="json"),
        "prompt_version": config.prompt_version, "prompt_sha256": prompt_sha256(config.prompt_version),
        "experiment_id": "hsi-vqagen-icce2027-rgb-robustness-v1",
        "revision_provenance": "configured_registry_revision_not_runtime_attestation",
        "sample_manifest_sha256": "d" * 64, "image_variant": "shuffled",
        "image_mapping_sha256": "e" * 64,
        "requested_sample_ids": [source, image], "completed_sample_ids": [source],
        "failed_sample_ids": [image], "case_input_provenance": {source: provenance},
        "case_telemetry": {source: {
            "latency_seconds": 1.0,
            "effective_generation_settings": config.generation.model_dump(mode="json"),
            "attempt_provenance": "base_request",
        }},
    }
    (folder / "run_manifest.json").write_text(json.dumps(manifest))
    common = {
        "experiment_id": manifest["experiment_id"], "case_id": f"R02:{source}",
        "sample_id": source, "config_id": config.id, "model": config.checkpoint,
        "revision": config.revision, "backend": config.preferred_backend,
        "input_condition": config.condition, "prompt_version": config.prompt_version,
        "prompt_sha256": manifest["prompt_sha256"],
        "generation_settings": manifest["generation_settings"], "provenance": provenance,
        "effective_generation_settings": manifest["generation_settings"],
        "attempt_provenance": "base_request",
        "validation_status": "valid", "latency_seconds": 1.0,
        "question_type": "attribute", "answer": "answer", "evidence": "evidence",
    }
    rows = [{**common, "pair_index": index} for index in range(1, 5)]
    (folder / "normalized.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    kwargs = dict(config_dirs={"R02": config.output_dir}, pairs=[],
                  expected_samples=[source, image], expected_configs={"R02": config},
                  expected_experiment_id=manifest["experiment_id"],
                  expected_provenance=expected, expected_mapping_sha256="e" * 64,
                  allow_partial=True)
    study.analyze_study_outputs(tmp_path, **kwargs)
    rows[0]["revision"] = "f" * 40
    (folder / "normalized.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="row provenance"):
        study.analyze_study_outputs(tmp_path, **kwargs)
    rows[0]["revision"] = config.revision
    manifest["image_mapping_sha256"] = "f" * 64
    (folder / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mapping"):
        study.analyze_study_outputs(tmp_path, **kwargs)
    manifest["image_mapping_sha256"] = "e" * 64
    (folder / "run_manifest.json").write_text(json.dumps(manifest))
    rows[0]["validation_status"] = "failed"
    (folder / "normalized.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="row validation status"):
        study.analyze_study_outputs(tmp_path, **kwargs)
    rows[0]["validation_status"] = "valid"
    rows[0]["effective_generation_settings"] = {
        **config.generation.model_dump(mode="json"), "seed": config.generation.seed + 1
    }
    (folder / "normalized.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="effective generation"):
        study.analyze_study_outputs(tmp_path, **kwargs)
