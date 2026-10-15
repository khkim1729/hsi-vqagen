"""Case-level descriptive statistics for frozen Korean VQA studies."""

from __future__ import annotations

import json
import csv
import hashlib
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hsi_vqagen.config import ExperimentConfig
from hsi_vqagen.evaluation.korean import source_description_cosine
from hsi_vqagen.inference.prompts import prompt_sha256


PAIRED_METRICS = (
    "question_type_count",
    "mean_source_description_cosine",
    "mean_answer_characters",
    "latency_seconds",
)


def _cluster_bootstrap_ci(values: list[float], *, reps: int, seed: int) -> list[float]:
    if reps < 2:
        raise ValueError("bootstrap_reps must be at least two")
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(reps)
    )
    return [draws[int(0.025 * (reps - 1))], draws[int(0.975 * (reps - 1))]]


def _eligible_ci_points(
    labels: list[str], deltas: list[float | None], intervals: list[list[float] | None]
) -> list[tuple[str, float, float, float]]:
    return [
        (label, delta, delta - interval[0], interval[1] - delta)
        for label, delta, interval in zip(labels, deltas, intervals)
        if delta is not None and interval is not None
    ]


def _verify_registered_output(
    manifest: dict[str, Any], rows: list[dict[str, Any]],
    config: ExperimentConfig, experiment_id: str,
    frozen: dict[str, dict[str, Any]], mapping_sha256: str | None,
) -> None:
    expected = {
        "config_id": config.id, "model": config.checkpoint,
        "revision": config.revision, "backend": config.preferred_backend,
        "condition": config.condition, "prompt_version": config.prompt_version,
        "prompt_sha256": prompt_sha256(config.prompt_version),
        "generation_settings": config.generation.model_dump(mode="json"),
        "experiment_id": experiment_id,
        "revision_provenance": "configured_registry_revision_not_runtime_attestation",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"registry provenance mismatch for {config.id}: {key}")
    if config.image_variant is not None:
        if manifest.get("image_variant") != config.image_variant:
            raise ValueError(f"image variant mismatch for {config.id}")
        required_mapping = mapping_sha256 if config.image_variant == "shuffled" else None
        if manifest.get("image_mapping_sha256") != required_mapping:
            raise ValueError(f"image mapping mismatch for {config.id}")
    elif manifest.get("image_variant") is not None or manifest.get("image_mapping_sha256") is not None:
        raise ValueError(f"unexpected image variant or mapping for {config.id}")
    saved_inputs = manifest.get("case_input_provenance")
    if not isinstance(saved_inputs, dict):
        raise ValueError(f"missing case input provenance for {config.id}")
    for row in rows:
        sample_id = row.get("sample_id")
        row_expected = {
            "experiment_id": experiment_id, "case_id": f"{config.id}:{sample_id}",
            "sample_id": sample_id, "config_id": config.id, "model": config.checkpoint,
            "revision": config.revision, "backend": config.preferred_backend,
            "input_condition": config.condition, "prompt_version": config.prompt_version,
            "prompt_sha256": expected["prompt_sha256"],
            "generation_settings": expected["generation_settings"],
        }
        if any(row.get(key) != value for key, value in row_expected.items()):
            raise ValueError(f"row provenance mismatch for {config.id}/{sample_id}")
        telemetry = manifest.get("case_telemetry", {}).get(sample_id, {})
        if (row.get("validation_status") not in {"valid", "repaired"}
                or (telemetry.get("validation_status") is not None
                    and row.get("validation_status") != telemetry["validation_status"])):
            raise ValueError(f"row validation status mismatch for {config.id}/{sample_id}")
        if row.get("latency_seconds") != telemetry.get("latency_seconds"):
            raise ValueError(f"row latency provenance mismatch for {config.id}/{sample_id}")
        base_settings = expected["generation_settings"]
        if row["validation_status"] == "valid":
            effective_settings = base_settings
            attempt_provenance = "base_request"
        else:
            policy = manifest.get("validation_retry_policy", {})
            if policy.get("enabled") is not True or policy.get("seed_offset") != 1:
                raise ValueError(f"unregistered generation retry for {config.id}/{sample_id}")
            effective_settings = {**base_settings, "seed": base_settings["seed"] + 1}
            attempt_provenance = "validation_retry_seed_plus_one"
        if (row.get("effective_generation_settings") != effective_settings
                or row.get("attempt_provenance") != attempt_provenance
                or telemetry.get("effective_generation_settings") != effective_settings
                or telemetry.get("attempt_provenance") != attempt_provenance):
            raise ValueError(f"effective generation provenance mismatch for {config.id}/{sample_id}")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or provenance != saved_inputs.get(sample_id):
            raise ValueError(f"row provenance mismatch for {config.id}/{sample_id}")
        entry = frozen.get(sample_id)
        if entry is None:
            raise ValueError(f"unknown frozen sample in {config.id}/{sample_id}")
        if (provenance.get("dataset_shard") != entry.get("shard")
                or provenance.get("description_sha256") != entry.get("description_sha256")
                or provenance.get("rgb_sha256") != entry.get("rgb_sha256")
                or hashlib.sha256(str(provenance.get("description", "")).encode()).hexdigest()
                != entry.get("description_sha256")):
            raise ValueError(f"frozen input provenance mismatch for {config.id}/{sample_id}")
        if config.image_variant is not None:
            image_id = (entry.get("shuffled_rgb_sample_id")
                        if config.image_variant == "shuffled" else sample_id) if config.allow_image else None
            image_digest = frozen[image_id]["rgb_sha256"] if image_id else None
            if (provenance.get("image_variant") != config.image_variant
                    or provenance.get("image_source_sample_id") != image_id
                    or provenance.get("image_rgb_sha256") != image_digest):
                raise ValueError(f"image row provenance mismatch for {config.id}/{sample_id}")


def analyze_study_outputs(
    output_root: Path,
    *,
    config_dirs: dict[str, str],
    pairs: list[tuple[str, str]],
    expected_samples: list[str],
    bootstrap_reps: int = 2000,
    seed: int = 20271001,
    expected_configs: dict[str, ExperimentConfig] | None = None,
    expected_experiment_id: str | None = None,
    expected_provenance: dict[str, dict[str, Any]] | None = None,
    expected_mapping_sha256: str | None = None,
    allow_partial: bool = False,
    min_ci_scenes: int = 10,
) -> dict[str, Any]:
    """Aggregate independent scenes; never treat the four QA rows as four samples."""

    output_root = Path(output_root)
    if not expected_samples or len(set(expected_samples)) != len(expected_samples):
        raise ValueError("expected samples must be a nonempty unique ordered cohort")
    if not config_dirs:
        raise ValueError("at least one configuration is required")
    if (expected_configs is None) != (expected_experiment_id is None) or (
        expected_configs is not None and (expected_provenance is None or set(expected_configs) != set(config_dirs))
    ):
        raise ValueError("registry, experiment ID, and frozen provenance must be supplied together")
    if min_ci_scenes < 2:
        raise ValueError("min_ci_scenes must be at least two")
    summaries: list[dict[str, Any]] = []
    case_metrics: list[dict[str, Any]] = []
    manifest_hashes: set[str] = set()
    any_partial = False
    for config_id, directory in config_dirs.items():
        folder = output_root / directory
        manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("config_id") != config_id:
            raise ValueError(f"configuration mismatch in {folder}")
        if manifest.get("prompt_version") != "vqa-generation-ko-v1":
            raise ValueError(f"Korean prompt required for paper analysis: {config_id}")
        manifest_hashes.add(str(manifest.get("sample_manifest_sha256")))
        requested = manifest.get("requested_sample_ids")
        if not isinstance(requested, list) or not requested or requested != [
            item for item in expected_samples if item in requested
        ]:
            raise ValueError(f"sample manifest order differs in {config_id}")
        missing = [item for item in expected_samples if item not in requested]
        if missing:
            any_partial = True
            if not allow_partial:
                raise ValueError(f"final report requires complete cohort: {config_id}")
        completed = manifest.get("completed_sample_ids", [])
        failed = manifest.get("failed_sample_ids", [])
        if (len(set(completed)) != len(completed) or len(set(failed)) != len(failed)
                or set(completed) | set(failed) != set(requested)):
            raise ValueError(f"run status does not cover expected scenes in {config_id}")
        if set(completed) & set(failed):
            raise ValueError(f"scene is both completed and failed in {config_id}")
        if failed:
            any_partial = True
            if not allow_partial:
                raise ValueError(f"final report requires complete cohort: {config_id}")
        rows = [
            json.loads(line)
            for line in (folder / "normalized.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if expected_configs is not None:
            _verify_registered_output(
                manifest, rows, expected_configs[config_id], expected_experiment_id,
                expected_provenance, expected_mapping_sha256,
            )
        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        type_counts: Counter[str] = Counter()
        for row in rows:
            if row.get("config_id") != config_id or row.get("sample_id") not in completed:
                raise ValueError(f"normalized row does not match completed run in {config_id}")
            by_scene[str(row["sample_id"])].append(row)
            type_counts[str(row.get("question_type", ""))] += 1
        if set(by_scene) != set(completed):
            raise ValueError(f"normalized rows are missing completed scenes in {config_id}")
        for sample_id in completed:
            scene_rows = by_scene[sample_id]
            if len(scene_rows) != 4 or {row.get("pair_index") for row in scene_rows} != {1, 2, 3, 4}:
                raise ValueError(f"normalized pair count mismatch for {config_id}/{sample_id}")
            case_metrics.append({
                "config_id": config_id,
                "sample_id": sample_id,
                "pair_count": 4,
                "question_type_count": len({row["question_type"] for row in scene_rows}),
                "mean_source_description_cosine": statistics.fmean(
                    source_description_cosine(
                        f"{row['answer']} {row['evidence']}",
                        str(row.get("provenance", {}).get("description", "")),
                    )
                    for row in scene_rows
                ),
                "mean_answer_characters": statistics.fmean(len(str(row["answer"])) for row in scene_rows),
                "latency_seconds": float(manifest["case_telemetry"][sample_id]["latency_seconds"]),
            })
        latencies = [
            float(manifest["case_telemetry"][sample_id]["latency_seconds"])
            for sample_id in completed
        ]
        summaries.append({
            "config_id": config_id,
            "requested_case_count": len(expected_samples),
            "run_requested_case_count": len(requested),
            "missing_sample_ids": missing,
            "completed_case_count": len(completed),
            "failed_case_count": len(failed),
            "validation_rate": len(completed) / len(expected_samples),
            "pair_count": len(rows),
            "question_type_distribution": dict(sorted(type_counts.items())),
            "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        })
    if len(manifest_hashes) != 1:
        raise ValueError("sample manifest hash differs across configurations")

    cases_by_key = {(row["config_id"], row["sample_id"]): row for row in case_metrics}
    comparisons = []
    for left_id, right_id in pairs:
        if left_id not in config_dirs or right_id not in config_dirs:
            raise ValueError("comparison refers to an unregistered configuration")
        paired_scenes = [
            sample_id for sample_id in expected_samples
            if (left_id, sample_id) in cases_by_key and (right_id, sample_id) in cases_by_key
        ]
        metrics = {}
        for metric_index, metric in enumerate(PAIRED_METRICS):
            deltas = [
                float(cases_by_key[(left_id, sample_id)][metric])
                - float(cases_by_key[(right_id, sample_id)][metric])
                for sample_id in paired_scenes
            ]
            metrics[metric] = {
                "mean_delta": statistics.fmean(deltas) if deltas else None,
                "ci95": _cluster_bootstrap_ci(
                    deltas, reps=bootstrap_reps, seed=seed + metric_index
                ) if len(deltas) >= min_ci_scenes else None,
            }
        comparisons.append({
            "left_config_id": left_id,
            "right_config_id": right_id,
            "paired_scene_count": len(paired_scenes),
            "metrics": metrics,
        })
    return {
        "schema_version": "hsi-vqagen.study-analysis.v1",
        "sample_manifest_sha256": next(iter(manifest_hashes)),
        "report_status": "exploratory_partial" if any_partial else "final_complete",
        "minimum_ci_scenes": min_ci_scenes,
        "expected_scene_count": len(expected_samples),
        "bootstrap_unit": "scene (one case per configuration)",
        "bootstrap_replicates": bootstrap_reps,
        "metric_note": (
            "Source-description cosine is a lexical overlap proxy, not correctness. "
            "Question types are model-provided labels. Human review is required for answer quality and visual grounding."
        ),
        "configurations": summaries,
        "case_metrics": case_metrics,
        "comparisons": comparisons,
    }


def write_study_artifacts(report: dict[str, Any], prefix: Path) -> None:
    """Export auditable JSON/CSV and a compact descriptive paper figure."""

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.parent / f"{prefix.name}_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def write_csv(suffix: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = prefix.parent / f"{prefix.name}_{suffix}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write_csv("cases", report["case_metrics"])
    write_csv("summary", [
        {
            **item,
            "question_type_distribution": json.dumps(
                item["question_type_distribution"], ensure_ascii=False, sort_keys=True
            ),
        }
        for item in report["configurations"]
    ])
    comparison_rows = [
        {
            "left_config_id": comparison["left_config_id"],
            "right_config_id": comparison["right_config_id"],
            "paired_scene_count": comparison["paired_scene_count"],
            "metric": metric,
            "mean_delta": values["mean_delta"],
            "ci95_low": values["ci95"][0] if values["ci95"] else None,
            "ci95_high": values["ci95"][1] if values["ci95"] else None,
        }
        for comparison in report["comparisons"]
        for metric, values in comparison["metrics"].items()
    ]
    write_csv("comparisons", comparison_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = report["configurations"]
    names = [item["config_id"] for item in configs]
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), constrained_layout=True)
    axes[0].bar(names, [item["validation_rate"] for item in configs], color="#246a73")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Valid case rate")
    axes[0].set_ylabel("Fraction of requested scenes")
    axes[1].bar(names, [item["mean_latency_seconds"] or 0 for item in configs], color="#c16e40")
    axes[1].set_title("Mean generation latency")
    axes[1].set_ylabel("Seconds per completed case")
    comparisons = report["comparisons"]
    labels = [f"{item['left_config_id']}/{item['right_config_id']}" for item in comparisons]
    deltas = [item["metrics"]["question_type_count"]["mean_delta"] for item in comparisons]
    ci = [item["metrics"]["question_type_count"]["ci95"] for item in comparisons]
    axes[2].axhline(0, color="#666666", linewidth=0.7)
    points = [(label, value) for label, value in zip(labels, deltas) if value is not None]
    if points:
        axes[2].plot([label for label, _ in points], [value for _, value in points],
                     "o", color="#463f81")
    ci_points = _eligible_ci_points(labels, deltas, ci)
    if ci_points:
        axes[2].errorbar(
            [item[0] for item in ci_points], [item[1] for item in ci_points],
            yerr=[[item[2] for item in ci_points], [item[3] for item in ci_points]],
            fmt="none", color="#463f81", capsize=3,
        )
    axes[2].set_title("Question-type count delta")
    axes[2].set_ylabel("Left minus right; 95% CI when n is sufficient")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=45)
    for extension in ("png", "pdf"):
        figure.savefig(prefix.parent / f"{prefix.name}_overview.{extension}", dpi=220)
    plt.close(figure)
