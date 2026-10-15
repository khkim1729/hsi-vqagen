#!/usr/bin/env python3
"""Create case-level tables and figures from completed frozen paper studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from hsi_vqagen.config import load_experiment_registry, load_robustness_registry
from hsi_vqagen.data.samples import (
    FROZEN_STUDY_CONTENT, rgb_mapping_hash, sample_manifest_hash, study_content_hash,
)
from hsi_vqagen.evaluation.study import analyze_study_outputs, write_study_artifacts


ROOT = Path(__file__).resolve().parents[1]
PAPER50_SHA256 = "40d3f07e006cd255dee4245fec224b5da16ed958f8dc47b458d83048524a8ec2"


def resolve_study(study: str) -> dict:
    if study == "paper50":
        manifest = yaml.safe_load((ROOT / "configs" / "paper_samples_50.yaml").read_text(encoding="utf-8"))
        registry = load_experiment_registry(ROOT / "configs" / "experiments.yaml")
        pairs = [("C01", "C10"), ("C02", "C11"), ("C03", "C09"), ("C04", "C06")]
        config_ids = [item for pair in pairs for item in pair]
        expected_hash = PAPER50_SHA256
        expected_experiment_id = "hsi-vqagen-icce2027-paper50-v1"
        expected_mapping_sha256 = None
    elif study == "robustness":
        manifest = yaml.safe_load((ROOT / "configs" / "robustness_samples_20.yaml").read_text(encoding="utf-8"))
        registry = load_robustness_registry(ROOT / "configs" / "robustness_experiments.yaml")
        config_ids = [f"R{index:02d}" for index in range(1, 7)]
        pairs = [
            ("R01", "R02"), ("R01", "R03"), ("R02", "R03"),
            ("R04", "R05"), ("R04", "R06"), ("R05", "R06"),
        ]
        expected_hash = registry.sample_manifest_sha256
        expected_experiment_id = "hsi-vqagen-icce2027-rgb-robustness-v1"
        expected_mapping_sha256 = registry.image_mapping_sha256
        mapping = [
            (entry["sample_id"], entry["shuffled_rgb_sample_id"])
            for entry in manifest["samples"]
        ]
        if (
            rgb_mapping_hash(mapping) != registry.image_mapping_sha256
            or manifest.get("image_mapping_sha256") != registry.image_mapping_sha256
        ):
            raise ValueError("robustness RGB mapping hash differs from frozen registry")
    else:
        raise ValueError(f"unknown study: {study}")

    sample_ids = [entry["sample_id"] for entry in manifest["samples"]]
    if sample_manifest_hash(sample_ids) != expected_hash or manifest.get("manifest_sha256") != expected_hash:
        raise ValueError("study sample manifest hash differs from frozen cohort")
    if manifest.get("experiment_id") != expected_experiment_id:
        raise ValueError("study experiment ID differs from frozen cohort")
    if (manifest.get("content_sha256") != study_content_hash(manifest["samples"])
            or manifest["content_sha256"] != FROZEN_STUDY_CONTENT[expected_experiment_id]):
        raise ValueError("study content digest differs from frozen cohort")
    frozen = {entry["sample_id"]: entry for entry in manifest["samples"]}
    if any(not isinstance(entry.get(key), str) or len(entry[key]) != 64
           for entry in frozen.values() for key in ("description_sha256", "rgb_sha256")):
        raise ValueError("study manifest lacks frozen content digests")
    return {
        "expected_samples": sample_ids,
        "config_dirs": {key: registry.by_id[key].output_dir for key in config_ids},
        "pairs": pairs,
        "sample_manifest_sha256": expected_hash,
        "experiment_id": expected_experiment_id,
        "expected_provenance": frozen,
        "expected_configs": {
            key: registry.by_id[key].model_copy(update={"prompt_version": "vqa-generation-ko-v1"})
            for key in config_ids
        },
        "expected_mapping_sha256": expected_mapping_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=("paper50", "robustness"), required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--artifact-prefix", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20271001)
    parser.add_argument("--allow-partial", action="store_true",
                        help="emit explicitly exploratory statistics for incomplete runs")
    args = parser.parse_args()
    setup = resolve_study(args.study)
    report = analyze_study_outputs(
        args.outputs,
        config_dirs=setup["config_dirs"],
        pairs=setup["pairs"],
        expected_samples=setup["expected_samples"],
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        expected_configs=setup["expected_configs"],
        expected_experiment_id=setup["experiment_id"],
        expected_provenance=setup["expected_provenance"],
        expected_mapping_sha256=setup["expected_mapping_sha256"],
        allow_partial=args.allow_partial,
    )
    if report["sample_manifest_sha256"] != setup["sample_manifest_sha256"]:
        raise ValueError("output manifests do not match the selected frozen study")
    write_study_artifacts(report, args.artifact_prefix)
    print(json.dumps({
        "study": args.study,
        "report_status": report["report_status"],
        "configurations": len(report["configurations"]),
        "completed_cases": sum(item["completed_case_count"] for item in report["configurations"]),
        "artifact_prefix": str(args.artifact_prefix),
    }, indent=2))


if __name__ == "__main__":
    main()
