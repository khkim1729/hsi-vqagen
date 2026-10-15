#!/usr/bin/env python3
"""Run one configuration against the immutable five-sample manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from hsi_vqagen.config import load_experiment_registry, load_robustness_registry
from hsi_vqagen.data.records import load_dataset_records
from hsi_vqagen.data.samples import (
    APPROVED_SAMPLE_IDS,
    load_fixed_samples,
    load_rgb_swap_manifest,
    load_study_samples,
    sample_manifest_hash,
)
from hsi_vqagen.inference.backends import VLLMBackend
from hsi_vqagen.inference.runner import run_configuration


ROOT = Path(__file__).resolve().parents[1]


def _exit_code(summary) -> int:
    return 1 if summary.failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id")
    parser.add_argument("--base-url", required=True, help="vLLM endpoint ending in /v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "feasibility")
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--sample-manifest-sha256")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--robustness-registry", type=Path)
    parser.add_argument("--local-paths", type=Path, default=ROOT / "configs" / "local_paths.yaml")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--load-seconds", type=float, default=0.0)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument(
        "--retry-invalid-once",
        action="store_true",
        help="retry one schema/content validation failure with seed+1",
    )
    parser.add_argument(
        "--prompt-version",
        choices=("vqa-generation-v1", "vqa-generation-ko-v1"),
    )
    return parser


def select_registry_for_run(args):
    if args.robustness_registry is None:
        return load_experiment_registry(ROOT / "configs" / "experiments.yaml")
    registry = load_robustness_registry(args.robustness_registry)
    if args.sample_manifest_sha256 != registry.sample_manifest_sha256 or args.sample_count != 20:
        raise ValueError("robustness sample manifest must match the frozen 20-scene registry")
    if args.prompt_version not in (None, "vqa-generation-ko-v1"):
        raise ValueError("robustness prompt must remain vqa-generation-ko-v1")
    return registry


def select_samples_for_run(records, args):
    """Keep the original five-scene gate unless an explicit frozen cohort is given."""

    if args.sample_manifest is None:
        if args.sample_manifest_sha256 is not None or args.sample_count is not None:
            raise ValueError("sample-manifest is required with custom sample options")
        return (
            load_fixed_samples(records, ROOT / "configs" / "feasibility_samples.yaml"),
            sample_manifest_hash(APPROVED_SAMPLE_IDS),
        )
    if args.output_root.resolve() == (ROOT / "outputs" / "feasibility").resolve():
        raise ValueError("custom sample-manifest requires an explicit output-root")
    if not args.sample_manifest_sha256 or args.sample_count is None:
        raise ValueError("custom sample-manifest requires sample-manifest-sha256 and sample-count")
    selected = load_study_samples(
        records,
        args.sample_manifest,
        expected_count=args.sample_count,
        expected_hash=args.sample_manifest_sha256,
    )
    return selected, args.sample_manifest_sha256


def main() -> None:
    args = build_parser().parse_args()

    registry = select_registry_for_run(args)
    try:
        config = registry.by_id[args.config_id]
    except KeyError as exc:
        raise SystemExit(f"unknown configuration: {args.config_id}") from exc
    if args.prompt_version:
        config = config.model_copy(update={"prompt_version": args.prompt_version})
    local_paths = yaml.safe_load(args.local_paths.read_text(encoding="utf-8"))
    records = load_dataset_records(Path(local_paths["dataset_root"]))
    fixed, manifest_hash = select_samples_for_run(records, args)
    experiment_id = None
    if args.sample_manifest is not None:
        study_manifest = yaml.safe_load(args.sample_manifest.read_text(encoding="utf-8"))
        experiment_id = study_manifest.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError("study manifest requires a nonempty experiment_id")
        if args.robustness_registry is not None and experiment_id != "hsi-vqagen-icce2027-rgb-robustness-v1":
            raise ValueError("robustness experiment ID differs from frozen study")
    image_sources = None
    if args.robustness_registry is not None:
        full, frozen_mapping = load_rgb_swap_manifest(
            records, args.sample_manifest,
            expected_count=20,
            expected_sample_hash=registry.sample_manifest_sha256,
            expected_mapping_hash=registry.image_mapping_sha256,
        )
        if [row.sample_id for row in full] != [row.sample_id for row in fixed]:
            raise ValueError("robustness sample order differs from frozen RGB mapping")
        if config.image_variant == "shuffled":
            image_sources = frozen_mapping
    if args.limit is not None:
        fixed = fixed[: args.limit]
    backend = VLLMBackend.from_endpoint(
        args.base_url,
        load_seconds=args.load_seconds,
        server_version="0.30.0",
        gpu_index=args.gpu_index,
    )
    summary = run_configuration(
        config=config,
        records=fixed,
        backend=backend,
        output_root=args.output_root,
        sample_manifest_sha256=manifest_hash,
        resume=not args.no_resume,
        retry_invalid_once=args.retry_invalid_once,
        image_sources=image_sources,
        experiment_id=experiment_id,
        frozen_image_mapping_sha256=(
            registry.image_mapping_sha256
            if args.robustness_registry is not None and config.image_variant == "shuffled"
            else None
        ),
    )
    print(summary.model_dump_json(indent=2))
    raise SystemExit(_exit_code(summary))


if __name__ == "__main__":
    main()
