#!/usr/bin/env python3
"""Run one configuration against the immutable five-sample manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from hsi_vqagen.config import load_experiment_registry
from hsi_vqagen.data.records import load_dataset_records
from hsi_vqagen.data.samples import load_fixed_samples
from hsi_vqagen.inference.backends import VLLMBackend
from hsi_vqagen.inference.runner import run_configuration


ROOT = Path(__file__).resolve().parents[1]


def _exit_code(summary) -> int:
    return 1 if summary.failed else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id")
    parser.add_argument("--base-url", required=True, help="vLLM endpoint ending in /v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "feasibility")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--load-seconds", type=float, default=0.0)
    parser.add_argument("--gpu-index", type=int)
    args = parser.parse_args()

    registry = load_experiment_registry(ROOT / "configs" / "experiments.yaml")
    try:
        config = registry.by_id[args.config_id]
    except KeyError as exc:
        raise SystemExit(f"unknown configuration: {args.config_id}") from exc
    local_paths = yaml.safe_load((ROOT / "configs" / "local_paths.yaml").read_text())
    records = load_dataset_records(Path(local_paths["dataset_root"]))
    fixed = load_fixed_samples(records, ROOT / "configs" / "feasibility_samples.yaml")
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
        sample_manifest_sha256=registry.sample_manifest_sha256,
        resume=not args.no_resume,
    )
    print(summary.model_dump_json(indent=2))
    raise SystemExit(_exit_code(summary))


if __name__ == "__main__":
    main()
