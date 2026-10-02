#!/usr/bin/env python3
"""Generate preliminary paper grids from the fixed-sample outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from hsi_vqagen.config import load_experiment_registry
from hsi_vqagen.data.records import load_dataset_records
from hsi_vqagen.data.samples import load_fixed_samples
from hsi_vqagen.figures.early import (
    load_first_pairs,
    make_condition_grid,
    make_gemma_ablation_grid,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs" / "early_comparison")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument("--sample-count", type=int, default=2)
    args = parser.parse_args()

    local_paths = yaml.safe_load((ROOT / "configs" / "local_paths.yaml").read_text())
    records = load_dataset_records(Path(local_paths["dataset_root"]))
    fixed = load_fixed_samples(records, ROOT / "configs" / "feasibility_samples.yaml")
    load_experiment_registry(ROOT / "configs" / "experiments.yaml")
    # Spread the main-paper examples across sites instead of taking adjacent rows.
    representative = fixed[::2][: args.sample_count]
    chosen = [(record.sample_id, record.rgb_path) for record in representative]
    rows = load_first_pairs(args.outputs)
    make_condition_grid(rows=rows, samples=chosen, condition="multimodal", output_dir=args.figures)
    make_condition_grid(rows=rows, samples=chosen, condition="text_only", output_dir=args.figures)
    make_gemma_ablation_grid(rows=rows, samples=chosen, output_dir=args.figures)


if __name__ == "__main__":
    main()
