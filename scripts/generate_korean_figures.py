#!/usr/bin/env python3
"""Generate Korean VQA comparison grids with the source description."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from hsi_vqagen.data.records import load_dataset_records
from hsi_vqagen.data.samples import load_fixed_samples
from hsi_vqagen.figures.early import load_first_pairs
from hsi_vqagen.figures.korean import (
    MULTIMODAL_COLUMNS,
    TEXT_ONLY_COLUMNS,
    make_korean_grid,
)


ROOT = Path(__file__).resolve().parents[1]


def _sample_paths(records) -> list[tuple[str, Path]]:
    return [(record.sample_id, record.rgb_path) for record in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs" / "korean_comparison")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    local_paths = yaml.safe_load((ROOT / "configs" / "local_paths.yaml").read_text())
    records = load_dataset_records(Path(local_paths["dataset_root"]))
    fixed = load_fixed_samples(records, ROOT / "configs" / "feasibility_samples.yaml")
    rows = load_first_pairs(args.outputs)
    representative = _sample_paths([fixed[0], fixed[2]])
    all_samples = _sample_paths(fixed)

    for suffix, samples in (("", representative), ("_all", all_samples)):
        make_korean_grid(
            rows=rows, samples=samples, columns=MULTIMODAL_COLUMNS,
            title="한국어 VQA 비교: RGB + 한국어 description",
            stem=f"korean_multimodal_grid{suffix}", output_dir=args.figures,
        )
        make_korean_grid(
            rows=rows, samples=samples, columns=TEXT_ONLY_COLUMNS,
            title="한국어 VQA 비교: 한국어 description only (RGB는 독자 참고용)",
            stem=f"korean_text_only_grid{suffix}", output_dir=args.figures,
        )
        make_korean_grid(
            rows=rows, samples=samples,
            columns=(("Gemma-4-12B\nRGB + description", "C03"), ("Gemma-4-12B\ndescription only", "C09")),
            title="Gemma-4-12B 한국어 controlled modality ablation",
            stem=f"korean_gemma4_12b_ablation{suffix}", output_dir=args.figures,
        )


if __name__ == "__main__":
    main()
