#!/usr/bin/env python3
"""Skip model loading when a Korean smoke-test group is already complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hsi_vqagen.config import load_experiment_registry


ROOT = Path(__file__).resolve().parents[1]


def group_is_complete(output_root: Path, output_dirs: dict[str, str]) -> bool:
    for config_id, output_dir in output_dirs.items():
        manifest_path = output_root / output_dir / "run_manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if (
            manifest.get("config_id") != config_id
            or manifest.get("prompt_version") != "vqa-generation-ko-v1"
            or len(manifest.get("completed_sample_ids", [])) < 1
            or manifest.get("failed_sample_ids")
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("config_ids", nargs="+")
    args = parser.parse_args()
    registry = load_experiment_registry(ROOT / "configs" / "experiments.yaml")
    output_dirs = {
        config_id: registry.by_id[config_id].output_dir for config_id in args.config_ids
    }
    raise SystemExit(0 if group_is_complete(args.output_root, output_dirs) else 1)


if __name__ == "__main__":
    main()
