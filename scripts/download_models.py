#!/usr/bin/env python3
"""Download each unique pinned checkpoint once into the shared /data cache."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from hsi_vqagen.config import load_experiment_registry


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/data/hsi-vqagen-cache/huggingface/hub")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="download only this registered checkpoint")
    args = parser.parse_args()
    registry = load_experiment_registry(ROOT / "configs" / "experiments.yaml")
    unique = {}
    for config in registry.configurations:
        unique.setdefault(config.checkpoint, config.revision)
    for checkpoint, revision in unique.items():
        if args.checkpoint and checkpoint != args.checkpoint:
            continue
        print(f"Downloading {checkpoint}@{revision}", flush=True)
        kwargs = {}
        if checkpoint == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
            kwargs["ignore_patterns"] = [
                "model-*.safetensors",
                "model.safetensors.index.json",
                "*.gguf",
            ]
        snapshot_download(
            repo_id=checkpoint,
            revision=revision,
            cache_dir=CACHE,
            max_workers=8,
            **kwargs,
        )
        print(f"Completed {checkpoint}", flush=True)
    if args.checkpoint and args.checkpoint not in unique:
        raise SystemExit(f"checkpoint is not registered: {args.checkpoint}")


if __name__ == "__main__":
    main()
