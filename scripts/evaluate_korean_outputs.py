#!/usr/bin/env python3
"""Write Korean VQA proxy metrics from normalized JSONL outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hsi_vqagen.evaluation.korean import evaluate_korean_outputs


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/normalized.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    summaries = evaluate_korean_outputs(load_rows(args.outputs))
    if not summaries:
        raise SystemExit(f"no normalized rows found under {args.outputs}")
    payload = {
        "schema_version": "hsi-vqagen.korean-metrics.v1",
        "metric_note": (
            "source-description cosine is a character 2-4 gram TF-IDF lexical grounding proxy, "
            "not a semantic correctness or human-quality score"
        ),
        "configurations": [summary.model_dump(mode="json") for summary in summaries],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(summaries[0].model_dump())
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary.model_dump(mode="json") for summary in summaries)


if __name__ == "__main__":
    main()
