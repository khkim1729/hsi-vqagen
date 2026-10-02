"""Build compact qualitative grids from real normalized feasibility outputs."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image


MODEL_COLUMNS = (
    ("Qwen3-VL-8B", "C01", "C10"),
    ("InternVL3-8B", "C02", "C11"),
    ("Gemma-4-12B", "C03", "C09"),
    ("Mistral Small 3.1 24B", "C04", "C06"),
)


def truncate(value: str, limit: int) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def load_first_pairs(output_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(output_root.glob("*/normalized.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["config_id"]), str(row["sample_id"]))
            if key not in indexed or int(row["pair_index"]) < int(indexed[key]["pair_index"]):
                indexed[key] = row
    return indexed


def _qa_text(row: dict[str, Any]) -> str:
    question = textwrap.fill("Q: " + truncate(str(row["question"]), 125), 32)
    answer = textwrap.fill("A: " + truncate(str(row["answer"]), 85), 32)
    return question + "\n\n" + answer


def _save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_condition_grid(
    *,
    rows: dict[tuple[str, str], dict[str, Any]],
    samples: list[tuple[str, Path]],
    condition: str,
    output_dir: Path,
) -> None:
    config_index = 1 if condition == "multimodal" else 2
    fig, axes = plt.subplots(len(samples), 5, figsize=(16, 4.25 * len(samples)))
    for row_index, (sample_id, rgb_path) in enumerate(samples):
        image_ax = axes[row_index, 0]
        image_ax.imshow(Image.open(rgb_path).convert("RGB"))
        image_ax.set_title("HSI-derived RGB", fontsize=11, weight="bold")
        image_ax.set_xlabel(truncate(sample_id, 32), fontsize=8)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        for column, (label, multimodal_id, text_id) in enumerate(MODEL_COLUMNS, 1):
            config_id = (multimodal_id, text_id)[config_index - 1]
            ax = axes[row_index, column]
            ax.axis("off")
            ax.set_title(label, fontsize=11, weight="bold")
            ax.text(
                0.02, 0.98, _qa_text(rows[(config_id, sample_id)]),
                va="top", ha="left", fontsize=9.5, linespacing=1.3,
                transform=ax.transAxes,
            )
    title = "RGB + Description" if condition == "multimodal" else "Description Only (RGB shown only as a reference)"
    fig.suptitle(f"Preliminary qualitative comparison: {title}", fontsize=16, weight="bold", y=1.005)
    fig.tight_layout(h_pad=2.0, w_pad=1.0)
    _save(fig, output_dir, f"early_{condition}_grid")


def make_gemma_ablation_grid(
    *,
    rows: dict[tuple[str, str], dict[str, Any]],
    samples: list[tuple[str, Path]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(len(samples), 3, figsize=(11, 4.25 * len(samples)))
    for row_index, (sample_id, rgb_path) in enumerate(samples):
        axes[row_index, 0].imshow(Image.open(rgb_path).convert("RGB"))
        axes[row_index, 0].set_title("HSI-derived RGB", fontsize=11, weight="bold")
        axes[row_index, 0].set_xlabel(truncate(sample_id, 38), fontsize=8)
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])
        for column, (config_id, label) in enumerate(
            (("C03", "Gemma-4-12B\nRGB + Description"), ("C09", "Gemma-4-12B\nDescription Only")), 1
        ):
            axes[row_index, column].axis("off")
            axes[row_index, column].set_title(label, fontsize=11, weight="bold")
            axes[row_index, column].text(
                0.02, 0.98, _qa_text(rows[(config_id, sample_id)]),
                va="top", ha="left", fontsize=10, linespacing=1.3,
                transform=axes[row_index, column].transAxes,
            )
    fig.suptitle("Controlled modality ablation with fixed Gemma-4-12B weights", fontsize=16, weight="bold", y=1.005)
    fig.tight_layout(h_pad=2.0, w_pad=1.0)
    _save(fig, output_dir, "early_gemma4_12b_ablation")
