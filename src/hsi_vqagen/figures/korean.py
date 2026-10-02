"""Korean qualitative grids with source descriptions and real model outputs."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Sequence

import koreanize_matplotlib  # noqa: F401  # registers a Korean-capable font
import matplotlib.pyplot as plt
from PIL import Image

from hsi_vqagen.figures.early import truncate


MULTIMODAL_COLUMNS = (
    ("Qwen3-VL-8B", "C01"),
    ("InternVL3-8B", "C02"),
    ("Gemma-4-12B", "C03"),
    ("Mistral Small 3.1 24B", "C04"),
)
PRIMARY_TEXT_ONLY_COLUMNS = (
    ("Qwen3-8B", "C05"),
    ("Mistral Small 3.1 24B", "C06"),
    ("Gemma-4-31B", "C07"),
    ("Qwen3-32B", "C08"),
)
CONTROLLED_TEXT_ONLY_COLUMNS = (
    ("Qwen3-VL-8B", "C10"),
    ("InternVL3-8B", "C11"),
    ("Gemma-4-12B", "C09"),
    ("Mistral Small 3.1 24B", "C06"),
)
# Backward-compatible name for callers that expect the main text-only figure.
TEXT_ONLY_COLUMNS = PRIMARY_TEXT_ONLY_COLUMNS


def _qa_text(row: dict[str, Any]) -> str:
    question = textwrap.fill("질문: " + truncate(str(row["question"]), 105), 25)
    answer = textwrap.fill("답변: " + truncate(str(row["answer"]), 75), 25)
    return question + "\n\n" + answer


def _source_text(row: dict[str, Any]) -> str:
    description = str(row.get("provenance", {}).get("description", ""))
    return textwrap.fill(truncate(description, 260), 29)


def make_korean_grid(
    *,
    rows: dict[tuple[str, str], dict[str, Any]],
    samples: list[tuple[str, Path]],
    columns: Sequence[tuple[str, str]],
    title: str,
    stem: str,
    output_dir: Path,
) -> None:
    """Render RGB, source/GT description, and one actual Q/A per model."""

    figure_columns = 2 + len(columns)
    fig, axes = plt.subplots(
        len(samples), figure_columns,
        figsize=(3.45 * figure_columns, 4.5 * len(samples)),
        squeeze=False,
    )
    for row_index, (sample_id, rgb_path) in enumerate(samples):
        image_ax = axes[row_index, 0]
        image_ax.imshow(Image.open(rgb_path).convert("RGB"))
        image_ax.set_title("HSI-derived RGB", fontsize=11, weight="bold")
        image_ax.set_xlabel(truncate(sample_id, 30), fontsize=8)
        image_ax.set_xticks([])
        image_ax.set_yticks([])

        source_row = rows[(columns[0][1], sample_id)]
        source_ax = axes[row_index, 1]
        source_ax.axis("off")
        source_ax.set_title("Source/GT description", fontsize=11, weight="bold")
        source_ax.text(0.02, 0.98, _source_text(source_row), va="top", fontsize=9.3, linespacing=1.28)

        for column_index, (label, config_id) in enumerate(columns, 2):
            ax = axes[row_index, column_index]
            ax.axis("off")
            ax.set_title(label, fontsize=11, weight="bold")
            ax.text(
                0.02, 0.98, _qa_text(rows[(config_id, sample_id)]),
                va="top", fontsize=9.5, linespacing=1.3,
            )
    fig.suptitle(title, fontsize=16, weight="bold", y=1.005)
    fig.tight_layout(h_pad=2.0, w_pad=1.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
