import json
from pathlib import Path

import pytest
from PIL import Image


def add_sample(
    shard: Path,
    sample_id: str,
    description: str,
    *,
    sft_description: str | None = None,
    image_size: tuple[int, int] = (8, 6),
    cube_shape: tuple[int, int, int] = (64, 64, 426),
) -> tuple[dict, dict]:
    patch = shard / "patches" / sample_id
    rgb = patch / "artifacts" / "rgb" / f"{sample_id}.png"
    rgb.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size, color=(20, 80, 140)).save(rgb)
    (patch / "meta.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "shard": shard.name,
                "quality": {"cube_shape": list(cube_shape), "bands": cube_shape[-1]},
                "prompt_set": {"version": "fixture-v1", "hash": "fixture-hash"},
            }
        ),
        encoding="utf-8",
    )
    description_row = {
        "sample_id": sample_id,
        "final_description": description,
        "prompt_set": {"version": "fixture-v1", "hash": "fixture-hash"},
    }
    sft_row = {
        "sample_id": sample_id,
        "images": {"rgb": str(rgb)},
        "structured_answer": {
            "final_description": sft_description
            if sft_description is not None
            else description
        },
    }
    return description_row, sft_row


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def paired_shards(tmp_path: Path) -> Path:
    root = tmp_path / "shards"
    shard = root / "s0001"
    first = add_sample(shard, "sample-a", "short description")
    second = add_sample(shard, "sample-b", "a somewhat longer description")
    write_jsonl(shard / "descriptions.jsonl", [first[0], second[0]])
    write_jsonl(shard / "sft.jsonl", [second[1], first[1]])
    return root
