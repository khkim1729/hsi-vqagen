from pathlib import Path

import pytest
from PIL import Image

from hsi_vqagen.config import load_experiment_registry
from hsi_vqagen.data.records import DatasetRecord


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def record(tmp_path: Path) -> DatasetRecord:
    rgb_path = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(rgb_path)
    meta_path = tmp_path / "meta.json"
    meta_path.write_text("{}", encoding="utf-8")
    return DatasetRecord(
        sample_id="sample-001",
        shard="s0000",
        description="식생과 밝은 토양이 공간적으로 구분되어 나타난다.",
        rgb_path=rgb_path,
        meta_path=meta_path,
        hsi_shape=(64, 64, 426),
        rgb_size=(384, 384),
        prompt_version="source-prompt-v1",
        prompt_hash="source-hash",
    )


@pytest.fixture(scope="session")
def registry():
    return load_experiment_registry(ROOT / "configs" / "experiments.yaml")
