from pathlib import Path

import pytest
import yaml

from hsi_vqagen import config


def test_robustness_registry_is_distinct_and_weight_controlled() -> None:
    """Catches a shuffled condition silently changing weights or decoding settings."""
    registry = config.load_robustness_registry(Path("configs/robustness_experiments.yaml"))
    assert list(registry.by_id) == [f"R{index:02d}" for index in range(1, 7)]
    assert [registry.by_id[key].image_variant for key in registry.by_id] == [
        "paired", "shuffled", "none", "paired", "shuffled", "none"
    ]
    for ids in (("R01", "R02", "R03"), ("R04", "R05", "R06")):
        configs = [registry.by_id[key] for key in ids]
        assert len({item.checkpoint for item in configs}) == 1
        assert len({item.revision for item in configs}) == 1
        assert len({item.generation for item in configs}) == 1
        assert len({item.prompt_version for item in configs}) == 1
    assert registry.sample_manifest_sha256 == (
        "636ce872445090612592511e4805eba08dbd4f42634e82eacb71532b28238d9b"
    )
    assert registry.image_mapping_sha256 == (
        "3420119df2d0eed7b80b08907d88b26ed7d932a598828ac840df93138ff6fc10"
    )


def test_robustness_registry_rejects_checkpoint_drift(tmp_path: Path) -> None:
    """Catches a same-weight RGB experiment becoming a cross-model comparison."""
    original = yaml.safe_load(Path("configs/robustness_experiments.yaml").read_text())
    original["parent_registry"] = str(Path("configs/experiments.yaml").resolve())
    original["configurations"][1]["source_config_id"] = "C03"
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(original), encoding="utf-8")
    with pytest.raises(ValueError, match="same checkpoint|control"):
        config.load_robustness_registry(changed)
