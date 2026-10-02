from pathlib import Path

import pytest
from pydantic import ValidationError

from hsi_vqagen.config import ExperimentRegistry, load_experiment_registry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "experiments.yaml"

EXPECTED_OUTPUTS = {
    "C01": "qwen3_vl_8b_multimodal",
    "C02": "internvl3_8b_multimodal",
    "C03": "gemma4_12b_multimodal",
    "C04": "mistral_small_3_1_24b_multimodal",
    "C05": "qwen3_8b_text_only",
    "C06": "mistral_small_3_1_24b_text_only",
    "C07": "gemma4_31b_text_only",
    "C08": "qwen3_32b_text_only",
    "C09": "gemma4_12b_text_only",
    "C10": "qwen3_vl_8b_text_only",
    "C11": "internvl3_8b_text_only",
}


def test_registry_has_exact_eleven_configuration_contract() -> None:
    registry = load_experiment_registry(REGISTRY_PATH)
    by_id = registry.by_id

    assert list(by_id) == list(EXPECTED_OUTPUTS)
    assert {key: item.output_dir for key, item in by_id.items()} == EXPECTED_OUTPUTS
    assert len({item.output_dir for item in registry.configurations}) == 11
    assert len({item.checkpoint for item in registry.configurations}) == 7
    assert all(by_id[key].condition == "multimodal" for key in ("C01", "C02", "C03", "C04"))
    assert all(by_id[key].condition == "text_only" for key in EXPECTED_OUTPUTS if key >= "C05")
    assert all(by_id[key].allow_image is False for key in EXPECTED_OUTPUTS if key >= "C05")
    assert registry.sample_manifest_sha256 == (
        "e64324facd80724ec65afb0734f09d559d1038d919cf2204f908244851205b9f"
    )


@pytest.mark.parametrize(
    ("multimodal_id", "text_only_id"),
    (("C01", "C10"), ("C02", "C11"), ("C03", "C09"), ("C04", "C06")),
)
def test_same_checkpoint_pairs_hold_controllable_settings_fixed(
    multimodal_id: str, text_only_id: str
) -> None:
    registry = load_experiment_registry(REGISTRY_PATH)
    multimodal = registry.by_id[multimodal_id]
    text_only = registry.by_id[text_only_id]

    assert multimodal.checkpoint == text_only.checkpoint
    assert multimodal.revision == text_only.revision
    assert multimodal.prompt_version == text_only.prompt_version
    assert multimodal.generation == text_only.generation
    assert multimodal.allow_image is True
    assert text_only.allow_image is False


def test_registry_rejects_modality_pair_drift() -> None:
    registry = load_experiment_registry(REGISTRY_PATH)
    payload = registry.model_dump(mode="json")
    c10 = next(item for item in payload["configurations"] if item["id"] == "C10")
    c10["generation"]["max_new_tokens"] += 1

    with pytest.raises(ValidationError, match="C01/C10"):
        ExperimentRegistry.model_validate(payload)


def test_registry_rejects_images_in_text_only_conditions() -> None:
    registry = load_experiment_registry(REGISTRY_PATH)
    payload = registry.model_dump(mode="json")
    c07 = next(item for item in payload["configurations"] if item["id"] == "C07")
    c07["allow_image"] = True

    with pytest.raises(ValidationError, match="text-only"):
        ExperimentRegistry.model_validate(payload)
