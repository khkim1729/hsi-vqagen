import json

import pytest

from hsi_vqagen.inference.prompts import build_generation_request, build_messages


def _content(messages: list[dict]) -> list[dict]:
    return messages[-1]["content"]


@pytest.mark.parametrize("config_id", [f"C{number:02d}" for number in range(1, 12)])
def test_prompt_keeps_task_contract_and_description(record, registry, config_id: str) -> None:
    config = registry.by_id[config_id]
    messages = build_messages(record, config)
    serialized = json.dumps(messages, ensure_ascii=False)

    assert record.description in serialized
    assert "exactly 4" in serialized
    assert "short, specific answer" in serialized
    assert "question_type" in serialized
    assert "spectral or material claims only when explicitly stated" in serialized


def test_multimodal_has_one_image_before_description_and_text_only_has_none(
    record, registry
) -> None:
    for config_id in ("C01", "C02", "C03", "C04"):
        content = _content(build_messages(record, registry.by_id[config_id]))
        assert [item["type"] for item in content].count("image_url") == 1
        image_index = next(i for i, item in enumerate(content) if item["type"] == "image_url")
        text_index = next(i for i, item in enumerate(content) if item["type"] == "text")
        assert image_index < text_index
        assert content[image_index]["image_url"]["url"].startswith("data:image/png;base64,")

    for config_id in ("C05", "C06", "C07", "C08", "C09", "C10", "C11"):
        request = build_generation_request(record, registry.by_id[config_id], backend="vllm")
        payload = request.model_dump(mode="json", exclude_none=True)
        serialized = json.dumps(payload, ensure_ascii=False).lower()
        assert "image_url" not in serialized
        assert "base64" not in serialized
        assert str(record.rgb_path).lower() not in serialized
        assert request.rgb_path is None


@pytest.mark.parametrize(
    ("multimodal_id", "text_only_id"),
    (("C01", "C10"), ("C02", "C11"), ("C03", "C09"), ("C04", "C06")),
)
def test_ablation_pair_prompt_diff_is_only_evidence_mode_and_image(
    record, registry, multimodal_id: str, text_only_id: str
) -> None:
    multimodal = _content(build_messages(record, registry.by_id[multimodal_id]))
    text_only = _content(build_messages(record, registry.by_id[text_only_id]))

    multimodal_text = next(item["text"] for item in multimodal if item["type"] == "text")
    text_only_text = next(item["text"] for item in text_only if item["type"] == "text")
    canonical_multimodal = multimodal_text.replace(
        "[Evidence mode: RGB + description]", "[Evidence mode: CONTROLLED]"
    )
    canonical_text = text_only_text.replace(
        "[Evidence mode: description only]", "[Evidence mode: CONTROLLED]"
    )
    assert canonical_multimodal == canonical_text
    assert sum(item["type"] == "image_url" for item in multimodal) == 1
    assert sum(item["type"] == "image_url" for item in text_only) == 0
