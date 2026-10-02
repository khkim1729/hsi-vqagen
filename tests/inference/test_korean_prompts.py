from hsi_vqagen.inference.prompts import build_generation_request, prompt_sha256


def test_korean_prompt_requests_korean_only_vqa(record, registry) -> None:
    config = registry.by_id["C01"].model_copy(
        update={"prompt_version": "vqa-generation-ko-v1"}
    )

    request = build_generation_request(record, config, backend="vllm")

    assert request.experiment_id == "hsi-vqagen-feasibility-ko-v1"
    assert request.prompt_version == "vqa-generation-ko-v1"
    assert request.prompt_sha256 == prompt_sha256("vqa-generation-ko-v1")
    assert request.messages[0]["content"].startswith("한국어 초분광 원격탐사")
    user_text = request.messages[1]["content"][-1]["text"]
    assert "질문, 답변, 근거를 모두 자연스러운 한국어로" in user_text
    assert record.description in user_text


def test_korean_text_only_request_still_contains_no_image(record, registry) -> None:
    config = registry.by_id["C10"].model_copy(
        update={"prompt_version": "vqa-generation-ko-v1"}
    )

    request = build_generation_request(record, config, backend="vllm")

    assert request.input_condition == "text_only"
    assert request.rgb_path is None
    assert "image_url" not in str(request.messages)
