from types import SimpleNamespace

from hsi_vqagen.inference.backends import TransformersBackend, VLLMBackend
from hsi_vqagen.inference.prompts import build_generation_request
from hsi_vqagen.schema.generation import RawGeneration


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"pairs": []}'))]
        )


def test_vllm_backend_maps_request_to_openai_contract(record, registry) -> None:
    completions = _FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    request = build_generation_request(record, registry.by_id["C05"], backend="vllm")

    raw = VLLMBackend(client=client, load_seconds=1.25).generate(request)

    assert isinstance(raw, RawGeneration)
    assert raw.text == '{"pairs": []}'
    assert raw.model_load_seconds == 1.25
    assert completions.kwargs["model"] == request.model
    assert completions.kwargs["max_tokens"] == 1024
    assert completions.kwargs["temperature"] == 0.0
    assert "image_url" not in str(completions.kwargs["messages"])


def test_vllm_backend_inlines_system_message_when_template_requires_it(
    record, registry
) -> None:
    completions = _FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    request = build_generation_request(record, registry.by_id["C02"], backend="vllm")

    VLLMBackend(client=client).generate(request)

    messages = completions.kwargs["messages"]
    assert [message["role"] for message in messages] == ["user"]
    text_parts = [part["text"] for part in messages[0]["content"] if part["type"] == "text"]
    assert text_parts[0].startswith(
        "You generate grounded VQA data and return strict JSON only.\n\n"
    )


def test_vllm_backend_omits_chat_template_kwargs_for_mistral_tokenizer(
    record, registry
) -> None:
    completions = _FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    request = build_generation_request(record, registry.by_id["C04"], backend="vllm")

    VLLMBackend(client=client).generate(request)

    assert "chat_template_kwargs" not in completions.kwargs["extra_body"]


def test_transformers_backend_uses_shared_raw_contract(record, registry) -> None:
    seen = {}

    def runner(request):
        seen["request"] = request
        return '{"pairs": []}'

    request = build_generation_request(record, registry.by_id["C01"], backend="transformers")
    raw = TransformersBackend(
        runner=runner,
        load_seconds=2.5,
        preprocessing_class="FakeProcessor",
        peak_memory_reader=lambda: 1234,
    ).generate(request)

    assert seen["request"] is request
    assert raw.backend == "transformers"
    assert raw.model_load_seconds == 2.5
    assert raw.peak_vram_bytes == 1234
    assert raw.backend_metadata["preprocessing_class"] == "FakeProcessor"
