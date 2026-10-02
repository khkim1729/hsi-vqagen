import json

import pytest
from pydantic import ValidationError

from hsi_vqagen.inference.normalize import normalize_response
from hsi_vqagen.inference.prompts import build_generation_request
from hsi_vqagen.schema.generation import GeneratedPair, RawGeneration


def _pairs(count: int = 4) -> list[dict]:
    return [
        {
            "question": f"질문 {index}?",
            "answer": f"답변 {index}",
            "evidence": f"근거 {index}",
            "question_type": "spatial" if index % 2 else "interpretive",
        }
        for index in range(1, count + 1)
    ]


def _raw(text: str, request) -> RawGeneration:
    return RawGeneration(
        text=text,
        backend=request.backend,
        model=request.model,
        revision=request.revision,
        latency_seconds=0.5,
        model_load_seconds=3.0,
        peak_vram_bytes=1024,
        backend_metadata={"server": "test"},
    )


@pytest.mark.parametrize(
    "wrapper",
    (
        lambda value: value,
        lambda value: f"```json\n{value}\n```",
        lambda value: f"Here is the result:\n{value}\nDone.",
    ),
)
def test_normalizes_clean_fenced_and_extra_prose(record, registry, wrapper) -> None:
    request = build_generation_request(record, registry.by_id["C01"], backend="vllm")
    text = wrapper(json.dumps({"pairs": _pairs()}, ensure_ascii=False))

    case = normalize_response(_raw(text, request), request)

    assert case.validation_status == "valid"
    assert len(case.pairs) == 4
    rows = case.jsonl_rows()
    assert len(rows) == 4
    required = {
        "schema_version", "experiment_id", "case_id", "sample_id", "pair_index",
        "model", "revision", "backend", "input_condition", "provenance",
        "prompt_sha256", "generation_settings", "latency_seconds",
        "model_load_seconds", "peak_vram_bytes", "validation_status",
    }
    assert required <= set(rows[0])


@pytest.mark.parametrize(
    "pairs",
    (
        _pairs(3),
        _pairs(5),
        [{key: value for key, value in pair.items() if key != "answer"} for pair in _pairs()],
        [dict(pair, evidence="") for pair in _pairs()],
        [dict(pair, question="같은 질문?") for pair in _pairs()],
    ),
)
def test_invalid_pair_sets_fail_strict_validation(record, registry, pairs) -> None:
    request = build_generation_request(record, registry.by_id["C05"], backend="vllm")
    case = normalize_response(
        _raw(json.dumps({"pairs": pairs}, ensure_ascii=False), request), request
    )

    assert case.validation_status == "failed"
    assert case.pairs == ()
    assert case.validation_errors


def test_exactly_one_repair_attempt_can_recover(record, registry) -> None:
    request = build_generation_request(record, registry.by_id["C05"], backend="vllm")
    calls = []

    def repair(text: str, error: str) -> RawGeneration:
        calls.append((text, error))
        return _raw(json.dumps({"pairs": _pairs()}, ensure_ascii=False), request)

    case = normalize_response(_raw("not json", request), request, repair=repair)

    assert len(calls) == 1
    assert case.validation_status == "repaired"
    assert len(case.raw_attempts) == 2


def test_repair_failure_is_retained_without_recursion(record, registry) -> None:
    request = build_generation_request(record, registry.by_id["C05"], backend="vllm")
    calls = []

    def repair(text: str, error: str) -> RawGeneration:
        calls.append((text, error))
        return _raw("still not json", request)

    case = normalize_response(_raw("not json", request), request, repair=repair)

    assert len(calls) == 1
    assert case.validation_status == "failed"
    assert len(case.raw_attempts) == 2


def test_generated_pair_forbids_unsupported_manual_mutation() -> None:
    with pytest.raises(ValidationError):
        GeneratedPair(
            question="질문?",
            answer="답변",
            evidence="근거",
            question_type="spatial",
            invented_field="must fail",
        )
