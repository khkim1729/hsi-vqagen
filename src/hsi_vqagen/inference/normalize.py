"""Strict parsing, validation, and optional single-pass JSON repair."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import TypeAdapter, ValidationError

from hsi_vqagen.schema.generation import (
    GeneratedPair,
    GenerationRequest,
    NormalizedCase,
    RawGeneration,
)


RepairCallable = Callable[[str, str], RawGeneration]


def _extract_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found")


def _validate_pairs(text: str) -> tuple[GeneratedPair, ...]:
    payload = _extract_json(text)
    if not isinstance(payload, dict) or set(payload) != {"pairs"}:
        raise ValueError("top-level JSON must contain only the pairs field")
    pairs = TypeAdapter(list[GeneratedPair]).validate_python(payload["pairs"])
    if len(pairs) != 4:
        raise ValueError(f"expected exactly 4 pairs, got {len(pairs)}")
    normalized_questions = [pair.question.strip().casefold() for pair in pairs]
    if len(set(normalized_questions)) != len(normalized_questions):
        raise ValueError("questions must be distinct")
    return tuple(pairs)


def _raw_matches_request(raw: RawGeneration, request: GenerationRequest) -> None:
    expected = (request.backend, request.model, request.revision)
    observed = (raw.backend, raw.model, raw.revision)
    if observed != expected:
        raise ValueError("raw generation provenance does not match request")


def _error_text(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def normalize_response(
    raw: RawGeneration,
    request: GenerationRequest,
    repair: RepairCallable | None = None,
) -> NormalizedCase:
    attempts = [raw]
    errors: list[str] = []
    pairs: tuple[GeneratedPair, ...] = ()
    status = "failed"

    try:
        _raw_matches_request(raw, request)
        pairs = _validate_pairs(raw.text)
        status = "valid"
    except (ValueError, TypeError, ValidationError) as exc:
        errors.append(_error_text(exc))
        if repair is not None:
            repaired = repair(raw.text, errors[-1])
            attempts.append(repaired)
            try:
                _raw_matches_request(repaired, request)
                pairs = _validate_pairs(repaired.text)
                status = "repaired"
            except (ValueError, TypeError, ValidationError) as repair_exc:
                errors.append(_error_text(repair_exc))

    telemetry = attempts[-1]
    return NormalizedCase(
        experiment_id=request.experiment_id,
        case_id=request.case_id,
        sample_id=request.sample_id,
        config_id=request.config_id,
        model=request.model,
        revision=request.revision,
        backend=request.backend,
        input_condition=request.input_condition,
        provenance=request.provenance,
        prompt_version=request.prompt_version,
        prompt_sha256=request.prompt_sha256,
        generation_settings=request.generation.model_dump(mode="json"),
        latency_seconds=sum(attempt.latency_seconds for attempt in attempts),
        model_load_seconds=telemetry.model_load_seconds,
        peak_vram_bytes=max(
            (attempt.peak_vram_bytes for attempt in attempts if attempt.peak_vram_bytes is not None),
            default=None,
        ),
        validation_status=status,
        validation_errors=tuple(errors),
        raw_attempts=tuple(attempt.text for attempt in attempts),
        pairs=pairs if status != "failed" else (),
    )
