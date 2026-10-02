"""Resumable, case-oriented orchestration for one experiment configuration."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from hsi_vqagen.config import ExperimentConfig
from hsi_vqagen.data.records import DatasetRecord
from hsi_vqagen.inference.backends import GenerationBackend
from hsi_vqagen.inference.normalize import RepairCallable, normalize_response
from hsi_vqagen.inference.prompts import build_generation_request
from hsi_vqagen.inference.telemetry import peak_vram_measurement
from hsi_vqagen.schema.generation import NormalizedCase, RawGeneration
from hsi_vqagen.utils.io import atomic_write_json, atomic_write_text
from hsi_vqagen.utils.logging import redact_secrets


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_id: str
    output_dir: Path
    total: int
    completed: int
    failed: int
    skipped: int


def case_file_key(case_id: str) -> str:
    prefix = case_id.split(":", 1)[0].lower()
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _identity_payload(
    config: ExperimentConfig, sample_manifest_sha256: str, prompt_sha256: str
) -> dict[str, Any]:
    identity = {
        "schema_version": "hsi-vqagen.generation-config.v1",
        "config_id": config.id,
        "output_dir": config.output_dir,
        "model": config.checkpoint,
        "revision": config.revision,
        "condition": config.condition,
        "allow_image": config.allow_image,
        "backend": config.preferred_backend,
        "prompt_version": config.prompt_version,
        "prompt_sha256": prompt_sha256,
        "sample_manifest_sha256": sample_manifest_sha256,
        "generation_settings": config.generation.model_dump(mode="json"),
        "template_controls": config.template_controls,
        "preprocessing_exceptions": config.preprocessing_exceptions,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    identity["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return identity


def _load_case(path: Path) -> NormalizedCase | None:
    if not path.is_file():
        return None
    try:
        return NormalizedCase.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _case_telemetry(case: NormalizedCase, raw: RawGeneration | None) -> dict[str, Any]:
    method = None
    if raw is not None:
        method = raw.backend_metadata.get("measurement_method")
    measurement = peak_vram_measurement(
        reader=lambda: case.peak_vram_bytes,
        method=method or "backend_reported_peak",
    )
    return {
        "latency_seconds": case.latency_seconds,
        "model_load_seconds": case.model_load_seconds,
        **measurement.model_dump(mode="json"),
        "validation_status": case.validation_status,
    }


def run_configuration(
    *,
    config: ExperimentConfig,
    records: list[DatasetRecord],
    backend: GenerationBackend,
    output_root: Path,
    sample_manifest_sha256: str,
    resume: bool = True,
    repair: RepairCallable | None = None,
) -> RunSummary:
    output_dir = Path(output_root) / config.output_dir
    raw_dir = output_dir / "raw"
    case_dir = output_dir / "cases"
    raw_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("at least one dataset record is required")
    first_request = build_generation_request(records[0], config)
    identity = _identity_payload(
        config, sample_manifest_sha256, first_request.prompt_sha256
    )
    config_path = output_dir / "generation_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("identity_sha256") != identity["identity_sha256"]:
            raise ValueError("output directory identity mismatch; use a new output directory")
    else:
        atomic_write_json(config_path, identity)

    skipped = 0
    cases: list[NormalizedCase] = []
    raw_by_sample: dict[str, RawGeneration] = {}
    for record in records:
        request = build_generation_request(record, config)
        key = case_file_key(request.case_id)
        raw_path = raw_dir / f"{key}.json"
        case_path = case_dir / f"{key}.json"
        existing_case = _load_case(case_path) if resume else None
        if existing_case is not None and existing_case.validation_status in {"valid", "repaired"}:
            skipped += 1
            cases.append(existing_case)
            if raw_path.is_file():
                try:
                    raw_by_sample[record.sample_id] = RawGeneration.model_validate_json(
                        raw_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    pass
            continue

        started = time.perf_counter()
        try:
            raw = backend.generate(request)
        except Exception as exc:
            raw = RawGeneration(
                text="",
                backend=request.backend,
                model=request.model,
                revision=request.revision,
                latency_seconds=time.perf_counter() - started,
                model_load_seconds=0.0,
                backend_metadata={"measurement_method": "unavailable"},
                error=redact_secrets(f"{type(exc).__name__}: {exc}"),
            )
        case = normalize_response(raw, request, repair=repair if raw.error is None else None)
        atomic_write_json(raw_path, raw.model_dump(mode="json"))
        atomic_write_json(case_path, case.model_dump(mode="json"))
        cases.append(case)
        raw_by_sample[record.sample_id] = raw

    normalized_lines: list[str] = []
    error_lines: list[str] = []
    telemetry: dict[str, Any] = {}
    completed_ids: list[str] = []
    failed_ids: list[str] = []
    for case in cases:
        raw = raw_by_sample.get(case.sample_id)
        telemetry[case.sample_id] = _case_telemetry(case, raw)
        if case.validation_status in {"valid", "repaired"}:
            completed_ids.append(case.sample_id)
            normalized_lines.extend(
                json.dumps(row, ensure_ascii=False, sort_keys=True)
                for row in case.jsonl_rows()
            )
        else:
            failed_ids.append(case.sample_id)
            error = raw.error if raw is not None else None
            error_lines.append(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "sample_id": case.sample_id,
                        "error": redact_secrets(error or "; ".join(case.validation_errors)),
                        "validation_errors": [
                            redact_secrets(value) for value in case.validation_errors
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

    atomic_write_text(
        output_dir / "normalized.jsonl",
        "\n".join(normalized_lines) + ("\n" if normalized_lines else ""),
    )
    atomic_write_text(
        output_dir / "errors.jsonl",
        "\n".join(error_lines) + ("\n" if error_lines else ""),
    )
    load_times = [item["model_load_seconds"] for item in telemetry.values()]
    manifest = {
        "schema_version": "hsi-vqagen.run-manifest.v1",
        **identity,
        "requested_sample_ids": [record.sample_id for record in records],
        "completed_sample_ids": completed_ids,
        "failed_sample_ids": failed_ids,
        "model_load_seconds": max(load_times, default=0.0),
        "case_telemetry": telemetry,
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    return RunSummary(
        config_id=config.id,
        output_dir=output_dir,
        total=len(records),
        completed=len(completed_ids),
        failed=len(failed_ids),
        skipped=skipped,
    )
