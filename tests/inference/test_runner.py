import json
from pathlib import Path

import pytest

from hsi_vqagen.inference.runner import case_file_key, run_configuration
from hsi_vqagen.schema.generation import RawGeneration


def _valid_text(tag: str) -> str:
    return json.dumps(
        {
            "pairs": [
                {
                    "question": f"{tag} 질문 {index}?",
                    "answer": f"답 {index}",
                    "evidence": f"근거 {index}",
                    "question_type": "spatial",
                }
                for index in range(1, 5)
            ]
        },
        ensure_ascii=False,
    )


class _Backend:
    def __init__(self, texts: list[str]):
        self.texts = iter(texts)
        self.calls = []

    def generate(self, request):
        self.calls.append(request.case_id)
        return RawGeneration(
            text=next(self.texts),
            backend=request.backend,
            model=request.model,
            revision=request.revision,
            latency_seconds=0.25,
            model_load_seconds=4.5,
            peak_vram_bytes=2048,
            backend_metadata={"measurement_method": "torch.cuda.max_memory_allocated"},
        )


def test_case_file_key_is_stable_and_files_are_atomic(record, registry, tmp_path: Path) -> None:
    backend = _Backend([_valid_text("one")])
    config = registry.by_id["C05"]

    summary = run_configuration(
        config=config,
        records=[record],
        backend=backend,
        output_root=tmp_path,
        sample_manifest_sha256=registry.sample_manifest_sha256,
    )

    key = case_file_key(f"C05:{record.sample_id}")
    output = tmp_path / config.output_dir
    assert case_file_key(f"C05:{record.sample_id}") == key
    assert (output / "raw" / f"{key}.json").is_file()
    assert (output / "cases" / f"{key}.json").is_file()
    assert not list(output.rglob("*.tmp"))
    assert summary.completed == 1
    assert len((output / "normalized.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_load_seconds"] == 4.5
    assert manifest["case_telemetry"][record.sample_id]["latency_seconds"] == 0.25
    assert manifest["case_telemetry"][record.sample_id]["peak_vram_bytes"] == 2048


def test_resume_preserves_success_and_reruns_failed_only(record, registry, tmp_path: Path) -> None:
    second = record.model_copy(update={"sample_id": "sample-002"})
    config = registry.by_id["C05"]
    first_backend = _Backend([_valid_text("success"), "invalid"])
    first = run_configuration(
        config=config,
        records=[record, second],
        backend=first_backend,
        output_root=tmp_path,
        sample_manifest_sha256=registry.sample_manifest_sha256,
    )
    success_path = tmp_path / config.output_dir / "cases" / (
        case_file_key(f"C05:{record.sample_id}") + ".json"
    )
    original_success = success_path.read_bytes()
    assert (first.completed, first.failed) == (1, 1)

    second_backend = _Backend([_valid_text("recovered")])
    second_run = run_configuration(
        config=config,
        records=[record, second],
        backend=second_backend,
        output_root=tmp_path,
        sample_manifest_sha256=registry.sample_manifest_sha256,
        resume=True,
    )

    assert second_backend.calls == [f"C05:{second.sample_id}"]
    assert success_path.read_bytes() == original_success
    assert (second_run.completed, second_run.failed, second_run.skipped) == (2, 0, 1)
    assert len((tmp_path / config.output_dir / "normalized.jsonl").read_text().splitlines()) == 8


def test_refuses_to_mix_revision_or_settings_in_existing_output(
    record, registry, tmp_path: Path
) -> None:
    config = registry.by_id["C05"]
    run_configuration(
        config=config,
        records=[record],
        backend=_Backend([_valid_text("original")]),
        output_root=tmp_path,
        sample_manifest_sha256=registry.sample_manifest_sha256,
    )

    drifted = config.model_copy(update={"revision": "a" * 40})
    with pytest.raises(ValueError, match="identity mismatch"):
        run_configuration(
            config=drifted,
            records=[record],
            backend=_Backend([_valid_text("drift")]),
            output_root=tmp_path,
            sample_manifest_sha256=registry.sample_manifest_sha256,
        )


def test_backend_error_is_redacted_and_written_for_retry(record, registry, tmp_path: Path) -> None:
    fake_token = "h" + "f_" + "abcdefghijklmnopqrstuvwxyz"

    class BrokenBackend:
        def generate(self, request):
            raise RuntimeError(f"Authorization: Bearer {fake_token}")

    config = registry.by_id["C05"]
    summary = run_configuration(
        config=config,
        records=[record],
        backend=BrokenBackend(),
        output_root=tmp_path,
        sample_manifest_sha256=registry.sample_manifest_sha256,
    )

    errors = (tmp_path / config.output_dir / "errors.jsonl").read_text()
    assert summary.failed == 1
    assert fake_token not in errors
    assert "[REDACTED]" in errors


def test_optional_validation_retry_uses_next_seed_and_preserves_attempts(
    record, registry, tmp_path: Path
) -> None:
    config = registry.by_id["C05"]
    backend = _Backend(["not json", _valid_text("retry")])

    summary = run_configuration(
        config=config,
        records=[record],
        backend=backend,
        output_root=tmp_path,
        sample_manifest_sha256=registry.sample_manifest_sha256,
        retry_invalid_once=True,
    )

    assert summary.completed == 1
    case_path = next((tmp_path / config.output_dir / "cases").glob("*.json"))
    case = json.loads(case_path.read_text(encoding="utf-8"))
    assert case["validation_status"] == "repaired"
    assert len(case["raw_attempts"]) == 2
    manifest = json.loads(
        (tmp_path / config.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["validation_retry_policy"] == {
        "enabled": True,
        "maximum_retries": 1,
        "seed_offset": 1,
    }
