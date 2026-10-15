import json
from pathlib import Path

import pytest
from PIL import Image

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
        self.requests = []

    def generate(self, request):
        self.calls.append(request.case_id)
        self.requests.append(request)
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
    assert json.loads((output / "normalized.jsonl").read_text().splitlines()[0])["experiment_id"] == (
        "hsi-vqagen-feasibility-v1"
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_load_seconds"] == 4.5
    assert manifest["case_telemetry"][record.sample_id]["latency_seconds"] == 0.25
    assert manifest["case_telemetry"][record.sample_id]["peak_vram_bytes"] == 2048


def test_shuffled_run_uses_frozen_image_mapping_and_records_its_hash(
    record, registry, tmp_path: Path
) -> None:
    """Catches a robustness run reusing the correct image or omitting mapping identity."""
    other_path = tmp_path / "other.png"
    Image.new("RGB", (8, 8), color="red").save(other_path)
    other = record.model_copy(update={"sample_id": "sample-002", "rgb_path": other_path})
    config = registry.by_id["C01"].model_copy(update={
        "id": "R02", "output_dir": "r02_shuffled", "image_variant": "shuffled"
    })
    mapping = {record.sample_id: other, other.sample_id: record}

    summary = run_configuration(
        config=config, records=[record, other], backend=_Backend([
            _valid_text("one"), _valid_text("two")
        ]), output_root=tmp_path, sample_manifest_sha256="a" * 64,
        image_sources=mapping,
    )

    assert summary.completed == 2
    output = tmp_path / "r02_shuffled"
    rows = [json.loads(line) for line in (output / "normalized.jsonl").read_text().splitlines()]
    assert {row["provenance"]["image_source_sample_id"] for row in rows if row["sample_id"] == record.sample_id} == {other.sample_id}
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert len(manifest["image_mapping_sha256"]) == 64
    assert manifest["image_variant"] == "shuffled"


def test_shuffled_run_fails_before_writing_if_mapping_missing(record, registry, tmp_path: Path) -> None:
    """Catches partial robustness artifacts from an incomplete mapping."""
    config = registry.by_id["C01"].model_copy(update={
        "id": "R02", "output_dir": "r02_shuffled", "image_variant": "shuffled"
    })
    with pytest.raises(ValueError, match="image mapping"):
        run_configuration(
            config=config, records=[record], backend=_Backend([]),
            output_root=tmp_path, sample_manifest_sha256="a" * 64,
        )
    assert not (tmp_path / "r02_shuffled").exists()


def test_partial_shuffled_run_keeps_full_frozen_mapping_identity(
    record, registry, tmp_path: Path
) -> None:
    other_path = tmp_path / "other.png"
    Image.new("RGB", (8, 8), color="red").save(other_path)
    other = record.model_copy(update={"sample_id": "sample-002", "rgb_path": other_path})
    config = registry.by_id["C01"].model_copy(update={
        "id": "R02", "output_dir": "r02_partial", "image_variant": "shuffled"
    })
    mapping = {record.sample_id: other, other.sample_id: record}
    from hsi_vqagen.data.samples import rgb_mapping_hash
    full_hash = rgb_mapping_hash([(key, value.sample_id) for key, value in mapping.items()])
    summary = run_configuration(
        config=config, records=[record], backend=_Backend([_valid_text("partial")]),
        output_root=tmp_path, sample_manifest_sha256="a" * 64,
        image_sources=mapping, frozen_image_mapping_sha256=full_hash,
    )
    manifest = json.loads((summary.output_dir / "run_manifest.json").read_text())
    assert summary.completed == 1
    assert manifest["image_mapping_sha256"] == full_hash
    assert manifest["requested_sample_ids"] == [record.sample_id]


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


def test_resume_rejects_changed_input_provenance(record, registry, tmp_path: Path) -> None:
    config = registry.by_id["C05"]
    run_configuration(config=config, records=[record], backend=_Backend([_valid_text("old")]),
                      output_root=tmp_path, sample_manifest_sha256=registry.sample_manifest_sha256)
    changed = record.model_copy(update={"description": record.description + " altered"})
    backend = _Backend([_valid_text("new")])
    summary = run_configuration(config=config, records=[changed], backend=backend,
                                output_root=tmp_path, sample_manifest_sha256=registry.sample_manifest_sha256)
    assert summary.skipped == 0
    assert backend.calls == [f"C05:{record.sample_id}"]


def test_resume_rejects_changed_rgb_bytes(record, registry, tmp_path: Path) -> None:
    config = registry.by_id["C05"]
    run_configuration(config=config, records=[record], backend=_Backend([_valid_text("old")]),
                      output_root=tmp_path, sample_manifest_sha256=registry.sample_manifest_sha256)
    Image.new("RGB", (8, 8), color="blue").save(record.rgb_path)
    backend = _Backend([_valid_text("new")])
    summary = run_configuration(config=config, records=[record], backend=backend,
                                output_root=tmp_path, sample_manifest_sha256=registry.sample_manifest_sha256)
    assert summary.skipped == 0
    assert backend.calls == [f"C05:{record.sample_id}"]


def test_study_experiment_id_is_separate_from_legacy(record, registry, tmp_path: Path) -> None:
    config = registry.by_id["C05"]
    summary = run_configuration(config=config, records=[record], backend=_Backend([_valid_text("study")]),
                                output_root=tmp_path, sample_manifest_sha256="a" * 64,
                                experiment_id="hsi-vqagen-icce2027-paper50-v1")
    row = json.loads((summary.output_dir / "normalized.jsonl").read_text().splitlines()[0])
    assert row["experiment_id"] == "hsi-vqagen-icce2027-paper50-v1"
    assert row["provenance"]["description_sha256"]
    assert row["provenance"]["rgb_sha256"]
    manifest = json.loads((summary.output_dir / "run_manifest.json").read_text())
    assert manifest["revision_provenance"] == "configured_registry_revision_not_runtime_attestation"


def test_study_resume_rebuilds_case_missing_attempt_provenance(record, registry, tmp_path: Path) -> None:
    config = registry.by_id["C05"]
    study_id = "hsi-vqagen-icce2027-paper50-v1"
    first = run_configuration(config=config, records=[record], backend=_Backend([_valid_text("first")]),
                              output_root=tmp_path, sample_manifest_sha256="a" * 64,
                              experiment_id=study_id)
    case_path = next((first.output_dir / "cases").glob("*.json"))
    saved = json.loads(case_path.read_text())
    saved.pop("effective_generation_settings")
    saved.pop("attempt_provenance")
    case_path.write_text(json.dumps(saved))
    backend = _Backend([_valid_text("fresh")])
    second = run_configuration(config=config, records=[record], backend=backend,
                               output_root=tmp_path, sample_manifest_sha256="a" * 64,
                               experiment_id=study_id)
    assert second.skipped == 0
    assert backend.calls == [f"C05:{record.sample_id}"]
    saved = json.loads(case_path.read_text())
    saved["effective_generation_settings"]["seed"] = config.generation.seed + 1
    case_path.write_text(json.dumps(saved))
    third_backend = _Backend([_valid_text("fresh again")])
    third = run_configuration(config=config, records=[record], backend=third_backend,
                              output_root=tmp_path, sample_manifest_sha256="a" * 64,
                              experiment_id=study_id)
    assert third.skipped == 0
    assert third_backend.calls == [f"C05:{record.sample_id}"]


def test_study_output_identity_pins_retry_policy(record, registry, tmp_path: Path) -> None:
    config = registry.by_id["C05"]
    run_configuration(config=config, records=[record], backend=_Backend([_valid_text("first")]),
                      output_root=tmp_path, sample_manifest_sha256="a" * 64,
                      experiment_id="hsi-vqagen-icce2027-paper50-v1", retry_invalid_once=True)
    with pytest.raises(ValueError, match="identity mismatch"):
        run_configuration(config=config, records=[record], backend=_Backend([]),
                          output_root=tmp_path, sample_manifest_sha256="a" * 64,
                          experiment_id="hsi-vqagen-icce2027-paper50-v1",
                          retry_invalid_once=False)


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
    row = json.loads((tmp_path / config.output_dir / "normalized.jsonl").read_text().splitlines()[0])
    assert row["generation_settings"]["seed"] == config.generation.seed
    assert row["effective_generation_settings"]["seed"] == config.generation.seed + 1
    assert row["attempt_provenance"] == "validation_retry_seed_plus_one"
    assert manifest["case_telemetry"][record.sample_id]["effective_generation_settings"]["seed"] == (
        config.generation.seed + 1
    )
    attempts_path = tmp_path / config.output_dir / "raw" / (
        case_file_key(f"C05:{record.sample_id}") + ".attempts.json"
    )
    attempts = json.loads(attempts_path.read_text())
    assert len(attempts) == 2
    assert attempts[1]["backend_metadata"]["retry_seed"] == config.generation.seed + 1
    assert attempts[1]["backend_metadata"]["retry_mode"] == "corrective_prompt"
    assert len(attempts[1]["backend_metadata"]["repair_prompt_sha256"]) == 64
    retry_messages = backend.requests[1].messages
    assert retry_messages[-2] == {
        "role": "assistant", "content": [{"type": "text", "text": "not json"}]
    }
    assert retry_messages[-1]["role"] == "user"
    correction = retry_messages[-1]["content"][0]["text"]
    assert "exactly 4 distinct entries" in correction
    assert "Validation error: ValueError: no JSON object found" in correction
