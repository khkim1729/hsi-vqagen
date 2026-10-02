import pandas as pd

from hsi_vqagen.evaluation.checks import evaluate_deterministic
from hsi_vqagen.evaluation.tables import build_manual_review_sheet


def _row(
    config_id="C01",
    sample_id="sample-1",
    pair_index=1,
    question="어떤 영역이 더 밝습니까?",
    answer="오른쪽 영역입니다.",
    evidence="RGB의 오른쪽이 더 밝다.",
    condition="multimodal",
    description="밝은 오른쪽 영역과 어두운 왼쪽 영역이 있다.",
):
    return {
        "schema_version": "hsi-vqagen.normalized.v1",
        "experiment_id": "exp",
        "case_id": f"{config_id}:{sample_id}",
        "sample_id": sample_id,
        "config_id": config_id,
        "pair_index": pair_index,
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "question_type": "spatial",
        "model": "model",
        "revision": "a" * 40,
        "backend": "vllm",
        "input_condition": condition,
        "provenance": {"description": description},
        "prompt_sha256": "b" * 64,
        "generation_settings": {},
        "latency_seconds": 1.0,
        "model_load_seconds": 2.0,
        "peak_vram_bytes": 3,
        "validation_status": "valid",
    }


def test_checks_duplicate_paraphrase_empty_spectral_and_visual_leakage() -> None:
    rows = [
        _row(pair_index=1),
        _row(pair_index=2),
        _row(pair_index=3, question="", answer="답"),
        _row(pair_index=4, question="NDVI 값이 높은가?", answer="높다."),
        _row(
            config_id="C05",
            condition="text_only",
            pair_index=1,
            question="이미지에서 무엇이 보입니까?",
        ),
    ]
    summary = evaluate_deterministic(rows)
    codes = {issue.code for issue in summary.issues}
    assert "duplicate_question" in codes
    assert "paraphrase_question" in codes
    assert "empty_field" in codes
    assert "unsupported_spectral_claim" in codes
    assert "text_only_visual_source_leakage" in codes
    assert "pair_count_mismatch" in codes


def test_checks_required_matrix_cells() -> None:
    rows = [_row(pair_index=index) for index in range(1, 5)]
    summary = evaluate_deterministic(
        rows,
        require_configs=["C01", "C02"],
        require_samples=["sample-1", "sample-2"],
        require_pairs=4,
    )
    missing = [issue for issue in summary.issues if issue.code == "missing_matrix_cell"]
    assert {(issue.config_id, issue.sample_id) for issue in missing} == {
        ("C01", "sample-2"),
        ("C02", "sample-1"),
        ("C02", "sample-2"),
    }


def test_manual_review_sheet_has_blank_subjective_fields() -> None:
    rows = [_row(pair_index=index) for index in range(1, 5)]
    sheet = build_manual_review_sheet(rows)
    assert isinstance(sheet, pd.DataFrame)
    for column in (
        "correctness",
        "grounding",
        "hallucination",
        "diversity",
        "visual_dependence",
        "usefulness",
        "specificity",
    ):
        assert column in sheet
        assert sheet[column].eq("").all()
