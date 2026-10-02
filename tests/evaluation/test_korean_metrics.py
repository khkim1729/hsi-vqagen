from hsi_vqagen.evaluation.korean import (
    evaluate_korean_outputs,
    has_korean_text,
    source_description_cosine,
)


def _row(
    *,
    config_id="C01",
    sample_id="s1",
    pair_index=1,
    question="장면을 주로 덮는 것은 무엇인가?",
    answer="짙은 녹색의 숲 수관이다.",
    evidence="짙은 녹색 목본 식생이 장면 대부분을 덮는다.",
    description="짙은 녹색의 목본 식생이 장면 대부분을 덮는다. 열린 물은 없다.",
    condition="multimodal",
):
    return {
        "config_id": config_id,
        "sample_id": sample_id,
        "pair_index": pair_index,
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "question_type": "attribute",
        "input_condition": condition,
        "latency_seconds": 2.0,
        "provenance": {"description": description},
    }


def test_has_korean_text_requires_hangul() -> None:
    assert has_korean_text("짙은 숲 수관")
    assert not has_korean_text("dense forest canopy")


def test_source_description_cosine_rewards_grounded_korean_text() -> None:
    description = "짙은 녹색의 목본 식생이 장면 대부분을 덮는다. 열린 물은 없다."

    grounded = source_description_cosine(
        "짙은 녹색 목본 식생이 대부분을 덮는다", description
    )
    unrelated = source_description_cosine("밝은 건물과 넓은 도로가 있다", description)

    assert grounded > 0.45
    assert grounded > unrelated


def test_korean_summary_reports_compliance_grounding_diversity_and_warnings() -> None:
    rows = [_row(pair_index=index) for index in range(1, 5)]
    rows[1] = _row(
        pair_index=2,
        question="What is visible?",
        answer="A forest canopy",
        evidence="visible in the image",
    )

    summary = evaluate_korean_outputs(rows)[0]

    assert summary.config_id == "C01"
    assert summary.case_count == 1
    assert summary.pair_count == 4
    assert summary.korean_complete_pair_rate == 0.75
    assert 0 <= summary.mean_source_description_cosine <= 1
    assert 0 <= summary.mean_question_diversity <= 1
    assert summary.direct_visual_access_warning_count == 0


def test_numeric_only_answer_is_valid_korean_task_output() -> None:
    rows = [_row(pair_index=index) for index in range(1, 5)]
    rows[0] = _row(pair_index=1, answer="43%")

    summary = evaluate_korean_outputs(rows)[0]

    assert summary.korean_complete_pair_rate == 1.0


def test_text_only_visual_access_is_counted() -> None:
    rows = [
        _row(
            config_id="C10",
            condition="text_only",
            pair_index=index,
            question="이미지에서 보이는 수관은 무엇인가?" if index == 1 else f"질문 {index}",
        )
        for index in range(1, 5)
    ]

    summary = evaluate_korean_outputs(rows)[0]

    assert summary.direct_visual_access_warning_count == 1
