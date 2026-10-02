import pytest

from hsi_vqagen.evaluation.ablation import compare_modality_ablation


PAIRS = (("C01", "C10"), ("C02", "C11"), ("C03", "C09"), ("C04", "C06"))


def _rows(config_id: str, samples=("s1", "s2")) -> list[dict]:
    return [
        {
            "config_id": config_id,
            "sample_id": sample,
            "pair_index": pair_index,
            "question": f"{config_id} question {pair_index}",
            "answer": f"answer {pair_index}",
            "evidence": f"evidence {pair_index}",
            "question_type": "spatial",
        }
        for sample in samples
        for pair_index in range(1, 5)
    ]


@pytest.mark.parametrize(("multimodal_id", "text_only_id"), PAIRS)
def test_all_registered_ablation_pairs_match_by_sample_and_pair_index(
    multimodal_id: str, text_only_id: str
) -> None:
    report = compare_modality_ablation(
        _rows(multimodal_id), _rows(text_only_id)
    )
    assert report.multimodal_config_id == multimodal_id
    assert report.text_only_config_id == text_only_id
    assert len(report.comparisons) == 8
    assert {(row.sample_id, row.pair_index) for row in report.comparisons} == {
        (sample, index) for sample in ("s1", "s2") for index in range(1, 5)
    }
    assert all(row.description_absent_visual_detail == "" for row in report.comparisons)
    assert all(row.hallucination_direction == "" for row in report.comparisons)


def test_ablation_rejects_missing_or_misaligned_pair() -> None:
    text_rows = _rows("C10")[:-1]
    with pytest.raises(ValueError, match="pairing mismatch"):
        compare_modality_ablation(_rows("C01"), text_rows)


def test_ablation_rejects_unregistered_configuration_pair() -> None:
    with pytest.raises(ValueError, match="registered modality pair"):
        compare_modality_ablation(_rows("C01"), _rows("C09"))
