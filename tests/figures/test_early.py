import json

from hsi_vqagen.figures.early import load_first_pairs, truncate


def test_truncate_preserves_short_text_and_marks_long_text() -> None:
    assert truncate("short", 10) == "short"
    assert truncate("a very long sentence", 10) == "a very..."


def test_load_first_pairs_indexes_config_sample_and_first_pair(tmp_path) -> None:
    output = tmp_path / "model"
    output.mkdir()
    rows = [
        {"config_id": "C01", "sample_id": "s1", "pair_index": 2, "question": "q2"},
        {"config_id": "C01", "sample_id": "s1", "pair_index": 1, "question": "q1"},
    ]
    (output / "normalized.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    indexed = load_first_pairs(tmp_path)

    assert indexed[("C01", "s1")]["question"] == "q1"
