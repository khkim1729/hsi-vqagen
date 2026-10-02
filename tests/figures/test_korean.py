import warnings

from PIL import Image

from hsi_vqagen.figures.korean import (
    CONTROLLED_TEXT_ONLY_COLUMNS,
    PRIMARY_TEXT_ONLY_COLUMNS,
    make_korean_grid,
)


def test_text_only_figures_split_cross_model_and_controlled_conditions() -> None:
    assert [config_id for _, config_id in PRIMARY_TEXT_ONLY_COLUMNS] == [
        "C05", "C06", "C07", "C08"
    ]
    assert [config_id for _, config_id in CONTROLLED_TEXT_ONLY_COLUMNS] == [
        "C10", "C11", "C09", "C06"
    ]


def test_korean_grid_includes_source_description_and_writes_png_pdf(tmp_path) -> None:
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (32, 32), color=(20, 80, 30)).save(image_path)
    sample_id = "sample-1"
    rows = {
        ("C01", sample_id): {
            "question": "장면을 덮는 것은 무엇인가?",
            "answer": "숲 수관이다.",
            "provenance": {"description": "짙은 녹색의 숲 수관이 장면 대부분을 덮는다."},
        }
    }

    with warnings.catch_warnings(record=True) as caught:
        make_korean_grid(
            rows=rows,
            samples=[(sample_id, image_path)],
            columns=(("시험 모델", "C01"),),
            title="한국어 VQA 비교",
            stem="test_grid",
            output_dir=tmp_path,
        )

    assert (tmp_path / "test_grid.png").stat().st_size > 1_000
    assert (tmp_path / "test_grid.pdf").stat().st_size > 1_000
    assert not [warning for warning in caught if "Glyph" in str(warning.message)]
