from pathlib import Path

from hsi_vqagen.data.records import load_dataset_records


def test_records_pair_description_and_rgb_by_sample_id(paired_shards: Path) -> None:
    """Catches positional pairing when descriptions and SFT rows are reordered."""
    records = load_dataset_records(paired_shards)

    assert [record.sample_id for record in records] == ["sample-a", "sample-b"]
    assert records[0].description == "short description"
    assert records[0].rgb_path.name == "sample-a.png"
    assert records[0].hsi_shape == (64, 64, 426)
    assert records[0].rgb_size == (8, 6)
