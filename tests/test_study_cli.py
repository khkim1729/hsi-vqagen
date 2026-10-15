def test_study_cli_resolves_frozen_paper_and_robustness_matrices() -> None:
    """Catches an artifact run using an incomplete or mixed configuration matrix."""
    from scripts import evaluate_studies

    paper = evaluate_studies.resolve_study("paper50")
    robust = evaluate_studies.resolve_study("robustness")

    assert len(paper["expected_samples"]) == 50
    assert list(paper["config_dirs"]) == [
        "C01", "C10", "C02", "C11", "C03", "C09", "C04", "C06"
    ]
    assert paper["pairs"] == [
        ("C01", "C10"), ("C02", "C11"), ("C03", "C09"), ("C04", "C06")
    ]
    assert len(robust["expected_samples"]) == 20
    assert list(robust["config_dirs"]) == [f"R{index:02d}" for index in range(1, 7)]
    assert ("R01", "R02") in robust["pairs"]
    assert ("R04", "R05") in robust["pairs"]
    assert paper["experiment_id"] == "hsi-vqagen-icce2027-paper50-v1"
    assert robust["experiment_id"] == "hsi-vqagen-icce2027-rgb-robustness-v1"
    assert len(paper["expected_provenance"]) == 50
    assert all(len(entry["rgb_sha256"]) == 64 for entry in robust["expected_provenance"].values())
