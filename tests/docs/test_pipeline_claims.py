import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "hsi_description_pipeline.md"
EVIDENCE = ROOT / "artifacts" / "hsi_description_pipeline_evidence.json"

REQUIRED_CLAIMS = {
    "input",
    "preprocessing",
    "rgb_rendering",
    "index_selection_and_maps",
    "sam_kmedoids_clustering",
    "map_agents",
    "web_context",
    "catalog_synthesis",
    "reduce_synthesis",
    "output",
    "disabled_pixel_sam_and_tetracorder",
    "limitations",
}


def test_every_method_claim_has_traceable_evidence() -> None:
    """Catches a Method statement that cannot be traced to source or artifacts."""
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    claims = payload["claims"]
    assert set(claims) == REQUIRED_CLAIMS
    for claim, entries in claims.items():
        assert entries, f"claim has no evidence: {claim}"
        for entry in entries:
            assert entry["source_root"] in {"production", "legacy", "dataset"}
            assert entry["source_path"]
            assert not Path(entry["source_path"]).is_absolute()
            assert entry["symbol_or_field"]


def test_pipeline_document_separates_active_and_disabled_components() -> None:
    """Catches overclaiming optional legacy capabilities as production stages."""
    text = DOC.read_text(encoding="utf-8")
    for heading in (
        "## Input and preprocessing",
        "## RGB and spectral-index evidence",
        "## Agent workflow",
        "## Description synthesis and output",
        "## Disabled components",
        "## Limitations",
    ):
        assert heading in text
    assert "pixel-level supervised SAM observation was disabled" in text
    assert "Tetracorder material identification was disabled" in text
    assert "raw HSI cubes are not inputs to the VQA-generation models" in text
    assert "pixel-level supervised SAM generated the 10,000 descriptions" not in text
    assert "Tetracorder generated the 10,000 descriptions" not in text


def test_realized_prompt_and_model_roles_are_recorded() -> None:
    """Catches drift from the metadata attached to the audited 10k run."""
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert payload["prompt_set"] == {
        "version": "core-config-prompts@2026-07-30",
        "hash": "0e88f131955c0f41",
    }
    assert payload["model_roles"] == {
        "catalog": "gpt-5.4-mini",
        "cluster": "gpt-5.4-mini",
        "catalog_synthesis": "gpt-5.4",
        "reduce": "gpt-5.4",
        "web": "gpt-5.4-mini",
    }
    assert payload["batch_waves"] == {
        "realtime_side_channel": ["web_research"],
        "wave_1": ["catalog:*", "cluster"],
        "wave_2": ["catalog_synthesis"],
        "wave_3": ["reduce"],
    }
