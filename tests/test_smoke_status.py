import json
from pathlib import Path

from scripts.smoke_status import group_is_complete


def test_group_is_complete_only_for_valid_korean_smoke_manifests(tmp_path: Path) -> None:
    output_dirs = {"C01": "qwen", "C10": "qwen_text"}
    for config_id, output_dir in output_dirs.items():
        target = tmp_path / output_dir
        target.mkdir()
        (target / "run_manifest.json").write_text(
            json.dumps(
                {
                    "config_id": config_id,
                    "prompt_version": "vqa-generation-ko-v1",
                    "completed_sample_ids": ["sample-1"],
                    "failed_sample_ids": [],
                }
            ),
            encoding="utf-8",
        )

    assert group_is_complete(tmp_path, output_dirs)
    payload = json.loads((tmp_path / "qwen_text" / "run_manifest.json").read_text())
    payload["prompt_version"] = "vqa-generation-v1"
    (tmp_path / "qwen_text" / "run_manifest.json").write_text(json.dumps(payload))
    assert not group_is_complete(tmp_path, output_dirs)
