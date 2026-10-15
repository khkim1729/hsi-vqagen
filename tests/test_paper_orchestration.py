import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_orchestrator_plan_lists_all_checkpoint_groups_without_gpu_use(tmp_path: Path) -> None:
    """Catches a deferred run omitting a condition or creating runtime files in plan mode."""
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_paper_studies.sh"), "--plan", "paper50"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "C01 C10" in result.stdout
    assert "C02 C11" in result.stdout
    assert "C03 C09" in result.stdout
    assert "C04 C06" in result.stdout
    assert not list(tmp_path.rglob("server.pid"))


def test_robustness_orchestrator_plan_lists_paired_shuffled_and_anchor(tmp_path: Path) -> None:
    """Catches a stress test that skips the description-only anchor."""
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_paper_studies.sh"), "--plan", "robustness"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "R01 R02 R03" in result.stdout
    assert "R04 R05 R06" in result.stdout
