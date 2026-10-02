from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVE_SCRIPT = ROOT / "scripts" / "serve_model.sh"


def test_server_subprocess_can_find_venv_build_tools() -> None:
    """FlashInfer JIT subprocesses must resolve the venv's ninja executable."""

    text = SERVE_SCRIPT.read_text(encoding="utf-8")
    assert 'PATH="$repo_root/.venv/bin:$PATH"' in text
    setsid_line = next(line for line in text.splitlines() if line.startswith("setsid env"))
    assert 'PATH="$repo_root/.venv/bin:$PATH"' in setsid_line
