from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_required_project_skeleton_exists() -> None:
    """Catches removal or omission of a required reproducibility entry point."""
    required = [
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "requirements.in",
        ROOT / "requirements-dev.in",
        ROOT / "requirements.lock",
        ROOT / "configs" / "local_paths.example.yaml",
        ROOT / "src" / "hsi_vqagen" / "__init__.py",
        ROOT / "docs" / "environment.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert not missing, f"missing required project files: {missing}"


def test_gitignore_excludes_private_and_large_runtime_artifacts() -> None:
    """Catches accidental tracking of credentials, paths, caches, or model weights."""
    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    required_patterns = {
        ".venv/",
        "configs/local_paths.yaml",
        "outputs/",
        ".cache/",
        ".env",
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*credentials*",
    }
    missing = sorted(pattern for pattern in required_patterns if pattern not in ignore_text)
    assert not missing, f"missing security ignore patterns: {missing}"


def test_tracked_examples_are_portable_and_secret_free() -> None:
    """Catches leakage of this server's paths or common token prefixes."""
    paths = [ROOT / "README.md", ROOT / "configs" / "local_paths.example.yaml"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths if path.exists())
    assert "/home/khkim/" not in text
    assert "/data/jypark/" not in text
    assert not re.search(r"(?:ghp_|olp_|hf_)[A-Za-z0-9_-]{10,}", text)
