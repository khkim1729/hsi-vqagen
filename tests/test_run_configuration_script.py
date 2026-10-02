from types import SimpleNamespace

from scripts.run_configuration import _exit_code


def test_run_configuration_returns_failure_when_any_case_failed() -> None:
    assert _exit_code(SimpleNamespace(failed=1)) == 1
    assert _exit_code(SimpleNamespace(failed=0)) == 0
