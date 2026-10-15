import os
import socket
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVE_SCRIPT = ROOT / "scripts" / "serve_model.sh"


def test_server_resolves_explicit_existing_runtime_without_starting_gpu(tmp_path: Path) -> None:
    """Catches commands binding to a missing main-worktree venv during paper runs."""
    runtime = tmp_path / "shared_venv"
    env = os.environ.copy()
    env["HSI_VQAGEN_VENV"] = str(runtime)

    result = subprocess.run(
        ["bash", str(SERVE_SCRIPT), "--show-runtime"],
        capture_output=True, text=True, env=env, check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == str(runtime)
    assert not list(tmp_path.rglob("server.pid"))


def test_occupied_port_is_rejected_without_starting_server() -> None:
    """An older compatible endpoint must never be accepted as this run's child."""
    from scripts.server_process import port_is_available

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        assert not port_is_available(listener.getsockname()[1])


def test_bound_but_not_listening_port_is_rejected() -> None:
    from scripts.server_process import port_is_available

    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        assert not port_is_available(bound.getsockname()[1])


def test_listener_must_belong_to_new_process_group() -> None:
    from scripts.server_process import listener_belongs_to_group

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert listener_belongs_to_group(port, os.getpgrp())
        assert not listener_belongs_to_group(port, 99999999)


def test_model_revision_is_only_verified_when_endpoint_exposes_it() -> None:
    from scripts.check_server import verify_model_revision

    revision = "a" * 40
    assert verify_model_revision({"id": "checkpoint"}, revision) == (False, None)
    assert verify_model_revision({"id": "checkpoint", "revision": revision}, revision) == (True, revision)
    try:
        verify_model_revision({"id": "checkpoint", "revision": "b" * 40}, revision)
    except ValueError as exc:
        assert "revision mismatch" in str(exc)
    else:
        raise AssertionError("mismatched served revision was accepted")


def test_supervisor_cleans_group_if_server_leader_exits(monkeypatch) -> None:
    """All subprocess and signal operations are mocked; no process is stopped."""
    from scripts import server_supervisor

    calls = []

    class Child:
        def wait(self):
            calls.append("wait")
            return 1

    monkeypatch.setattr(server_supervisor.subprocess, "Popen",
                        lambda command: calls.append(tuple(command)) or Child())
    monkeypatch.setattr(server_supervisor.os, "getpgrp", lambda: 123456)
    monkeypatch.setattr(server_supervisor.os, "getpid", lambda: 123456)
    monkeypatch.setattr(server_supervisor.os, "killpg",
                        lambda pgid, sig: calls.append(("killpg", pgid, sig)))
    monkeypatch.setattr(server_supervisor.signal, "signal", lambda *args: None)
    assert server_supervisor.supervise(["fake", "serve"]) == 1
    assert "wait" in calls
    assert any(item[0] == "killpg" and item[1] == 123456 for item in calls if isinstance(item, tuple))


def test_supervisor_escalates_only_its_own_lingering_group(monkeypatch) -> None:
    """Mocked process API verifies bounded cleanup without terminating anything."""
    from scripts import server_supervisor

    signals = []

    class Child:
        def wait(self):
            return 1

    monkeypatch.setattr(server_supervisor.subprocess, "Popen", lambda _command: Child())
    monkeypatch.setattr(server_supervisor.os, "getpgrp", lambda: 123456)
    monkeypatch.setattr(server_supervisor.os, "getpid", lambda: 123456)
    monkeypatch.setattr(server_supervisor.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(server_supervisor.signal, "signal", lambda *args: None)
    monkeypatch.setattr(server_supervisor, "other_group_members", lambda _pgid: [654321])
    monkeypatch.setattr(server_supervisor.time, "monotonic", lambda: 100.0)
    assert server_supervisor.supervise(["fake", "serve"], cleanup_timeout=0) == 1
    assert signals == [
        (123456, server_supervisor.signal.SIGTERM),
        (123456, server_supervisor.signal.SIGKILL),
    ]
