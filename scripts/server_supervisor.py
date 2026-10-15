#!/usr/bin/env python3
"""Keep a launched vLLM process group owned until its workers are signalled."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def other_group_members(pgid: int) -> list[int]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            if os.getpgid(pid) == pgid:
                members.append(pid)
        except (OSError, ProcessLookupError):
            continue
    return members


def supervise(command: list[str], *, cleanup_timeout: float = 30.0) -> int:
    if not command:
        raise ValueError("server command is required")
    if os.getpgrp() != os.getpid():
        raise RuntimeError("server supervisor must lead a new process group")
    stopping = False

    def stop_group(_signum: int, _frame: object) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        # The supervisor remains group leader while it signals its own workers.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        os.killpg(os.getpgrp(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop_group)
    signal.signal(signal.SIGINT, stop_group)
    try:
        child = subprocess.Popen(command)
        return child.wait()
    finally:
        # Also clean up workers if the vLLM parent exits without a signal.
        stop_group(signal.SIGTERM, None)
        deadline = time.monotonic() + cleanup_timeout
        while other_group_members(os.getpgrp()):
            if time.monotonic() >= deadline:
                # The supervisor still owns this new session, so the group
                # cannot have been reused by an unrelated process.
                os.killpg(os.getpgrp(), signal.SIGKILL)
                break
            time.sleep(0.25)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    raise SystemExit(supervise(argv))
