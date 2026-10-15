#!/usr/bin/env python3
"""Linux-only, read-only socket/process checks for a newly launched local server."""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path


def _listener_inodes(port: int) -> set[str]:
    found: set[str] = set()
    for name in ("tcp", "tcp6"):
        try:
            lines = Path("/proc/net", name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                found.add(fields[9])
    return found


def port_is_available(port: int) -> bool:
    if not 1 <= port <= 65535:
        raise ValueError("port must be 1-65535")
    if _listener_inodes(port):
        return False
    for family, address in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                if family == socket.AF_INET6:
                    probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                probe.bind((address, port))
        except OSError as exc:
            if family == socket.AF_INET6 and exc.errno in {97, 93}:
                continue  # IPv6 is unavailable on this host.
            return False
    return True


def listener_belongs_to_group(port: int, pgid: int) -> bool:
    """Require the listening socket FD to be held by this new session's group."""
    inodes = {f"socket:[{inode}]" for inode in _listener_inodes(port)}
    if not inodes:
        return False
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) != pgid:
                continue
            for fd in (entry / "fd").iterdir():
                try:
                    if os.readlink(fd) in inodes:
                        return True
                except OSError:
                    continue
        except (OSError, PermissionError, ProcessLookupError):
            continue
    return False


def process_start_ticks(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]


def same_launched_server(pid: int, start_ticks: str, model: str, revision: str) -> bool:
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        parts = [item.decode(errors="replace") for item in argv if item]
        return (os.getpgid(pid) == pid and process_start_ticks(pid) == start_ticks
                and "serve" in parts and model in parts and revision in parts
                and any("vllm" in part for part in parts)
                and any("server_supervisor.py" in part for part in parts))
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--assert-owned", action="store_true")
    mode.add_argument("--start-ticks", action="store_true")
    mode.add_argument("--same-launch", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--pgid", type=int)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--expected-start-ticks")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.preflight:
        if args.port is None or not port_is_available(args.port):
            raise SystemExit("requested server port is already occupied")
    elif args.assert_owned:
        if args.port is None or args.pgid is None or not listener_belongs_to_group(args.port, args.pgid):
            raise SystemExit("listener is not owned by newly launched server group")
    elif args.start_ticks:
        if args.pid is None:
            parser.error("--pid is required")
        print(process_start_ticks(args.pid))
    elif args.same_launch:
        if (args.pid is None or args.expected_start_ticks is None
                or args.model is None or args.revision is None
                or not same_launched_server(args.pid, args.expected_start_ticks, args.model, args.revision)):
            raise SystemExit("server PID no longer identifies this launch")


if __name__ == "__main__":
    main()
