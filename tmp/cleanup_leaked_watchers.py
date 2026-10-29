#!/usr/bin/env python
"""Kill leaked play_v24.sh watchers (and their UI children) WITHOUT
touching the live V24 chain.

Why this exists
---------------
`tmp/test_play_v24_watcher.py` starts a real watcher per test case. On
Windows the watcher's EXIT trap does not always fire (the test kills the
process group; a bash `trap EXIT` inside a killed group can be skipped),
so each test run leaks a `play_v24.sh` bash process and a `ui_server`
python process forever.

Measured on 2026-10-29: 33 leaked processes, 1107 CPU-seconds burnt and
1.2 GB RSS -- while an 18-hour training run was competing for the CPU.
That is exactly the contention pattern that made the live-parity tests
fail at random earlier (see the ui-server-parity-flakiness-windows skill).

How the live chain is protected
-------------------------------
NOT by environment variables -- psutil's `environ()` returns None for
V24_TAG on these Windows bash processes, and a naive "does the cmdline
mention play_v24.sh" test even matches the cleanup script's OWN command
line (and any diagnostic command that greps for it). Both approaches are
unreliable.

Instead: find the LISTEN sockets, and protect the full ancestry of the
server that owns the LIVE port. Everything else that is a watcher/UI is
a leak.

Usage:
  ./.venv-c4/Scripts/python.exe tmp/cleanup_leaked_watchers.py [--dry-run]
"""
import sys
import time

import psutil

LIVE_PORTS = {7860}


def cmdline(p):
    try:
        return p.cmdline() or []
    except Exception:
        return []


def is_watcher(p):
    j = " ".join(cmdline(p))
    return "play_v24.sh" in j and "bash" in p.name().lower()


def is_ui(p):
    cl = cmdline(p)
    return "ui_server" in " ".join(cl) and "-c" not in cl[:2]


def ancestry(pid):
    chain = []
    cur = pid
    while cur:
        try:
            p = psutil.Process(cur)
        except Exception:
            break
        chain.append(cur)
        cur = p.ppid()
    return chain


def main():
    dry = "--dry-run" in sys.argv

    # ---- 1. find the live server(s) and protect their ancestry ----------
    protected = set()
    live_servers = []
    for c in psutil.net_connections(kind="inet"):
        if c.status != "LISTEN" or not c.pid:
            continue
        if c.laddr.port not in LIVE_PORTS:
            continue
        try:
            p = psutil.Process(c.pid)
        except Exception:
            continue
        j = " ".join(cmdline(p))
        if "ui_server" not in j:
            continue
        live_servers.append((c.laddr.port, c.pid))
        protected.update(ancestry(c.pid))

    print(f"live servers on {sorted(LIVE_PORTS)}: {live_servers}")
    if not live_servers:
        print("WARNING: no live server found -- refusing to guess. Aborting.")
        return 1
    print(f"protecting {len(protected)} process(es) in the live ancestry")

    # ---- 2. collect leaks ------------------------------------------------
    leaks = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if p.pid in protected:
                continue
            if not (is_watcher(p) or is_ui(p)):
                continue
            leaks.append(p)
        except Exception:
            pass

    if not leaks:
        print("no leaked watchers found")
        return 0

    print(f"found {len(leaks)} leaked watcher/UI process(es)")
    if dry:
        for p in leaks:
            print(f"  WOULD KILL {p.pid} {p.name()} {' '.join(cmdline(p))[:70]}")
        return 0

    killed = []
    for p in leaks:
        try:
            kids = p.children(recursive=True)
        except Exception:
            kids = []
        for t in kids + [p]:
            try:
                if not (is_ui(t) or is_watcher(t) or "timeout" in t.name().lower()):
                    continue
                t.kill()
                killed.append((t.pid, "ui" if is_ui(t) else "watcher"))
            except Exception:
                pass

    time.sleep(1.5)
    alive = [(pid, k) for pid, k in killed if psutil.pid_exists(pid)]
    print(f"killed {len(killed)} process(es)")
    if alive:
        print(f"WARNING: still alive: {alive}", file=sys.stderr)
        return 1

    # ---- 3. verify the live chain survived ------------------------------
    still = psutil.pid_exists(live_servers[0][1])
    print(f"live server pid {live_servers[0][1]} alive after cleanup: {still}")
    if not still:
        print("FATAL: the live UI server was killed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
