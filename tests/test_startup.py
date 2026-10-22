#!/usr/bin/env python3
"""Guards the startup path: the dashboard must bind its port without waiting on disks.

The bug this locks down: `protect_model_storage()` ran before `ThreadedHTTPServer(...)`,
and `tmutil`/a synthetic `/Volumes` mount could block inside the kernel for minutes. The
app then looked dead — no port, no window, no error.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import mlx_helper  # noqa: E402

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  ok   {name}")
    except AssertionError as exc:
        failures.append(name)
        print(f"  FAIL {name}: {exc}")


def gui_source():
    with open(os.path.join(REPO, "mlx_gui_server.py"), encoding="utf-8") as fh:
        return fh.read()


def helper_source():
    with open(os.path.join(REPO, "mlx_helper.py"), encoding="utf-8") as fh:
        return fh.read()


def test_exclusion_runs_off_the_startup_path():
    src = gui_source()
    main = src.split('if __name__ == "__main__":', 1)[1]
    thread_at = main.find('name="mlx-tm-exclusion"')
    bind_at = main.find("ThreadedHTTPServer((")
    assert thread_at != -1, "the exclusion pass must run on its own daemon thread"
    assert bind_at != -1, "the server must still bind its port"
    assert thread_at < bind_at, "the thread must be dispatched before the port is bound"
    assert "daemon=True" in main[thread_at - 200:thread_at + 60], "thread must be a daemon"
    assert "mlx_helper.protect_model_storage()" in main
    # It may only be called from inside that thread, never inline in the main block.
    lines = [ln for ln in main.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    for i, line in enumerate(lines):
        if "mlx_helper.protect_model_storage()" in line:
            indent = len(line) - len(line.lstrip())
            assert indent >= 8, "protect_model_storage() must be nested in its thread target"
            assert lines[i - 1].strip().startswith("try:"), "the call must be guarded"
            break


def test_startup_calls_only_cheap_helpers_inline():
    main = gui_source().split('if __name__ == "__main__":', 1)[1]
    code = [ln for ln in main.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    top = [ln.strip() for ln in code if ln.startswith("    ") and not ln.startswith("     ")]
    # At four-space indent only the port bind, the print and the watchdog spawn are allowed.
    slow = [ln for ln in top
            if "subprocess" in ln or "os.walk" in ln or "tmutil" in ln or "protect_model_storage" in ln]
    assert not slow, f"startup must not shell out or walk volumes inline: {slow}"


def test_drive_discovery_skips_synthetic_mounts():
    src = helper_source()
    assert 'entry.startswith("com.apple.")' in src, \
        "synthetic /Volumes mounts must be skipped before any stat"
    seg = src.split('entries = sorted(os.listdir("/Volumes"))', 1)[1]
    order = seg.find('entry.startswith("com.apple.")'), seg.find("_volume_record(mount")
    assert -1 not in order, "both the skip and the record call must exist"
    assert order[0] < order[1], "the skip must come before the volume is stat-ed"


def test_startup_is_fast_here():
    import time
    t = time.time()
    mlx_helper.list_storage_drives(force=True)
    elapsed = time.time() - t
    assert elapsed < 5.0, f"drive discovery took {elapsed:.1f}s"


print("Startup path")
check("exclusion pass runs off the startup path", test_exclusion_runs_off_the_startup_path)
check("startup block only calls cheap helpers", test_startup_calls_only_cheap_helpers_inline)
check("synthetic /Volumes mounts are skipped", test_drive_discovery_skips_synthetic_mounts)
check("drive discovery stays fast", test_startup_is_fast_here)

if failures:
    print(f"\n{len(failures)} failure(s)")
    sys.exit(1)
print("\nall startup checks passed")
