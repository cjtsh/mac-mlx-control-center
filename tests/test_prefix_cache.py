#!/usr/bin/env python3
"""Prefix-cache (mlx-vlm APC) wiring: env built for a launch, status for the dashboard.

No model and no network needed — the cache-endpoint probe points at a closed port.
Run: python3 tests/test_prefix_cache.py
"""
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import mlx_helper as h  # noqa: E402

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")


def run():
    tmp = tempfile.mkdtemp(prefix="pc_test_")
    settings = os.path.join(tmp, "settings.json")
    real_settings = h.SETTINGS_FILE
    saved_env = {k: os.environ.get(k) for k in ("MLX_VLM_APC", "APC_ENABLED", "APC_NUM_BLOCKS")}
    try:
        for k in saved_env:
            os.environ.pop(k, None)
        h.SETTINGS_FILE = settings

        # ---- defaults -----------------------------------------------------
        cfg = h.prefix_cache_settings()
        check("defaults: on, 4096 blocks, 20 GB disk", cfg["enabled"] and cfg["blocks"] == 4096
              and cfg["disk_gb"] == 20.0, cfg)
        check("defaults: 4 snapshots in RAM", cfg["entries"] == 4, cfg)
        check("defaults: 16 GB cache ceiling", cfg["memory_gb"] == 16.0, cfg)

        env = h.apc_env({}, engine="mlx-vlm")
        check("vlm launch: APC_ENABLED=1", env.get("APC_ENABLED") == "1", env)
        check("vlm launch: block size 16", env.get("APC_BLOCK_SIZE") == "16", env)
        check("vlm launch: num blocks from settings", env.get("APC_NUM_BLOCKS") == "4096", env)
        check("vlm launch: cache home under ~/.cache/mlx-vlm",
              env.get("MLX_VLM_CACHE_HOME", "").endswith(".cache/mlx-vlm"), env)
        check("vlm launch: disk tier capped", env.get("APC_DISK_MAX_GB") == "20.0", env)
        check("vlm launch: does not disable disk tier", "APC_DISK_ENABLED" not in env, env)
        check("vlm launch: snapshot count raised above mlx-vlm's 2",
              env.get("APC_EXACT_CACHE_ENTRIES") == "4", env)
        check("vlm launch: memory ceiling raised above mlx-vlm's 8 GB",
              env.get("APC_MEMORY_MAX_GB") == "16.0", env)

        # ---- mlx-lm is left alone (it has its own prompt cache) -----------
        lm = h.apc_env({"HF_HUB_CACHE": "/x"}, engine="mlx-lm")
        check("lm launch: untouched", lm == {"HF_HUB_CACHE": "/x"}, lm)

        # ---- explicit env wins -------------------------------------------
        kept = h.apc_env({"APC_NUM_BLOCKS": "512", "APC_DISK_MAX_GB": "4"}, engine="mlx-vlm")
        check("explicit APC_NUM_BLOCKS preserved", kept.get("APC_NUM_BLOCKS") == "512", kept)
        check("explicit APC_DISK_MAX_GB preserved", kept.get("APC_DISK_MAX_GB") == "4", kept)

        # ---- opt-outs -----------------------------------------------------
        os.environ["MLX_VLM_APC"] = "0"
        check("MLX_VLM_APC=0 disables", "APC_ENABLED" not in h.apc_env({}, engine="mlx-vlm"))
        os.environ.pop("MLX_VLM_APC", None)
        check("MLX_VLM_APC=0 via env dict disables",
              "APC_ENABLED" not in h.apc_env({"MLX_VLM_APC": "0"}, engine="mlx-vlm"))

        with open(settings, "w") as fh:
            json.dump({"prefix_cache": False}, fh)
        check("settings prefix_cache=false disables", "APC_ENABLED" not in h.apc_env({}, engine="mlx-vlm"))

        with open(settings, "w") as fh:
            json.dump({"prefix_cache": "off"}, fh)
        check("string \"off\" also disables", "APC_ENABLED" not in h.apc_env({}, engine="mlx-vlm"))

        with open(settings, "w") as fh:
            json.dump({"prefix_cache_blocks": 1024, "prefix_cache_disk_gb": 0}, fh)
        env = h.apc_env({}, engine="mlx-vlm")
        check("custom block count used", env.get("APC_NUM_BLOCKS") == "1024", env)
        check("disk_gb 0 disables the disk tier", env.get("APC_DISK_ENABLED") == "0", env)
        check("disk_gb 0 omits the cap", "APC_DISK_MAX_GB" not in env, env)

        with open(settings, "w") as fh:
            json.dump({"prefix_cache_blocks": 0}, fh)
        check("bogus block count falls back",
              h.prefix_cache_settings()["blocks"] == h.APC_DEFAULT_BLOCKS)

        with open(settings, "w") as fh:
            json.dump({"prefix_cache_entries": 8, "prefix_cache_memory_gb": 32}, fh)
        env = h.apc_env({}, engine="mlx-vlm")
        check("custom snapshot count used", env.get("APC_EXACT_CACHE_ENTRIES") == "8", env)
        check("custom memory ceiling used", env.get("APC_MEMORY_MAX_GB") == "32.0", env)

        with open(settings, "w") as fh:
            json.dump({"prefix_cache_memory_gb": 0}, fh)
        env = h.apc_env({}, engine="mlx-vlm")
        check("memory_gb 0 leaves mlx-vlm's own ceiling", "APC_MEMORY_MAX_GB" not in env, env)

        # ---- status -------------------------------------------------------
        info = h.apc_status(None)
        check("status: reports configuration", info["enabled"] is True and info["blocks"] >= 1, info)
        check("status: not running yet", info["server_running"] is False, info)

        closed = h.apc_status(9, timeout=0.4)  # discard port: almost certainly closed
        check("status: closed port does not raise", isinstance(closed, dict) and closed.get("server_running") is False,
              closed)
        check("status: closed port records an error", "error" in closed, closed)
    finally:
        h.SETTINGS_FILE = real_settings
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
