"""Auto-imported by CPython at interpreter start when this directory is on PYTHONPATH.

The MLX launchers put it there on purpose: it is the only hook that reaches a
server started as ``python -m mlx_vlm.server`` without changing its command line,
its port or its pid — so every existing health check, pid file and log keeps
working. It does nothing unless ``MLX_STREAM_GUARD=1`` is set for that process.
"""

import os

try:
    if os.environ.get("MLX_STREAM_GUARD", "1").strip() != "0":
        try:
            import stream_guard

            stream_guard.install()
        except Exception:  # never stop a model server from booting
            pass
except Exception:
    pass
