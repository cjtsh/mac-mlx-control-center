"""Checks that the reply budget scales with the loaded model (no model needed).

Run: python3 tests/test_token_budget.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mlx_helper  # noqa: E402

FAILS = []


def check(name, got, want):
    if got == want:
        print("  ok   %s -> %s" % (name, got))
    else:
        print("  FAIL %s -> %s (wanted %s)" % (name, got, want))
        FAILS.append(name)


def budget(ctx, requested=None):
    mlx_helper.effective_context_window = lambda _m: ctx
    return mlx_helper.output_token_budget("stub/model", requested)


def main():
    print("output_token_budget checks")
    check("288B-class window (262144) is not capped small", budget(262144), 32768)
    check("131072 window", budget(131072), 32768)
    check("65536 window", budget(65536), 16384)
    check("32768 window", budget(32768), 8192)
    check("8192 window keeps a usable floor", budget(8192), 4096)
    check("no window info falls back to 8192", budget(0), 8192)
    check("explicit setting wins", budget(262144, 4096), 4096)
    check("explicit setting can exceed auto", budget(65536, 16384), 16384)
    check("explicit setting clamps to the model", budget(32768, 100000), 32768)
    check("small explicit value is respected", budget(262144, 512), 512)

    real = mlx_helper.model_context_window("sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit")
    print("  info installed REAP model reports context_window=%s" % real)
    print("\n%d failure(s)" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
