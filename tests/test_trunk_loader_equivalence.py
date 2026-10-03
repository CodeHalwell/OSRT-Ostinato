"""main + router_bias_in_gates=True must compute the pinned trunk's forward.

Runs scripts/check_trunk_loader.py, which builds the same tiny model from each
tree's OSRT_V7 preset, perturbs the router balancing bias (a fresh model's
zero bias hides the difference), copies the pinned weights into main and
compares logits/loss in eval mode. Skipped where the pinned worktree is absent.
"""
import os
import subprocess
import sys

import pytest

PINNED = os.path.expanduser("~/osrt-trunk-pinned")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(not os.path.isdir(os.path.join(PINNED, "src")),
                    reason="pinned trunk worktree not present")
def test_legacy_gates_build_reproduces_pinned_forward():
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "check_trunk_loader.py"),
         "--pinned", PINNED],
        env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")},
        capture_output=True, text=True, timeout=900,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "router_bias_in_gates=True " in r.stdout and "MATCH" in r.stdout
    assert "router_bias_in_gates=False" in r.stdout and "DIFFERS" in r.stdout
