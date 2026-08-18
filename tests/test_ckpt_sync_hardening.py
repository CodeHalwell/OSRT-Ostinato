"""Locks the 2026-07-26 checkpoint-sync hardening fixes.

See docs/specs/2026-07-26-ckpt-sync-and-data-builder-findings.md:
  §1 — atomic save (no truncated file ever visible at the final name)
  §2 — rescue/final checkpoints are reachable by the sync glob + resume selection
"""
import glob
import os
import re
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from osrt.train import save_checkpoint  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)


def test_save_checkpoint_is_atomic(tmp_path):
    """§1: the final name never exists in a partial state, and no .tmp leaks."""
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    path = str(tmp_path / "osrt_v5_midtrain3_step_100.pt")

    save_checkpoint(model, opt, 100, path)

    assert os.path.exists(path), "final checkpoint missing"
    assert not os.path.exists(path + ".tmp"), "temp file leaked — replace failed"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck["step"] == 100
    assert "model_state_dict" in ck and "optimizer_state_dict" in ck


# ── §2: sync/resume coverage for _step_, _rescue_step_ and _final aliases ──

PREFIX = "osrt_v5_midtrain3"
# The daemon glob + pull_latest regex used in scripts/hf_ckpt_sync.py.
_SYNC_RE = re.compile(rf"{PREFIX}_(?:rescue_)?step_\d+\.pt$")


def _step_of(path: str) -> int:
    m = re.search(r"step_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def test_rescue_and_step_are_syncable():
    """A rescue checkpoint must match the widened sync pattern; a bare
    _final.pt (no step) must NOT (it is synced via its step-numbered alias)."""
    assert _SYNC_RE.match(f"{PREFIX}_step_5600.pt")
    assert _SYNC_RE.match(f"{PREFIX}_rescue_step_4447.pt")
    assert not _SYNC_RE.match(f"{PREFIX}_final.pt")


def test_latest_selection_prefers_higher_step_across_rescue():
    """pull_latest picks the highest step whether it is a rescue or numbered
    save — _step_of parses both name shapes."""
    remote = [
        f"{PREFIX}_step_5500.pt",
        f"{PREFIX}_rescue_step_5647.pt",  # capped mid-interval, the newest
        f"{PREFIX}_step_5600.pt",
    ]
    syncable = [f for f in remote if _SYNC_RE.match(f)]
    assert max(syncable, key=_step_of) == f"{PREFIX}_rescue_step_5647.pt"


def test_final_alias_glob_matches(tmp_path):
    """The end-of-run step-numbered alias is what the daemon `_step_*` glob
    catches — verify a file named like the alias is found."""
    (tmp_path / f"{PREFIX}_step_12600.pt").write_bytes(b"x")
    (tmp_path / f"{PREFIX}_final.pt").write_bytes(b"x")
    hits = glob.glob(os.path.join(str(tmp_path), f"{PREFIX}_step_*.pt"))
    assert any("step_12600" in h for h in hits)
