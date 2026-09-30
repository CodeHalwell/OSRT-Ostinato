"""HF Hub checkpoint sync for ephemeral/capped GPU sessions (Colab, etc.).

A Colab session's disk is wiped when the VM is released and there's a 24h cap,
so a multi-session run must persist checkpoints off-VM. This pushes each new
`{prefix}_step_*.pt` to a private HF repo as it's saved, and pulls the latest
back on start — so the resume scan in `osrt.train.run_training` chains across
sessions.

Cross-session checkpoint sync to a private HF repo. Needs HF_TOKEN in the env;
huggingface_hub reads it itself and nothing here ever prints it.
"""
from __future__ import annotations

import glob
import os
import re
import threading
import time

# The names the sync moves: interval saves and 23h rescues, i.e.
# `{prefix}_step_N.pt` and `{prefix}_rescue_step_N.pt`. A bare `{prefix}_final.pt`
# has no step and is synced through its hard-linked `{prefix}_step_{total}.pt`
# alias. `{prefix}_failed_step_N.pt` is the record of a run that declared itself
# bad: its prefix group is "{prefix}_failed", so it never matches a sync prefix.
_SYNC_RE = re.compile(r"^(?P<prefix>.+?)_(?:rescue_)?step_(?P<step>\d+)\.pt$")

# Small files that ride along with the checkpoints so a run keeps its identity
# across venues: pulled when absent locally, pushed when absent remotely, never
# pruned. `wandb_run_id.txt` is what lets Colab re-runs continue ONE W&B run.
SIDE_FILES = ("wandb_run_id.txt",)

_PROBE_NAME = ".osrt_sync_probe"
_BACKOFF_S = (5, 15, 45)  # after failed attempt 1, 2, 3 ... (never after the last)
_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{16,}")


def _step_of(path: str) -> int:
    m = re.search(r"step_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def _is_sync_name(name: str, prefix: str) -> bool:
    """True for exactly `{prefix}_step_N.pt` / `{prefix}_rescue_step_N.pt`."""
    m = _SYNC_RE.match(name)
    return bool(m) and m.group("prefix") == prefix


def _is_protected(name: str) -> bool:
    """Never pruned: the end-of-run final and any failed-early-stop record."""
    return "_final" in name or "_failed_" in name


def _err(e: BaseException) -> str:
    """`Type: message`, truncated, with anything token-shaped scrubbed."""
    return _TOKEN_RE.sub("hf_***", f"{type(e).__name__}: {str(e)[:160]}")


def _list_remote(api, repo_id: str) -> list[str] | None:
    """The repo's file list, or None when the repo does not exist yet. Every
    other failure (401/403 bad token, 5xx, dead network) propagates: treating
    it as "no repo" would silently fork the run at step 0."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        return list(api.list_repo_files(repo_id, repo_type="model"))
    except RepositoryNotFoundError:
        return None


def _local_files(ckpt_dir: str, prefix: str, *, final: bool = False,
                 failed: bool = False) -> list[str]:
    """Local candidates, NEWEST FIRST (side files and `_final.pt` last)."""
    pats = [f"{prefix}_step_*.pt", f"{prefix}_rescue_step_*.pt"]
    if final:
        pats.append(f"{prefix}_final.pt")
    if failed:
        pats.append(f"{prefix}_failed_step_*.pt")
    found: set[str] = set()
    for pat in pats:
        found.update(glob.glob(os.path.join(ckpt_dir, pat)))
    for name in SIDE_FILES:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            found.add(p)
    return sorted(found, key=lambda p: (_step_of(p), os.path.basename(p)),
                  reverse=True)


def _obsolete_cutoff(names, prefix: str, keep_remote: int) -> int:
    """Smallest step that survives a prune of `names` to the newest
    `keep_remote` step files. Uploading anything below it would be followed
    by deleting it, so callers skip those (a Modal volume can hold every
    interval save of a run; the remote deliberately does not)."""
    steps = sorted((_step_of(n) for n in names if _is_sync_name(n, prefix)),
                   reverse=True)
    if keep_remote < 1 or len(steps) <= keep_remote:
        return -1
    return steps[keep_remote - 1]


def _upload(api, repo_id: str, path: str, name: str, attempts: int = 3) -> bool:
    """Upload one file with bounded retries (5 s, then 15 s, between the three
    attempts). Returns False when every attempt failed; never raises."""
    for i in range(attempts):
        try:
            api.upload_file(path_or_fileobj=path, path_in_repo=name,
                            repo_id=repo_id, repo_type="model")
            return True
        except Exception as e:  # noqa: BLE001 — retried, then reported, never raised
            print(f"[hf-sync] upload {name} attempt {i + 1}/{attempts} failed: "
                  f"{_err(e)}", flush=True)
            if i + 1 < attempts:
                time.sleep(_BACKOFF_S[min(i, len(_BACKOFF_S) - 1)])
    return False


def pull_latest(repo_id: str, ckpt_dir: str, prefix: str,
                base_name: str | None = None) -> str | None:
    """Download the highest {prefix}_step_*.pt / {prefix}_rescue_step_*.pt from
    the repo into ckpt_dir (so the resume scan finds it), plus any SIDE_FILES
    and `base_name` absent locally. Returns the checkpoint name pulled, or None
    for a clean start.

    Only a repo that does not exist yet starts clean. A bad token, a 5xx or a
    dead network raises (see `_list_remote`) — better a loud abort than a run
    that quietly restarts from step 0 beside its own checkpoints."""
    from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(ckpt_dir, exist_ok=True)
    api = HfApi()
    files = _list_remote(api, repo_id)
    if files is None:
        print(f"[hf-sync] {repo_id} does not exist yet; starting clean",
              flush=True)
        files = []

    for name in (*SIDE_FILES, *([base_name] if base_name else [])):
        if name in files and not os.path.exists(os.path.join(ckpt_dir, name)):
            print(f"[hf-sync] pulling {name}...", flush=True)
            hf_hub_download(repo_id, name, repo_type="model",
                            local_dir=ckpt_dir)

    # Include rescue checkpoints: the 23h-cap `_rescue_step_*.pt` is often the
    # newest artifact of a capped session, and the local resume-scan already
    # ranks it. `_step_of` matches both names. (ckpt-sync §2)
    steps = [f for f in files if _is_sync_name(f, prefix)]
    if not steps:
        print("[hf-sync] no prior checkpoints in repo — starting from base",
              flush=True)
        return None
    latest = max(steps, key=_step_of)
    if os.path.exists(os.path.join(ckpt_dir, latest)):
        print(f"[hf-sync] resuming: {latest} already local", flush=True)
        return latest
    print(f"[hf-sync] resuming: pulling {latest}...", flush=True)
    hf_hub_download(repo_id, latest, repo_type="model", local_dir=ckpt_dir)
    return latest


class PushDaemon:
    """Mirrors new checkpoints to the repo. Construction is the fail-fast part:
    it creates the repo, PROVES the token can write (a read-only token would
    otherwise surface hours in, at the first checkpoint), and seeds the
    pushed-set from the remote listing so a just-pulled file is not uploaded
    straight back. `sync_once` is one pass; `start_push_daemon` loops it."""

    def __init__(self, api, repo_id: str, ckpt_dir: str, prefix: str,
                 keep_remote: int = 3) -> None:
        self.api = api
        self.repo_id = repo_id
        self.ckpt_dir = ckpt_dir
        self.prefix = prefix
        self.keep_remote = keep_remote
        self.thread: threading.Thread | None = None
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        self._write_probe()
        remote = _list_remote(api, repo_id) or []
        self.pushed: set[str] = {
            f for f in remote if _is_sync_name(f, prefix) or f in SIDE_FILES
        }

    def _write_probe(self) -> None:
        try:
            self.api.upload_file(
                path_or_fileobj=b"osrt sync write probe\n",
                path_in_repo=_PROBE_NAME, repo_id=self.repo_id,
                repo_type="model", commit_message="sync write probe",
            )
            self.api.delete_file(_PROBE_NAME, self.repo_id, repo_type="model",
                                 commit_message="sync write probe")
        except Exception as e:  # noqa: BLE001 — re-raised with the diagnosis
            raise RuntimeError(
                f"[hf-sync] cannot write to {self.repo_id} ({_err(e)}). "
                "Checkpoints would never be mirrored — check that HF_TOKEN has "
                "write access to this repo before spending GPU hours."
            ) from e

    def sync_once(self) -> bool:
        """One pass: upload every local file not known to be remote, newest
        first, then prune — but only when every upload in this pass succeeded,
        so a failed newest upload is never followed by deleting the older
        copies that are the only ones left. Returns True when nothing failed."""
        pending = [(p, os.path.basename(p))
                   for p in _local_files(self.ckpt_dir, self.prefix)]
        pending = [(p, n) for p, n in pending if n not in self.pushed]
        cutoff = _obsolete_cutoff(
            self.pushed | {n for _, n in pending}, self.prefix, self.keep_remote)
        ok = True
        for path, name in pending:
            if _is_sync_name(name, self.prefix) and _step_of(name) < cutoff:
                self.pushed.add(name)  # the prune would delete it at once
                continue
            print(f"[hf-sync] uploading {name}...", flush=True)
            if _upload(self.api, self.repo_id, path, name):
                self.pushed.add(name)
            else:
                ok = False
        if ok:
            self._prune()
        return ok

    def _prune(self) -> None:
        """Keep the newest `keep_remote` step/rescue files. `_final`, `_failed_`
        and SIDE_FILES are never candidates. Pruned names stay in `pushed`:
        they were mirrored, and re-uploading a still-local copy would loop."""
        if self.keep_remote < 1:
            return
        remote = [f for f in (_list_remote(self.api, self.repo_id) or [])
                  if _is_sync_name(f, self.prefix) and not _is_protected(f)]
        for old in sorted(remote, key=_step_of)[:-self.keep_remote]:
            self.api.delete_file(old, self.repo_id, repo_type="model")
            print(f"[hf-sync] pruned remote {old}", flush=True)

    def run_forever(self, interval: int) -> None:
        while True:
            try:
                self.sync_once()
            except Exception as e:  # noqa: BLE001 — never kill training on a sync hiccup
                print(f"[hf-sync] WARN {_err(e)}", flush=True)
            time.sleep(interval)


def start_push_daemon(repo_id: str, ckpt_dir: str, prefix: str,
                      interval: int = 60, keep_remote: int = 3) -> PushDaemon:
    """Background thread: upload new {prefix}_step_*.pt as they appear, and
    prune the repo to the newest `keep_remote` (HF is generous but 4.9GB each
    adds up). Fire-and-forget; the thread dies with the process, and `flush`
    covers the tail. Raises BEFORE any thread starts when the repo cannot be
    created or the token cannot write to it."""
    from huggingface_hub import HfApi

    daemon = PushDaemon(HfApi(), repo_id, ckpt_dir, prefix, keep_remote)
    daemon.thread = threading.Thread(
        target=daemon.run_forever, args=(interval,), daemon=True,
        name="hf-ckpt-sync",
    )
    daemon.thread.start()
    print(f"[hf-sync] push daemon started → {repo_id} (every {interval}s, keep "
          f"{keep_remote}; {len(daemon.pushed)} file(s) already remote)",
          flush=True)
    return daemon


def flush(repo_id: str, ckpt_dir: str, prefix: str,
          keep_remote: int = 3) -> bool:
    """Synchronously upload every local step/rescue/final/failed checkpoint
    (and SIDE_FILES) not already on the remote, NEWEST FIRST — the newest file
    is the one a pre-emption must not take — with bounded retries per file.
    The push daemon is `daemon=True` and both the 23h-rescue and the end-of-run
    paths exit within one poll interval of their final save, so the single
    most important file can miss its upload window. Call this from the trainer
    entrypoint after training returns, before the process exits, to make the
    tail durable. Returns True when everything that should be remote is.
    (ckpt-sync §2)"""
    from huggingface_hub import HfApi

    api = HfApi()
    remote: set[str] | None = None
    for i in range(3):
        try:
            remote = set(_list_remote(api, repo_id) or [])
            break
        except Exception as e:  # noqa: BLE001 — retried, then given up on
            print(f"[hf-sync] flush: cannot list repo (attempt {i + 1}/3): "
                  f"{_err(e)}", flush=True)
            if i < 2:
                time.sleep(_BACKOFF_S[i])
    if remote is None:
        print("[hf-sync] flush FAILED: repo unreachable, nothing uploaded",
              flush=True)
        return False

    local = _local_files(ckpt_dir, prefix, final=True, failed=True)
    cutoff = _obsolete_cutoff(
        remote | {os.path.basename(p) for p in local}, prefix, keep_remote)
    ok = True
    for path in local:
        name = os.path.basename(path)
        if name in remote:
            continue
        if _is_sync_name(name, prefix) and _step_of(name) < cutoff:
            continue  # older than what the remote keeps
        print(f"[hf-sync] flush uploading {name}...", flush=True)
        if _upload(api, repo_id, path, name):
            remote.add(name)
        else:
            ok = False
    print("[hf-sync] flush complete" if ok else
          "[hf-sync] flush INCOMPLETE — some files are NOT mirrored", flush=True)
    return ok
