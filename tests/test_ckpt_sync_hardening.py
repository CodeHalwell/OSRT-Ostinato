"""Locks the checkpoint-sync hardening fixes.

See docs/specs/2026-07-26-ckpt-sync-and-data-builder-findings.md:
  §1 — atomic save (no truncated file ever visible at the final name)
  §2 — rescue/final checkpoints are reachable by the sync glob + resume selection

and the 2026-09-30 review of scripts/hf_ckpt_sync.py:
  * only a missing repo may start clean; 401/403/5xx/network must raise
  * the push daemon seeds from the remote, proves the token can write first,
    prunes only after a fully successful pass, never prunes final/failed
  * flush goes newest-first with bounded retries and reports success

Everything network-shaped is a fake `HfApi`; the real module is imported so
the regex and step parser under test are the ones the trainer runs.
"""
import os
import sys
import types

import httpx
import pytest
import torch
import torch.nn as nn
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from osrt.train import save_checkpoint  # noqa: E402
from scripts import hf_ckpt_sync as sync  # noqa: E402
from scripts.hf_ckpt_sync import _SYNC_RE, _is_sync_name, _step_of  # noqa: E402


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


def test_rescue_and_step_are_syncable():
    """A rescue checkpoint must match the sync pattern; a bare _final.pt (no
    step) must NOT (it is synced via its step-numbered alias); a failed
    checkpoint or another run's prefix must not pass as this run's."""
    assert _SYNC_RE.match(f"{PREFIX}_step_5600.pt")
    assert _SYNC_RE.match(f"{PREFIX}_rescue_step_4447.pt")
    assert not _SYNC_RE.match(f"{PREFIX}_final.pt")
    assert _is_sync_name(f"{PREFIX}_step_5600.pt", PREFIX)
    assert _is_sync_name(f"{PREFIX}_rescue_step_4447.pt", PREFIX)
    assert not _is_sync_name(f"{PREFIX}_failed_step_4447.pt", PREFIX)
    assert not _is_sync_name("osrt_step_5600.pt", PREFIX)
    assert not _is_sync_name("osrt_v6_step_5600.pt", "osrt")


def test_latest_selection_prefers_higher_step_across_rescue():
    """pull_latest picks the highest step whether it is a rescue or numbered
    save — _step_of parses both name shapes."""
    remote = [
        f"{PREFIX}_step_5500.pt",
        f"{PREFIX}_rescue_step_5647.pt",  # capped mid-interval, the newest
        f"{PREFIX}_step_5600.pt",
    ]
    syncable = [f for f in remote if _is_sync_name(f, PREFIX)]
    assert max(syncable, key=_step_of) == f"{PREFIX}_rescue_step_5647.pt"
    assert _step_of(f"{PREFIX}_final.pt") == -1


def test_final_alias_glob_matches(tmp_path):
    """The end-of-run step-numbered alias is what the daemon glob catches; the
    bare _final.pt only joins for the exit flush."""
    (tmp_path / f"{PREFIX}_step_12600.pt").write_bytes(b"x")
    (tmp_path / f"{PREFIX}_final.pt").write_bytes(b"x")
    daemon_view = [os.path.basename(p)
                   for p in sync._local_files(str(tmp_path), PREFIX)]
    assert daemon_view == [f"{PREFIX}_step_12600.pt"]
    flush_view = [os.path.basename(p)
                  for p in sync._local_files(str(tmp_path), PREFIX, final=True)]
    assert flush_view == [f"{PREFIX}_step_12600.pt", f"{PREFIX}_final.pt"]


# ── fakes ──────────────────────────────────────────────────────────────────

REPO = "user/osrt-v7-ckpt"


def _http_error(cls, status: int, msg: str):
    req = httpx.Request("GET", f"https://huggingface.co/api/models/{REPO}")
    resp = httpx.Response(status, request=req, text=msg)
    return cls(f"{status} Client Error: {msg}", response=resp)


class _FakeApi:
    """Stands in for huggingface_hub.HfApi: a set of remote names plus a log
    of every write. `upload_failures[name] = n` fails that name's next n
    uploads; `deny_writes` fails every write (a read-only token)."""

    def __init__(self, files=(), *, list_error=None, upload_failures=None,
                 deny_writes=False):
        self.files = set(files)
        self.list_error = list_error
        self.upload_failures = dict(upload_failures or {})
        self.deny_writes = deny_writes
        self.attempts: list[str] = []   # every upload try, in order
        self.uploads: list[str] = []    # successful uploads, in order
        self.deletes: list[str] = []
        self.created = 0

    def list_repo_files(self, repo_id, repo_type="model"):
        assert repo_id == REPO
        if self.list_error is not None:
            raise self.list_error
        return sorted(self.files)

    def create_repo(self, repo_id, repo_type="model", private=True,
                    exist_ok=True):
        assert private and exist_ok
        self.created += 1

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id,
                    repo_type="model", **_):
        self.attempts.append(path_in_repo)
        if self.deny_writes:
            raise _http_error(HfHubHTTPError, 403,
                              "Forbidden: token hf_ABCDEFGHIJKLMNOPQRSTUVWX")
        left = self.upload_failures.get(path_in_repo, 0)
        if left > 0:
            self.upload_failures[path_in_repo] = left - 1
            raise _http_error(HfHubHTTPError, 503, "Service Unavailable")
        if isinstance(path_or_fileobj, str):
            assert os.path.exists(path_or_fileobj)
        self.uploads.append(path_in_repo)
        self.files.add(path_in_repo)

    def delete_file(self, path_in_repo, repo_id, repo_type="model", **_):
        if self.deny_writes:
            raise _http_error(HfHubHTTPError, 403, "Forbidden")
        self.deletes.append(path_in_repo)
        self.files.discard(path_in_repo)


class _StubThread:
    """threading.Thread that never runs: the tests drive `sync_once`."""

    started: list = []

    def __init__(self, target=None, args=(), daemon=False, name=None):
        self.target, self.args = target, args

    def start(self):
        _StubThread.started.append(self)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Install a fake hub around the module: `hub.api` is the fake HfApi,
    `hub.downloads` what pull_latest fetched, `hub.sleeps` the back-offs."""
    import huggingface_hub

    state = types.SimpleNamespace(api=None, downloads=[], sleeps=[],
                                  ckpt_dir=str(tmp_path))

    def use(api):
        state.api = api
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
        return api

    def fake_download(repo_id, filename, repo_type="model", local_dir=None):
        state.downloads.append(filename)
        with open(os.path.join(local_dir, filename), "wb") as f:
            f.write(b"ckpt")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(sync, "time", types.SimpleNamespace(
        sleep=lambda s: state.sleeps.append(s)))
    monkeypatch.setattr(sync, "threading",
                        types.SimpleNamespace(Thread=_StubThread))
    _StubThread.started = []
    state.use = use
    state.local = lambda *names: [
        open(os.path.join(state.ckpt_dir, n), "wb").close() for n in names]
    return state


# ── pull_latest ─────────────────────────────────────────────────────────────

def test_pull_latest_starts_clean_only_when_repo_is_missing(hub):
    hub.use(_FakeApi(list_error=_http_error(
        RepositoryNotFoundError, 404, "Repository Not Found")))
    assert sync.pull_latest(REPO, hub.ckpt_dir, "osrt") is None
    assert hub.downloads == []


@pytest.mark.parametrize("err", [
    _http_error(HfHubHTTPError, 401, "Unauthorized"),
    _http_error(HfHubHTTPError, 403, "Forbidden"),
    _http_error(HfHubHTTPError, 503, "Service Unavailable"),
    # huggingface_hub raises RepositoryNotFoundError for a 401 on a private
    # repo (missing/wrong token) too — only a real 404 may start clean.
    _http_error(RepositoryNotFoundError, 401, "Invalid username or password."),
    _http_error(RepositoryNotFoundError, 403, "Forbidden"),
    OSError("network is unreachable"),
])
def test_pull_latest_raises_on_anything_but_a_missing_repo(hub, err):
    hub.use(_FakeApi(list_error=err))
    with pytest.raises(type(err)):
        sync.pull_latest(REPO, hub.ckpt_dir, "osrt")
    assert hub.downloads == []


def test_pull_latest_existing_repo_without_checkpoints_starts_clean(hub):
    hub.use(_FakeApi(files=["README.md", "osrt_final.pt"]))
    assert sync.pull_latest(REPO, hub.ckpt_dir, "osrt") is None
    assert hub.downloads == []


def test_pull_latest_pulls_highest_step_including_rescue_and_side_files(hub):
    hub.use(_FakeApi(files=[
        "osrt_step_5500.pt", "osrt_rescue_step_5647.pt", "osrt_step_5600.pt",
        "osrt_failed_step_5700.pt",   # never resumed, so never pulled
        "osrt_final.pt", "wandb_run_id.txt", "README.md",
    ]))
    got = sync.pull_latest(REPO, hub.ckpt_dir, "osrt")
    assert got == "osrt_rescue_step_5647.pt"
    assert hub.downloads == ["wandb_run_id.txt", "osrt_rescue_step_5647.pt"]
    assert os.path.exists(os.path.join(hub.ckpt_dir, got))


def test_pull_latest_does_not_redownload_a_local_copy(hub):
    hub.use(_FakeApi(files=["osrt_step_100.pt", "wandb_run_id.txt"]))
    hub.local("osrt_step_100.pt", "wandb_run_id.txt")
    assert sync.pull_latest(REPO, hub.ckpt_dir, "osrt") == "osrt_step_100.pt"
    assert hub.downloads == []


def test_pull_latest_refreshes_side_files_when_the_remote_run_is_ahead(hub):
    """A persistent volume keeps its own wandb_run_id.txt; when another venue
    has progressed the run (a newer remote checkpoint) its side file is the
    truth and must replace the stale local one. (Codex review on PR #2.)"""
    hub.use(_FakeApi(files=["osrt_step_200.pt", "wandb_run_id.txt"]))
    hub.local("osrt_step_100.pt", "wandb_run_id.txt")
    assert sync.pull_latest(REPO, hub.ckpt_dir, "osrt") == "osrt_step_200.pt"
    assert hub.downloads == ["wandb_run_id.txt", "osrt_step_200.pt"]


def test_pull_latest_keeps_local_side_files_when_local_run_is_ahead(hub):
    hub.use(_FakeApi(files=["osrt_step_200.pt", "wandb_run_id.txt"]))
    hub.local("osrt_step_300.pt", "wandb_run_id.txt")
    # the remote's newest is still reported (it is what the repo holds) but
    # nothing is fetched over the local, more advanced run
    assert sync.pull_latest(REPO, hub.ckpt_dir, "osrt") == "osrt_step_200.pt"
    assert hub.downloads == ["osrt_step_200.pt"]


# ── push daemon ─────────────────────────────────────────────────────────────

def test_daemon_seeds_pushed_from_remote_and_uploads_only_new_files(hub):
    api = hub.use(_FakeApi(files=["osrt_step_100.pt", "wandb_run_id.txt"]))
    hub.local("osrt_step_100.pt", "osrt_step_200.pt", "wandb_run_id.txt")

    d = sync.start_push_daemon(REPO, hub.ckpt_dir, "osrt", interval=1)

    assert api.created == 1
    assert d.pushed == {"osrt_step_100.pt"}      # side files are not seeded by name
    assert len(_StubThread.started) == 1 and d.thread is _StubThread.started[0]
    # the write probe is the only write so far, and it cleaned up after itself
    assert api.uploads == [sync._PROBE_NAME]
    assert api.deletes == [sync._PROBE_NAME]
    assert sync._PROBE_NAME not in api.files

    assert d.sync_once() is True
    # the pulled checkpoint stayed put; the side file goes up once, by content
    assert api.uploads[1:] == ["osrt_step_200.pt", "wandb_run_id.txt"]
    assert d.sync_once() is True
    assert api.uploads[1:] == ["osrt_step_200.pt", "wandb_run_id.txt"]   # no re-uploads
    # An explicit --wandb-run-id rewrites the side file under a name the
    # remote already had: a name-only check never re-sent it (Codex, PR #2).
    with open(os.path.join(hub.ckpt_dir, "wandb_run_id.txt"), "w") as fh:
        fh.write("run-b\n")
    assert d.sync_once() is True
    assert api.uploads[1:] == ["osrt_step_200.pt", "wandb_run_id.txt",
                               "wandb_run_id.txt"]


def test_write_probe_failure_raises_before_any_thread_starts(hub):
    hub.use(_FakeApi(deny_writes=True))
    with pytest.raises(RuntimeError, match="cannot write") as info:
        sync.start_push_daemon(REPO, hub.ckpt_dir, "osrt")
    assert _StubThread.started == []
    assert "hf_ABCDEFGHIJKLMNOPQRSTUVWX" not in str(info.value)  # token scrubbed
    assert isinstance(info.value.__cause__, HfHubHTTPError)


def test_prune_keeps_newest_three_and_never_touches_final_or_failed(hub):
    api = hub.use(_FakeApi(files=[
        "osrt_step_100.pt", "osrt_step_200.pt", "osrt_step_300.pt",
        "osrt_step_400.pt", "osrt_rescue_step_450.pt",
        "osrt_final.pt", "osrt_failed_step_50.pt", "wandb_run_id.txt",
    ]))
    d = sync.PushDaemon(api, REPO, hub.ckpt_dir, "osrt", keep_remote=3)
    assert d.sync_once() is True
    assert api.deletes[1:] == ["osrt_step_100.pt", "osrt_step_200.pt"]
    assert api.files == {
        "osrt_step_300.pt", "osrt_step_400.pt", "osrt_rescue_step_450.pt",
        "osrt_final.pt", "osrt_failed_step_50.pt", "wandb_run_id.txt",
    }


def test_daemon_prunes_only_after_a_pass_where_every_upload_succeeded(hub):
    api = hub.use(_FakeApi(
        files=["osrt_step_100.pt", "osrt_step_200.pt", "osrt_step_300.pt",
               "osrt_step_400.pt"],
        upload_failures={"osrt_step_500.pt": 99}))
    hub.local("osrt_step_500.pt")
    d = sync.PushDaemon(api, REPO, hub.ckpt_dir, "osrt", keep_remote=3)

    assert d.sync_once() is False
    assert api.attempts.count("osrt_step_500.pt") == 3     # bounded retries
    assert hub.sleeps == [5, 15]
    assert api.deletes == [sync._PROBE_NAME]                # no prune
    assert "osrt_step_500.pt" not in d.pushed

    api.upload_failures.clear()
    assert d.sync_once() is True
    assert api.uploads[1:] == ["osrt_step_500.pt"]
    assert api.deletes[1:] == ["osrt_step_100.pt", "osrt_step_200.pt"]


def test_daemon_skips_local_files_the_prune_would_delete_at_once(hub):
    api = hub.use(_FakeApi(
        files=["osrt_step_300.pt", "osrt_step_400.pt", "osrt_step_500.pt"]))
    hub.local("osrt_step_100.pt", "osrt_step_200.pt", "osrt_step_600.pt")
    d = sync.PushDaemon(api, REPO, hub.ckpt_dir, "osrt", keep_remote=3)
    assert d.sync_once() is True
    assert api.uploads[1:] == ["osrt_step_600.pt"]
    assert api.deletes[1:] == ["osrt_step_300.pt"]
    assert {"osrt_step_100.pt", "osrt_step_200.pt"} <= d.pushed


# ── flush ───────────────────────────────────────────────────────────────────

def test_flush_uploads_newest_first_and_retries_with_backoff(hub):
    api = hub.use(_FakeApi(upload_failures={"osrt_step_200.pt": 1}))
    hub.local("osrt_step_100.pt", "osrt_step_200.pt", "osrt_rescue_step_250.pt")

    assert sync.flush(REPO, hub.ckpt_dir, "osrt") is True
    assert api.uploads == ["osrt_rescue_step_250.pt", "osrt_step_200.pt",
                           "osrt_step_100.pt"]
    assert api.attempts == ["osrt_rescue_step_250.pt", "osrt_step_200.pt",
                            "osrt_step_200.pt", "osrt_step_100.pt"]
    assert hub.sleeps == [5]


def test_flush_gives_up_after_three_attempts_and_reports_it(hub):
    api = hub.use(_FakeApi(upload_failures={"osrt_step_200.pt": 99}))
    hub.local("osrt_step_100.pt", "osrt_step_200.pt")

    assert sync.flush(REPO, hub.ckpt_dir, "osrt") is False
    assert api.attempts.count("osrt_step_200.pt") == 3
    assert hub.sleeps == [5, 15]
    assert api.uploads == ["osrt_step_100.pt"]   # the rest still went up


def test_flush_covers_final_failed_and_side_files_and_skips_remote(hub):
    api = hub.use(_FakeApi(files=["osrt_step_17000.pt"]))
    hub.local("osrt_step_17000.pt", "osrt_step_18000.pt", "osrt_final.pt",
              "osrt_failed_step_7.pt", "wandb_run_id.txt")
    assert sync.flush(REPO, hub.ckpt_dir, "osrt") is True
    assert api.uploads[0] == "osrt_step_18000.pt"
    assert set(api.uploads) == {"osrt_step_18000.pt", "osrt_failed_step_7.pt",
                                "osrt_final.pt", "wandb_run_id.txt"}


def test_flush_resends_side_files_the_remote_already_names(hub):
    api = hub.use(_FakeApi(files=["osrt_step_100.pt", "wandb_run_id.txt"]))
    hub.local("osrt_step_100.pt", "wandb_run_id.txt")
    assert sync.flush(REPO, hub.ckpt_dir, "osrt") is True
    assert api.uploads == ["wandb_run_id.txt"]


def test_flush_returns_false_when_the_repo_is_unreachable(hub):
    api = hub.use(_FakeApi(list_error=_http_error(
        HfHubHTTPError, 503, "Service Unavailable")))
    hub.local("osrt_step_100.pt")
    assert sync.flush(REPO, hub.ckpt_dir, "osrt") is False
    assert api.attempts == []
    assert hub.sleeps == [5, 15]
