"""T-061 (plan "Tests" 1, 2 and the clock part of 5): --data-workers / --data-prefetch / --torch-threads, and the
measure window's endpoint event order.

1. Flags: absent = the explicit defaults (4 / 8 / 4) = today's config, log, rows and checkpoints; given -> used (the
   training pool's size and window, the trainer's threads, T in the config) in both batch modes, and logged in the CLI
   header; 0 / negative rejected; a resume with another T is rejected; a resume with other N / P continues the same
   stream; the data workers still run one torch thread; the startup pools keep PRIOR_WORKERS / PRIOR_PREFETCH.
2. Same batches, positions path: N in {3, 4, 6, 8, 10} x P in {8, 20} give the serial loader's positions tensors,
   provenance, read hashes, sample digests, batcher counters and loader / batcher state over >= 30 distinct files
   (an epoch crossed); games path: N / P20 as test_worker_count.py.
3. Endpoint event order (fake clock): a 1 s memory sample at the start leaves the outer time unchanged, and so does a
   1 s memory sample at the end; the worker counters are read at the clock, so the busy delta is the window's too."""
import argparse
import re

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import build_game
from MahjongAI.tests import test_worker_count as WC
from MahjongAI.tests.test_positions_train import SIZES, _run, assert_runs_equal, step_lines
from MahjongAI.tests.test_resume_rows import _assert_equal
from MahjongAI.utils import phase_timing as PT
from MahjongAI.utils.dataloader import DataLoader, DataLoaderCluster
from MahjongAI.utils.game_workers import GameWorkers
from MahjongAI.utils.position_batcher import PositionBatcher


# ---------------------------------------------------------------- 1. flags
class _Pools:
    """GameWorkers that records (n, window) of every pool a run builds, in order."""

    def __init__(self):
        self.made = []

    def __call__(self, n, build, window=8, **k):
        self.made.append((n, window))
        return GameWorkers(n, build, window=window, **k)


def _log_lines(path, name):
    """The run log without timestamps, the run dir name replaced (two runs compare line by line)."""
    return [re.sub(r"^\[[^]]+\] ", "", line).replace(f"/{name}/", "/<run>/") for line in open(path)]


def test_absent_flags_equal_the_explicit_defaults(tmp_path, monkeypatch, train_data_root):
    _run(tmp_path, monkeypatch, train_data_root, "absent", steps=6, save_every=3)
    _run(tmp_path, monkeypatch, train_data_root, "explicit", steps=6, save_every=3, data_workers=4, data_prefetch=8,
         torch_threads=4)
    ha, rows_a = SL.Metrics(str(tmp_path / "absent" / "metrics.jsonl")).read()
    hb, rows_b = SL.Metrics(str(tmp_path / "explicit" / "metrics.jsonl")).read()
    assert ha["config"]["torch_threads"] == SL.TORCH_THREADS == 4
    assert set(ha["config"]) == {"model", "total_steps", "batch_size", "lr", "train_years", "data_seed", "model_seed",
                                 "train_files", "train_files_sha256", "val_year", "val_days", "val_games", "val_seed",
                                 "prior_games", "prior_seed", "prior_l2", "expected_skips", "torch_threads",
                                 "embd_size", "n_layers", "n_params", "batch_mode", "position_sizes", "device"}
    _assert_equal({k: v for k, v in ha.items() if k != "started"}, {k: v for k, v in hb.items() if k != "started"},
                  "header")
    _assert_equal(rows_a, rows_b, "rows")
    for step in (3, 6):
        _assert_equal(torch.load(tmp_path / "absent" / "checkpoints" / f"step_{step}.pt", weights_only=True),
                      torch.load(tmp_path / "explicit" / "checkpoints" / f"step_{step}.pt", weights_only=True),
                      f"step_{step}.pt")
    assert _log_lines(tmp_path / "absent.log", "absent") == _log_lines(tmp_path / "explicit.log", "explicit")


@pytest.mark.parametrize("batch_mode", ["positions", "games"])
def test_given_flags_are_used(tmp_path, monkeypatch, train_data_root, batch_mode):
    pools, threads = _Pools(), []
    set_threads = torch.set_num_threads
    monkeypatch.setattr(SL, "GameWorkers", pools)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: (threads.append(n), set_threads(n))[1])
    _run(tmp_path, monkeypatch, train_data_root, "flags", steps=2, save_every=2, batch_mode=batch_mode,
         data_workers=3, data_prefetch=20, torch_threads=2)
    # the training pool first (train), then the validation and the prior startup pools, unchanged
    assert pools.made == [(3, 20), (SL.PRIOR_WORKERS, SL.PRIOR_PREFETCH), (SL.PRIOR_WORKERS, SL.PRIOR_PREFETCH)]
    assert threads[0] == 2  # _train, before the model is built (the CPU eval shares the process's setting)
    header, _ = SL.Metrics(str(tmp_path / "flags" / "metrics.jsonl")).read()
    assert header["config"]["torch_threads"] == 2


def test_the_default_run_builds_the_default_pools(tmp_path, monkeypatch, train_data_root):
    pools = _Pools()
    monkeypatch.setattr(SL, "GameWorkers", pools)
    _run(tmp_path, monkeypatch, train_data_root, "plain", steps=1, save_every=1)
    assert pools.made == [(SL.DATA_WORKERS, SL.DATA_PREFETCH), (SL.PRIOR_WORKERS, SL.PRIOR_PREFETCH),
                          (SL.PRIOR_WORKERS, SL.PRIOR_PREFETCH)]


def test_invalid_values_are_rejected(tmp_path):
    for bad in ("0", "-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            SL.positive_int(bad)
    assert SL.positive_int("10") == 10
    for kw in ({"data_workers": 0}, {"data_prefetch": -1}, {"torch_threads": 0}, {"data_workers": 2.0}):
        with pytest.raises(AssertionError, match="must be an integer >= 1"):
            SL.train("encoder_conv", 1, str(tmp_path / "bad"), lambda *_: None, **kw)


def _args(**kw):
    base = dict(model_name="encoder_conv", max_steps=100, run_dir="runs/x", eval_every=50, save_every=50,
                embd_size=64, n_layers=1, batch_mode="positions", device="mps", stop_after_step=None,
                phase_timing=False, measure_window=(2, 3), profile=False, data_workers=None, data_prefetch=None,
                torch_threads=None)
    return argparse.Namespace(**{**base, **kw})


def test_the_header_logs_the_resolved_settings_only_when_given():
    plain = ("Training encoder_conv for 100 steps in runs/x (eval every 50, save every 50, embd_size 64, n_layers 1, "
             "batch mode positions, device mps), measure window 2,3")
    assert SL.run_header(_args()) == plain
    assert SL.run_header(_args(data_workers=10)) == plain + ", data workers 10, data prefetch 8, torch threads 4"
    assert SL.run_header(_args(data_workers=4, data_prefetch=20, torch_threads=2)) == (
        plain + ", data workers 4, data prefetch 20, torch threads 2")


def test_a_resume_with_other_torch_threads_is_rejected(tmp_path, monkeypatch, train_data_root):
    _run(tmp_path, monkeypatch, train_data_root, "t", steps=4, save_every=2, stop_after_step=2, torch_threads=2)
    with pytest.raises(AssertionError, match="torch_threads 2 -> 3"):
        _run(tmp_path, monkeypatch, train_data_root, "t", steps=4, save_every=2, torch_threads=3)
    with pytest.raises(AssertionError, match="torch_threads 2 -> 4"):  # the flag absent: the default T
        _run(tmp_path, monkeypatch, train_data_root, "t", steps=4, save_every=2)
    _run(tmp_path, monkeypatch, train_data_root, "t", steps=4, save_every=2, torch_threads=2)  # same T resumes
    assert sorted(step_lines(tmp_path / "t.log")) == [1, 2, 3, 4]


def test_a_resume_with_other_workers_and_prefetch_continues_the_same_stream(tmp_path, monkeypatch, train_data_root):
    """b: 5 updates with 3 workers / P20, resumed to 12 with 6 workers / P8; a: unbroken, the defaults. Same rows (the
    batcher counters in them), step lines (read hash, sample digest) and checkpoints 8 and 12."""
    _run(tmp_path, monkeypatch, train_data_root, "a")
    _run(tmp_path, monkeypatch, train_data_root, "b", stop_after_step=5, data_workers=3, data_prefetch=20)
    _run(tmp_path, monkeypatch, train_data_root, "b", data_workers=6, data_prefetch=8)
    assert "resuming at step 5" in open(tmp_path / "b.log").read()
    assert_runs_equal(tmp_path, "a", "b", [8, 12])


def _thread_parse(filename):
    return filename, None


def _thread_build(halfturns, tokens):
    return torch.get_num_threads()


def test_data_workers_run_one_torch_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(3)
    try:
        with GameWorkers(2, _thread_build, window=2, timeout_s=60, parse=_thread_parse) as w:
            assert [w.fetch(f"{i}.log").game()[0] for i in range(4)] == [1, 1, 1, 1]
    finally:
        torch.set_num_threads(threads)


# ---------------------------------------------------------------- 2. same batches, positions path
STEPS = 14


def _positions_stream(root, workers):
    """Per step: (flat peek tensors, peeked provenance, (read, sample digest), report); then the final cluster and
    batcher state and the distinct files read."""
    loader = DataLoader(str(root / "2021"), 1, lambda *_: None, seed=SL.DATA_SEED, expected_skips=SL.EXPECTED_SKIPS,
                        logged_skips=SL.LOGGED_SKIPS, workers=workers)
    cluster = DataLoaderCluster([loader], seed=SL.DATA_SEED)
    batcher = PositionBatcher(cluster, SIZES)
    steps, files = [], set()
    for _ in range(STEPS):
        batches = batcher.peek()
        flat = {h: [t for c in (b[0], *b[1], b[2], b[3], b[4]) for t in ([c] if torch.is_tensor(c) else [])]
                for h, b in batches.items()}
        provenance = batcher.peeked()
        read, digest = batcher.commit()
        files.update(read)
        steps.append((flat, provenance, (read, digest), batcher.report()))
    return steps, cluster.state_dict(), batcher.state_dict(), files


@pytest.fixture(scope="module")
def serial_stream(tmp_path_factory):
    from MahjongAI.tests.conftest import build_train_data_root
    root = build_train_data_root(tmp_path_factory.mktemp("sweep") / "data")
    return root, _positions_stream(root, None)


@pytest.mark.parametrize("n,p", [(n, p) for n in (3, 4, 6, 8, 10) for p in (8, 20)])
def test_workers_and_prefetch_give_the_serial_positions_stream(serial_stream, n, p):
    root, (ref_steps, ref_cluster, ref_batcher, ref_files) = serial_stream
    assert len(ref_files) >= 30 and ref_cluster["loaders"][0]["epoch"] >= 1  # P = 20 fills; an epoch is crossed
    with GameWorkers(n, build_game, window=p) as w:
        steps, cluster, batcher, files = _positions_stream(root, w)
        assert len(w.pids()) == n and w.window == p
    assert files == ref_files
    for k, (a, b) in enumerate(zip(ref_steps, steps)):
        _assert_equal(a, b, f"step {k + 1}")
    _assert_equal(ref_cluster, cluster, "cluster state")
    _assert_equal(ref_batcher, batcher, "batcher state")


@pytest.mark.parametrize("n", [4, 8, 10])
def test_games_path_with_prefetch_20_gives_the_same_batches(small_roots, monkeypatch, n):
    """test_worker_count.py at P = 20: n workers / P20 against 2 workers / P8."""
    base = WC._run(small_roots, 2)
    monkeypatch.setattr(SL, "DATA_PREFETCH", 20)
    other = WC._run(small_roots, n)
    _assert_equal(base, other, f"{n} workers / P20")


# ---------------------------------------------------------------- 3. endpoint event order
class _Batcher:
    def __init__(self):
        self.rows_out = {h: {0: 0} for h in PT.HEADS}
        self.rows_in = {h: {0: 0} for h in PT.HEADS}
        self.games_admitted = {0: 0}


class _TimedWorkers:
    """Busy counters that grow with the fake clock (a worker busy all the time) and an exposed wait."""

    def __init__(self, now, events):
        self.now, self.events, self.exposed_wait_s = now, events, 0.0

    def busy_seconds(self):
        self.events.append("counters")
        return [(self.now[0], int(self.now[0]))]


def test_endpoint_samples_stay_outside_the_window():
    now, events, logs = [0.0], [], []
    batcher = _Batcher()

    def clock():
        events.append("clock")
        return now[0]

    def slow_memory():  # a 1 s memory sample
        events.append("memory")
        now[0] += 1.0
        return {"rss": 1.0}
    w = PT.MeasureWindow(logs.append, 2, 3, lambda: events.append("sync"), batcher, lambda: {h: 0 for h in PT.HEADS},
                         _TimedWorkers(now, events), memory=slow_memory, clock=clock)
    for step in range(1, 6):
        now[0] += 10.0  # update `step` takes 10 s of training after the previous one (a sample's cost shifts it)
        w.update_done(step)
        if step == 2:
            assert events == ["memory", "sync", "clock", "counters"], events
            events.clear()
        if step == 5:
            assert events == ["sync", "clock", "counters", "memory"], events
        w.step_end(step)
    p = PT.parse_run([f"[t] {ln}" for ln in logs])
    # 1 s samples at either end: the outer time is the 3 updates' 30 s (start clock at 21 s, after its sample; end at
    # 51 s, before its sample), and so is the busy delta (the counters read at the clocks)
    assert p["end"]["outer"] == pytest.approx(30.0, abs=0.01), p["end"]
    assert p["end"]["busy"] == [pytest.approx(30.0, abs=0.01)], p["end"]
    assert {e: m["step"] for e, m in p["memory"].items()} == {"start": 2, "end": 5}
