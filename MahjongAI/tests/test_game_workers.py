"""T-011 D1: games parsed and tensor-built in worker processes give the serial path's batches, skips, errors, loader
state and RNG use; a dead or hung worker is fatal and leaves no process; workers start lazily and stop on every stop
path. Each comparison runs a serial and a worker path side by side, each with its own processor."""
import functools
import multiprocessing
import os
import pickle
import re
import signal
import struct
import threading
import time

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import EncoderConvModel, TransformerTensorProcessor, build_game
from MahjongAI.utils.dataloader import DataLoader, DataLoaderCluster, UnexpectedSkip
from MahjongAI.process_xml import InvalidGameException, process
from MahjongAI.utils import dataloader as DL
from MahjongAI.utils.game_workers import DataWorkerError, GameWorkers, RemoteError


# ---------------------------------------------------------------- processors that a test forces to skip, fail or stall
class SkippingProcessor(TransformerTensorProcessor):
    def build_tensors(self, turns, tokens):
        self.skipped["forced"] += 1  # one forced tensor-level skip per round
        return super().build_tensors(turns, tokens)


class Unpicklable(Exception):
    """Pickles, but does not unpickle: its __init__ needs two arguments and its args hold one."""

    def __init__(self, message, code):
        super().__init__(message)


class UnpicklableInvalid(InvalidGameException):
    """An invalid game whose exception refuses pickling."""

    def __reduce__(self):
        raise TypeError("UnpicklableInvalid does not pickle")


def _parse_or_raise(filename):
    """process(), but a file named *-unpick* raises an exception that does not survive pickling."""
    if "-unpick" in filename:
        if "-unpickinv" in filename:
            raise UnpicklableInvalid("Weird seat")
        raise Unpicklable(f"cannot read {os.path.basename(filename)}", 7)
    return process(filename)


class FailingProcessor(TransformerTensorProcessor):
    error, extra = ValueError, ()

    def __init__(self):
        super().__init__()
        self.calls = 0

    def build_tensors(self, turns, tokens):
        self.calls += 1
        if self.calls == 2:  # the 2nd round of the first game, in both paths
            self.skipped["before failure"] += 1
            raise self.error("forced build failure", *self.extra)
        return super().build_tensors(turns, tokens)


class UnpicklableFailingProcessor(FailingProcessor):
    error, extra = Unpicklable, (7,)


class StallingProcessor(TransformerTensorProcessor):
    def build_tensors(self, turns, tokens):
        time.sleep(30)


def _send_half_then_die(fd, body):
    """A worker's frame writer that sends half of a frame, lets the parent read it, then dies (os._exit)."""
    os.write(fd, struct.pack("!Q", len(body)) + body[:len(body) // 2])
    time.sleep(1.0)
    os._exit(3)


def _workers(n=2, processor_cls=None, **kw):
    build = functools.partial(build_game, processor_cls=processor_cls) if processor_cls else build_game
    return GameWorkers(n, build, window=kw.pop("window", 4), **kw)


def _cluster(roots, batch_size, expected_skips, workers=None, log=None):
    loaders = [DataLoader(str(r), batch_size, log or (lambda *_: None), seed=0, expected_skips=expected_skips,
                          logged_skips=SL.LOGGED_SKIPS, workers=workers) for r in roots]
    return DataLoaderCluster(loaders, seed=0)


def _flat(batches):
    return [t for head in batches for t in (head[0], *head[1], head[2], head[3], head[4])]


def _assert_batches_equal(a, b, where=""):
    for i, (x, y) in enumerate(zip(_flat(a), _flat(b))):
        assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y), f"{where} tensor {i}"


def _gone(pids):
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        return False
    return True


# ---------------------------------------------------------------- equality on the T-005 configuration
@pytest.mark.parametrize("batch_size", [1, 2])
def test_batches_equal_across_loaders_epochs_and_empty_batches(small_roots, batch_size):
    """Two loaders (mixed draws), past two epoch changes, expected skips counted; with batch size 1 a skipped game
    gives an empty batch. Per batch: tensors, files, cluster state, skips and log lines equal."""
    logs_a, logs_b = [], []
    serial = _cluster(small_roots, batch_size, SL.EXPECTED_SKIPS, log=logs_a.append)
    with _workers() as w:
        fast = _cluster(small_roots, batch_size, SL.EXPECTED_SKIPS, w, log=logs_b.append)
        pa, pb = TransformerTensorProcessor(), TransformerTensorProcessor()
        empty, n = 0, 15 // batch_size * 2 + 3
        for k in range(n):
            ga, ta = next(serial)
            gb, tb = next(fast)
            assert len(ga) == len(gb)
            empty += not ga
            if ga:  # the trainer skips an empty batch before the model
                _assert_batches_equal(pa.prepare_batches(ga, ta), pb.prepare_batches(gb, tb), f"batch {k}")
            assert serial.last_batch_files == fast.last_batch_files and serial.state_dict() == fast.state_dict(), k
        assert serial.epoch >= 2 and serial.draws == n
        assert sum(serial.skipped().values()) >= 8  # 4 skipped games per epoch
        assert empty > 0 if batch_size == 1 else True
        assert logs_a == logs_b and pa.skipped == pb.skipped


# ---------------------------------------------------------------- error phases
def test_parse_errors_raise_the_same_unexpected_skips_and_keep_pending(small_roots, tmp_path):
    """No skip is expected: every invalid game and a parser error (a truncated log) raise UnexpectedSkip. The raise
    sequence (batch, message, loader state with `pending`) and the log lines are the serial path's, and so are the
    batches read after each raise (pending recovery with prefetched results outstanding)."""
    broken = small_roots[0] / "20210101" / "20210101_00" / "2021010100gm-0000-0000-broken00.log"
    src = next(small_roots[0].glob("*/*/*.log")).resolve()
    broken.write_bytes(src.read_bytes()[:5000])

    def run(workers):
        logs, events = [], []
        cluster = _cluster(small_roots, 3, frozenset(), workers, logs.append)
        processor = TransformerTensorProcessor()
        for k in range(14):
            try:
                games, tokens = next(cluster)
                tensors = _flat(processor.prepare_batches(games, tokens)) if games else []
                events.append((k, "batch", list(cluster.last_batch_files), tensors))
            except UnexpectedSkip as e:
                events.append((k, "raise", str(e), cluster.state_dict()))
        return logs, events

    logs_a, events_a = run(None)
    with _workers() as w:
        logs_b, events_b = run(w)
    assert logs_a == logs_b
    assert any("ParseError" in line for line in logs_a)
    assert [e[:2] for e in events_a] == [e[:2] for e in events_b]
    for a, b in zip(events_a, events_b):
        if a[1] == "raise":
            assert a[2:] == b[2:], a[0]
        else:
            assert a[2] == b[2] and all(torch.equal(x, y) for x, y in zip(a[3], b[3])), a[0]
    raises = [e for e in events_a if e[1] == "raise"]
    assert len(raises) >= 5 and any(ld["pending"] for e in raises for ld in e[3]["loaders"])


@pytest.mark.parametrize("processor_cls", [FailingProcessor, UnpicklableFailingProcessor])
def test_build_error_raises_at_the_build_point(small_roots, processor_cls):
    """A tensor-build failure surfaces in prepare_batches, after the loader accepted the batch, with the same message,
    loader state and tensor-level skip counts in both paths, and the same type; an exception that does not pickle
    arrives as a RemoteError record of phase "build" with the original type's name."""
    serial = _cluster(small_roots, 2, SL.EXPECTED_SKIPS)
    ga, ta = next(serial)
    pa = processor_cls()
    with pytest.raises(processor_cls.error, match="^forced build failure$"):
        pa.prepare_batches(ga, ta)
    with _workers(processor_cls=processor_cls) as w:
        fast = _cluster(small_roots, 2, SL.EXPECTED_SKIPS, w)
        gb, tb = next(fast)  # the worker's build failure is not a loader event
        assert fast.state_dict() == serial.state_dict() and fast.last_batch_files == serial.last_batch_files
        pb = TransformerTensorProcessor()
        expected = ValueError if processor_cls.error is ValueError else RemoteError
        with pytest.raises(expected, match="^forced build failure$") as error:
            pb.prepare_batches(gb, tb)
    if expected is RemoteError:
        assert (error.value.phase, error.value.type_name) == ("build", "Unpicklable")
    assert pa.skipped == pb.skipped == {"before failure": 1}


@pytest.mark.parametrize("kind, key", [("unpick00", "Unpicklable"), ("unpickinv0", "invalid:Weird seat")])
def test_unpicklable_parse_error_keeps_the_serial_skip_key_and_log(small_roots, monkeypatch, kind, key):
    """A parse error that does not survive pickling: the worker sends a RemoteError record (phase "parse"), and the
    loader counts the serial path's skip key, writes its log line and reads the same batches."""
    src = next(small_roots[0].glob("*/*/*.log")).resolve()
    day = small_roots[0] / "20210101" / "20210101_00"
    day.mkdir(parents=True, exist_ok=True)
    (day / f"2021010100gm-0000-0000-{kind}.log").write_bytes(src.read_bytes())
    monkeypatch.setattr(DL, "process", _parse_or_raise)

    def run(workers):
        logs = []
        cluster = _cluster(small_roots, 2, SL.EXPECTED_SKIPS | {key}, workers, logs.append)
        batches = [(next(cluster), list(cluster.last_batch_files))[1] for _ in range(8)]
        return logs, batches, cluster.skipped(), cluster.state_dict()

    serial = run(None)
    with _workers(parse=_parse_or_raise) as w:
        fast = run(w)
    assert serial == fast and serial[2][key] >= 1
    assert (key == "Unpicklable") == any(": Unpicklable: cannot read " in line for line in serial[0])
    record = pickle.loads(pickle.dumps(RemoteError("parse", "Unpicklable", "text", None, False)))
    assert (record.phase, record.type_name, str(record), record.msg) == ("parse", "Unpicklable", "text", None)


def test_tensor_skips_raise_the_same_unexpected_skip_before_the_update(small_roots):
    """A forced tensor-level skip in every round: the training step's _check_tensor_skips raises the same
    UnexpectedSkip (same counts) in both paths, before backward and the optimizer touch the weights."""
    def step(model, cluster):
        torch.manual_seed(0)
        params = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        games, tokens = next(cluster)
        results = model(games, tokens, sample=True)
        try:
            SL._check_tensor_skips(model.tensor_processor, "the training batch", cluster.last_batch_files)
            sum(r.loss for r in results.values()).backward()
            opt.step()
        finally:
            assert all(torch.equal(p, q) for p, q in zip(params, model.parameters()))

    model = EncoderConvModel(verbose=False)
    model.tensor_processor = SkippingProcessor()
    with pytest.raises(UnexpectedSkip) as serial_error:
        step(model, _cluster(small_roots, 2, SL.EXPECTED_SKIPS))
    model = EncoderConvModel(verbose=False)
    with _workers(processor_cls=SkippingProcessor) as w, pytest.raises(UnexpectedSkip) as worker_error:
        step(model, _cluster(small_roots, 2, SL.EXPECTED_SKIPS, w))
    assert str(worker_error.value) == str(serial_error.value) and "'forced'" in str(serial_error.value)


# ---------------------------------------------------------------- sample cap and RNG
def test_sample_cap_targets_gradients_and_rng_equal(small_roots):
    """A small sample cap (every head above it), one seed before loader -> model -> backward: the selected targets,
    the logits, every gradient and the final RNG state are equal."""
    def run(workers):
        torch.manual_seed(0)
        model = EncoderConvModel(verbose=False).train()
        cluster = _cluster(small_roots, 2, SL.EXPECTED_SKIPS, workers)
        torch.manual_seed(5)
        games, tokens = next(cluster)
        results = model(games, tokens, sample=True, sample_size=20)
        sum(r.loss for r in results.values()).backward()
        return ({h: (r.y, r.logits.detach()) for h, r in results.items()},
                [p.grad for p in model.parameters()], torch.get_rng_state())

    out_a, grads_a, rng_a = run(None)
    with _workers() as w:
        out_b, grads_b, rng_b = run(w)
    assert out_a.keys() == out_b.keys() == set(SL.HEADS) and all(len(out_a[h][0]) == 20 for h in out_a)
    for h in out_a:
        assert torch.equal(out_a[h][0], out_b[h][0]) and torch.equal(out_a[h][1], out_b[h][1]), h
    assert all(torch.equal(a, b) for a, b in zip(grads_a, grads_b))
    assert torch.equal(rng_a, rng_b)


# ---------------------------------------------------------------- resume with prefetched results outstanding
def test_state_dict_mid_epoch_resumes_to_the_same_batches(small_roots):
    serial = _cluster(small_roots, 2, SL.EXPECTED_SKIPS)
    expected = [(next(serial), list(serial.last_batch_files))[1] for _ in range(9)]
    with _workers() as w:
        fast = _cluster(small_roots, 2, SL.EXPECTED_SKIPS, w)
        for _ in range(4):
            next(fast)
        state = fast.state_dict()
        assert w.in_flight or w.done  # results prefetched and not read yet
    with _workers() as w2:
        resumed = _cluster(small_roots, 2, SL.EXPECTED_SKIPS, w2)
        resumed.load_state_dict(state)
        got = [(next(resumed), list(resumed.last_batch_files))[1] for _ in range(5)]
    assert got == expected[4:]


# ---------------------------------------------------------------- fatal worker errors
def test_dead_worker_is_fatal_names_the_file_and_leaves_no_process(small_roots):
    serial = _cluster(small_roots, 2, SL.EXPECTED_SKIPS)
    w = _workers()
    fast = _cluster(small_roots, 2, SL.EXPECTED_SKIPS, w)
    next(serial), next(fast)
    before = fast.state_dict()
    pids = w.pids()
    assert len(pids) == 2
    os.kill(pids[0], signal.SIGKILL)
    time.sleep(0.5)
    with pytest.raises(DataWorkerError, match=r"died \(exit code -9\) while fetching .*\.log"):
        next(fast)
    assert w.pids() == [] and _gone(pids) and not multiprocessing.active_children()
    # the failed batch was rolled back (draw, skips, position): a resume reads the serial path's next batches
    assert fast.state_dict() == before
    resumed = _cluster(small_roots, 2, SL.EXPECTED_SKIPS)
    resumed.load_state_dict(fast.state_dict())
    for _ in range(4):
        next(serial), next(resumed)
        assert resumed.last_batch_files == serial.last_batch_files


def test_worker_death_mid_frame_is_fatal_and_leaves_no_process(small_roots):
    """A worker dies after half of a result frame is sent (the other workers still run): fetch must not block on the
    cut frame. It raises DataWorkerError long before the deadline, and every worker is stopped."""
    w = _workers(send=_send_half_then_die, timeout_s=60)
    fast = _cluster(small_roots, 2, SL.EXPECTED_SKIPS, w)
    w.prefetch([])
    pids = w.pids()
    out = {}

    def run():
        t = time.monotonic()
        try:
            next(fast)
        except BaseException as e:
            out["error"] = e
        out["seconds"] = time.monotonic() - t

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(30)
    assert not thread.is_alive(), "fetch is blocked on a cut result frame"
    assert isinstance(out.get("error"), DataWorkerError), out
    assert re.search(r"(closed its result pipe \([1-9]\d* byte\(s\) of an unfinished frame\)|died \(exit code 3\)) "
                     r"while fetching .*\.log", str(out["error"])), str(out["error"])
    assert out["seconds"] < 30 and w.pids() == [] and _gone(pids) and not multiprocessing.active_children()


def test_hung_worker_times_out_and_leaves_no_process(small_roots):
    w = _workers(processor_cls=StallingProcessor, timeout_s=3)
    fast = _cluster(small_roots, 2, SL.EXPECTED_SKIPS, w)
    t = time.monotonic()
    with pytest.raises(DataWorkerError, match=r"no result for .*\.log from the data workers within 3 s"):
        next(fast)
    pids_seen = w.pids()
    assert time.monotonic() - t < 30 and pids_seen == [] and not multiprocessing.active_children()


# ---------------------------------------------------------------- lazy start, shutdown on every stop path
class _Recorded(GameWorkers):
    made = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.started = []
        _Recorded.made.append(self)

    def _start(self):
        super()._start()
        self.started = self.started or self.pids()


@pytest.mark.parametrize("stop", ["normal", "error", "interrupt"])
def test_train_starts_workers_lazily_and_stops_them_on_every_stop_path(tmp_path, monkeypatch, train_data_root, stop):
    monkeypatch.setattr(SL, "DATA_ROOT", train_data_root)
    monkeypatch.setattr(SL, "VAL_GAMES", 1)
    monkeypatch.setattr(SL, "PRIOR_GAMES", 2)
    monkeypatch.setattr(SL, "PRIOR_WORKERS", 2)
    monkeypatch.setattr(SL, "GameWorkers", _Recorded)
    _Recorded.made = []  # [the training pool (made first, started lazily), the validation pool (T-047 S1), the prior's pool (P)]
    started_before_loop = []
    check = SL._check_tensor_skips

    def spy(processor, where, files=()):
        if where.startswith("the training batch"):
            started_before_loop.append(bool(_Recorded.made[0].started))
            if stop != "normal" and len(started_before_loop) == 2:
                raise (RuntimeError("forced stop") if stop == "error" else KeyboardInterrupt())
        return check(processor, where, files)

    monkeypatch.setattr(SL, "_check_tensor_skips", spy)
    fit_prior, started_after_prior = SL.fit_prior, []

    def fit_prior_spy(log):
        out = fit_prior(log)
        prior_pool = _Recorded.made[2]
        started_after_prior.append((bool(_Recorded.made[0].started), prior_pool.closed and prior_pool.procs == []))
        return out

    monkeypatch.setattr(SL, "fit_prior", fit_prior_spy)
    threads = torch.get_num_threads()
    try:
        if stop == "normal":
            SL.train("encoder_conv", 3, str(tmp_path / "run"), SL.RunLog(str(tmp_path / "train.log")), 3, 3)
        else:
            with pytest.raises(RuntimeError if stop == "error" else KeyboardInterrupt):
                SL.train("encoder_conv", 3, str(tmp_path / "run"), SL.RunLog(str(tmp_path / "train.log")), 3, 3)
    finally:
        torch.set_num_threads(threads)
    w, val_pool, prior_pool = _Recorded.made
    assert len(val_pool.started) == SL.PRIOR_WORKERS and _gone(val_pool.started) and val_pool.closed  # closed at load
    assert started_after_prior == [(False, True)] and started_before_loop[0]  # started by the first training batch
    assert len(w.started) == SL.DATA_WORKERS
    assert len(prior_pool.started) == SL.PRIOR_WORKERS and _gone(prior_pool.started)  # closed when the prior was fitted
    assert w.procs == [] and w.closed and _gone(w.started) and not multiprocessing.active_children()
    if stop != "normal":  # the stop saved a checkpoint whose loader state keeps the stopped batch's games
        assert os.path.exists(tmp_path / "run" / "checkpoints" / "step_1.pt")


def test_workers_do_not_start_before_use(small_roots):
    w = _workers()
    _cluster(small_roots, 2, SL.EXPECTED_SKIPS, w)
    assert w.pids() == [] and not multiprocessing.active_children()
    w.close()
