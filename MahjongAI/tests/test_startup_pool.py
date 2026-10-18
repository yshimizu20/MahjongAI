"""T-047 S1: the startup builds the validation set (PrebuiltGame objects with the header's shanten step) through a
worker pool. Against the serial path (PRIOR_WORKERS = 0): the same accepted games, order, masks, labels, counts, header
references and prior bias; the same skip counters, log lines, error phases and UnexpectedSkip for expected exclusions,
an unexpected parse error, a tensor skip and a build failure, on both sides of the accepted-prefix boundary and with
the workers finishing out of order. Every startup phase closes its pool on a failure, a KeyboardInterrupt and a worker
death; training starts with one pool; a resume never refits the prior."""
import functools
import hashlib
import multiprocessing
import os
import signal
import time

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import PrebuiltGame, TransformerTensorProcessor
from MahjongAI.process_xml import process
from MahjongAI.utils.dataloader import DataLoader, UnexpectedSkip
from MahjongAI.utils.game_workers import DataWorkerError, GameWorkers


# ---------------------------------------------------------------- hooks the workers import (module level)
def _slow_parse(filename):
    """process(), slowed for half of the files (by name): the workers finish out of submission order."""
    if int(hashlib.sha256(os.path.basename(filename).encode()).hexdigest(), 16) % 2 == 0:
        time.sleep(0.4)
    return process(filename)


def _round_key(tokens) -> str:
    return hashlib.sha256(repr(list(tokens)).encode()).hexdigest()


def _flagged(tokens) -> bool:
    """The round named by T047_FLAGGED (set before a pool starts: the spawned workers inherit it)."""
    return _round_key(tokens) in os.environ.get("T047_FLAGGED", "").split(",")


class SkipOnSomeGames(TransformerTensorProcessor):
    """A tensor-level skip in the flagged round (the same round in every process)."""

    def build_tensors(self, turns, tokens):
        if _flagged(tokens):
            self.skipped["forced"] += 1
        return super().build_tensors(turns, tokens)


class FailOnSomeGames(TransformerTensorProcessor):
    def build_tensors(self, turns, tokens):
        if _flagged(tokens):
            raise ValueError("forced build failure")
        return super().build_tensors(turns, tokens)


def _flag_game(monkeypatch, root, index: int):
    """Flag the last round of the validation loader's accepted game `index` (0-based, read order)."""
    loader = DataLoader(str(root / SL.VAL_YEAR), 1, lambda *_: None, days=SL.VAL_DAYS, seed=SL.VAL_SEED,
                        expected_skips=SL.EXPECTED_SKIPS, logged_skips=SL.LOGGED_SKIPS)
    games = [SL._next_game(loader) for _ in range(index + 1)]
    monkeypatch.setenv("T047_FLAGGED", _round_key(games[index][2][-1]))


class _Recorded(GameWorkers):
    made = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _Recorded.made.append(self)


def _slow_pool(*a, **k):
    return _Recorded(*a, parse=_slow_parse, **k)


def _gone(pids):
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        return False
    return True


# ---------------------------------------------------------------- the validation set: pool vs serial
def _load(monkeypatch, workers, n_games, build=None):
    """ValSet.load with `workers` pool workers (0 = serial): (ValSet or the exception, log lines)."""
    monkeypatch.setattr(SL, "PRIOR_WORKERS", workers)
    monkeypatch.setattr(SL, "VAL_GAMES", n_games)
    if build is not None:
        monkeypatch.setattr(SL, "VAL_BUILD", build)
    logs = []
    try:
        return SL.ValSet.load(logs.append), logs
    except Exception as e:
        return e, logs


def _assert_same_valset(a, b):
    assert a.ids == b.ids
    assert all(isinstance(g, PrebuiltGame) for g in a.games + b.games)
    for ta, tb in zip(a.tensors, b.tensors):
        for head in SL.HEADS:
            for x, y in zip(ta[head], tb[head]):
                assert (x is None and y is None) or (x.dtype == y.dtype and torch.equal(x, y)), head
    for ga, gb in zip(a.games, b.games):
        assert (ga.shanten_best is None) == (gb.shanten_best is None)
        if ga.shanten_best is not None:
            assert torch.equal(torch.as_tensor(ga.shanten_best), torch.as_tensor(gb.shanten_best))


def _same_outcome(serial, pooled):
    (a, logs_a), (b, logs_b) = serial, pooled
    assert logs_a == logs_b
    assert isinstance(a, Exception) == isinstance(b, Exception), (a, b)
    if isinstance(a, Exception):
        assert type(a) is type(b) and str(a) == str(b), (a, b)
        return "raise"
    _assert_same_valset(a, b)
    return "ok"


def _accepted_order(root):
    """The validation loader's files in read order and the number of accepted games before each (serial loader,
    every skip tolerated)."""
    loader = DataLoader(str(root / SL.VAL_YEAR), 1, lambda *_: None, days=SL.VAL_DAYS, seed=SL.VAL_SEED,
                        expected_skips=None)
    return list(loader.file_list)


def test_valset_header_and_bias_equal_serial(monkeypatch, wide_val_root):
    """Expected exclusions on the way, workers finishing out of order: the same games, tensors, shanten step, header
    references, prior bias and log lines."""
    monkeypatch.setattr(SL, "DATA_ROOT", wide_val_root)
    monkeypatch.setattr(SL, "PRIOR_GAMES", 6)
    monkeypatch.setattr(SL, "GameWorkers", _slow_pool)
    serial, pooled = _load(monkeypatch, 0, 12), _load(monkeypatch, 3, 12)
    assert _same_outcome(serial, pooled) == "ok"
    assert any("skipped {" in line and "{}" not in line.split("skipped")[-1] for line in serial[1]), serial[1]
    logs = []
    monkeypatch.setattr(SL, "PRIOR_WORKERS", 0)
    bias_serial = SL.fit_prior(logs.append)
    monkeypatch.setattr(SL, "PRIOR_WORKERS", 3)
    bias_pooled = SL.fit_prior(logs.append)
    for head in SL.HEADS:
        assert torch.equal(bias_serial[head], bias_pooled[head]), head
    header_serial, header_pooled = SL.baseline_header(serial[0], bias_serial), SL.baseline_header(pooled[0], bias_pooled)
    assert header_serial == header_pooled
    assert sum(r[2] for r in header_serial["shanten_greedy"]) > 0


def test_header_shanten_step_equals_shanten_greedy(monkeypatch, wide_val_root):
    """The header's shanten-greedy hits from the stored shanten step equal baselines.shanten_greedy recomputed from
    the hands (the 468d71c header) for a bias with ties broken both ways."""
    from MahjongAI.utils.baselines import shanten_greedy
    monkeypatch.setattr(SL, "DATA_ROOT", wide_val_root)
    val, _ = _load(monkeypatch, 2, 8)
    g = torch.Generator().manual_seed(0)
    for bias in (torch.zeros(37, dtype=torch.float64), torch.randn(37, generator=g, dtype=torch.float64)):
        width = {h: val.head_samples(h)[0][1].shape[1] for h in SL.HEADS}
        header = SL.baseline_header(val, {h: bias if h == "discard" else torch.zeros(width[h], dtype=torch.float64)
                                          for h in SL.HEADS})
        expected = [[gid, len(y), int(shanten_greedy(hand, mask, y, bias).sum()) if len(y) else 0]
                    for gid, mask, y, hand in val.head_samples("discard")]
        assert header["shanten_greedy"] == expected


@pytest.mark.parametrize("kind", ["parse_error", "tensor_skip", "build_failure"])
def test_errors_around_the_accepted_prefix_boundary_equal_serial(monkeypatch, wide_val_root, kind):
    """For every boundary (VAL_GAMES) the pool and the serial path end the same way: the same UnexpectedSkip / build
    error, or the same validation set. Across the boundaries both outcomes occur: a bad game after the boundary is
    built by a worker (prefetched) but never counted."""
    monkeypatch.setattr(SL, "DATA_ROOT", wide_val_root)
    monkeypatch.setattr(SL, "GameWorkers", _slow_pool)
    build = None
    if kind == "parse_error":
        files = _accepted_order(wide_val_root)
        broken = files[len(files) // 2]
        target = broken + ".broken.log"
        os.rename(broken, target)  # keep the read order's slot: replace the link by a truncated copy
        with open(target, "rb") as f:
            data = f.read()
        os.remove(target)
        with open(broken, "wb") as f:
            f.write(data[:5000])
    else:  # the accepted game 6 (7th): after the boundary for 2 and 5, inside it for 9 and 14
        _flag_game(monkeypatch, wide_val_root, 6)
        cls = SkipOnSomeGames if kind == "tensor_skip" else FailOnSomeGames
        build = functools.partial(SL.build_val_game, processor_cls=cls)
    outcomes = []
    for n in (2, 5, 9, 14):
        serial = _load(monkeypatch, 0, n, build)
        pooled = _load(monkeypatch, 3, n, build)
        outcomes.append(_same_outcome(serial, pooled))
        if outcomes[-1] == "raise":
            error = serial[0]
            if kind == "parse_error":
                assert isinstance(error, UnexpectedSkip) and "ParseError" in str(error), error
            elif kind == "tensor_skip":
                assert isinstance(error, UnexpectedSkip) and "'forced'" in str(error), error
            else:
                assert isinstance(error, ValueError) and str(error) == "forced build failure", error
        assert not multiprocessing.active_children()
    # inside the prefix from some boundary on: ok below it, raise from it (the flagged game: between 5 and 9)
    assert "ok" in outcomes and "raise" in outcomes and outcomes == sorted(outcomes), outcomes
    if kind != "parse_error":
        assert outcomes == ["ok", "ok", "raise", "raise"], outcomes


# ---------------------------------------------------------------- lifecycle of every startup phase
def _train(tmp_path, steps=2):
    SL.train("encoder_conv", steps, str(tmp_path / "run"), SL.RunLog(str(tmp_path / "train.log")), 1, 1)


@pytest.fixture
def small_run(monkeypatch, wide_val_root):
    monkeypatch.setattr(SL, "DATA_ROOT", wide_val_root)
    monkeypatch.setattr(SL, "VAL_GAMES", 4)
    monkeypatch.setattr(SL, "PRIOR_GAMES", 4)
    monkeypatch.setattr(SL, "PRIOR_WORKERS", 2)
    monkeypatch.setattr(SL, "GameWorkers", _Recorded)
    _Recorded.made = []
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


def _assert_all_closed():
    for pool in _Recorded.made:
        assert pool.closed and pool.procs == [], pool
    assert not multiprocessing.active_children()


@pytest.mark.parametrize("phase", ["validation", "prior"])
@pytest.mark.parametrize("stop", ["error", "interrupt", "worker_death"])
def test_startup_pool_phases_close_every_pool(tmp_path, monkeypatch, small_run, phase, stop):
    """_next_game serves the validation set (calls 1..VAL_GAMES), then the prior: the stop comes at the 2nd game of
    the phase. Every pool is closed and no process is left."""
    calls, next_game = [0], SL._next_game
    at = 2 if phase == "validation" else SL.VAL_GAMES + 2

    def spy(loader):
        calls[0] += 1
        if calls[0] == at:
            pool = _Recorded.made[-1]
            assert pool.procs, "the phase's pool is not running"
            if stop == "error":
                raise RuntimeError("forced stop")
            if stop == "interrupt":
                raise KeyboardInterrupt()
            os.kill(pool.pids()[0], signal.SIGKILL)
            time.sleep(0.5)
        return next_game(loader)

    monkeypatch.setattr(SL, "_next_game", spy)
    expected = {"error": RuntimeError, "interrupt": KeyboardInterrupt, "worker_death": DataWorkerError}[stop]
    with pytest.raises(expected):
        _train(tmp_path)
    assert calls[0] == at
    _assert_all_closed()


@pytest.mark.parametrize("stop", ["error", "interrupt"])
def test_header_phase_failure_closes_every_pool(tmp_path, monkeypatch, small_run, stop):
    def failing_header(val, bias):
        raise RuntimeError("forced header failure") if stop == "error" else KeyboardInterrupt()

    monkeypatch.setattr(SL, "baseline_header", failing_header)
    with pytest.raises(RuntimeError if stop == "error" else KeyboardInterrupt):
        _train(tmp_path)
    _assert_all_closed()


def test_validation_build_failure_closes_every_pool(tmp_path, monkeypatch, small_run):
    _flag_game(monkeypatch, SL.DATA_ROOT, 2)
    monkeypatch.setattr(SL, "VAL_BUILD", functools.partial(SL.build_val_game, processor_cls=FailOnSomeGames))
    with pytest.raises(ValueError, match="^forced build failure$"):
        _train(tmp_path)
    _assert_all_closed()


def test_training_starts_with_one_pool_and_a_resume_never_refits_the_prior(tmp_path, monkeypatch, small_run):
    """At the first training batch only the training pool runs (the startup pools are closed). A resume of the run
    passes with fit_prior raising, and builds its validation set through a pool again."""
    seen = []
    check = SL._check_tensor_skips

    def spy(processor, where, files=()):
        if where.startswith("the training batch"):
            seen.append((len(multiprocessing.active_children()), [p.closed for p in _Recorded.made[1:]]))
        return check(processor, where, files)

    monkeypatch.setattr(SL, "_check_tensor_skips", spy)
    _train(tmp_path, steps=2)
    training, *startup = _Recorded.made
    assert len(startup) == 2 and seen and seen[0] == (SL.DATA_WORKERS, [True, True]), seen
    _assert_all_closed()

    def no_refit(log):
        raise AssertionError("fit_prior called on a resume")

    monkeypatch.setattr(SL, "fit_prior", no_refit)
    _Recorded.made = []
    _train(tmp_path, steps=3)  # resumes from step 2
    assert len(_Recorded.made) == 2  # the training pool and the validation pool; no prior pool
    log = open(tmp_path / "train.log").read()
    assert "resuming at step 2; prior loaded (no refit)" in log and "resume check:" in log
    _assert_all_closed()


def test_combined_eval_consumes_the_pool_built_games(monkeypatch, wide_val_root):
    """E1 + S1: the pool-built validation games evaluate to the serial-built ones' rows exactly (counts, top-1, NLL)
    at several games per call; the eval never builds a game again."""
    monkeypatch.setattr(SL, "DATA_ROOT", wide_val_root)
    serial, _ = _load(monkeypatch, 0, 9)
    pooled, _ = _load(monkeypatch, 3, 9)
    torch.manual_seed(0)
    model = SL.EncoderConvModel(verbose=False)
    import MahjongAI.agents.encoder_conv as EC
    monkeypatch.setattr(EC, "build_game", lambda *a, **k: (_ for _ in ()).throw(AssertionError("built again")))
    assert serial.evaluate(model, games_per_forward=4) == pooled.evaluate(model, games_per_forward=4)
