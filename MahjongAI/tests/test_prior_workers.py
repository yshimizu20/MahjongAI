"""T-011 P: the prior's games parsed and built by a worker pool give the serial path's accepted games (same ids, same
order), masks, labels, sample counts, skip counts, log lines and bias; fit_prior closes its pool."""
import multiprocessing

import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import build_game
from MahjongAI.utils.game_workers import GameWorkers

PRIOR_GAMES = 14  # of the 42 fixture 2021 games: the prior's loader skips files on the way (asserted below)


def _assert_samples_equal(a: SL.PriorSample, b: SL.PriorSample):
    assert a.files == b.files and len(a.files) == PRIOR_GAMES
    assert a.files_read == b.files_read and a.skipped == b.skipped
    for head in SL.HEADS:
        assert len(a.masks[head]) == len(b.masks[head]) == len(a.labels[head]) == len(b.labels[head]) == PRIOR_GAMES
        assert [len(y) for y in a.labels[head]] == [len(y) for y in b.labels[head]], f"{head}: per-game counts"
        for i, (ma, mb, ya, yb) in enumerate(zip(a.masks[head], b.masks[head], a.labels[head], b.labels[head])):
            assert torch.equal(ma, mb) and torch.equal(ya, yb), f"{head}: game {i} masks / labels differ"


def test_prior_sample_and_bias_with_workers_equal_serial(monkeypatch, train_data_root):
    monkeypatch.setattr(SL, "DATA_ROOT", train_data_root)
    monkeypatch.setattr(SL, "PRIOR_GAMES", PRIOR_GAMES)
    log_serial, log_workers = [], []
    serial = SL.prior_sample(log_serial.append)
    with GameWorkers(3, build_game, window=5) as workers:
        pooled = SL.prior_sample(log_workers.append, workers)
    assert sum(serial.skipped.values()) > 0, "the sample must include files the prior excludes"
    assert serial.files_read == PRIOR_GAMES + sum(serial.skipped.values())
    _assert_samples_equal(pooled, serial)
    assert log_workers == log_serial
    bias_serial, bias_pooled = SL.prior_bias(log_serial.append, serial), SL.prior_bias(log_workers.append, pooled)
    for head in SL.HEADS:
        torch.testing.assert_close(bias_pooled[head], bias_serial[head], atol=1e-6, rtol=0)
    assert log_workers == log_serial


class _Recorded(GameWorkers):
    made = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _Recorded.made.append(self)


def test_fit_prior_uses_its_own_pool_and_closes_it(monkeypatch, train_data_root):
    monkeypatch.setattr(SL, "DATA_ROOT", train_data_root)
    monkeypatch.setattr(SL, "PRIOR_GAMES", PRIOR_GAMES)
    monkeypatch.setattr(SL, "GameWorkers", _Recorded)
    log = []
    serial = SL.prior_bias(log.append, SL.prior_sample(log.append))
    for n in (2, 0):  # 0 = serial, no pool
        monkeypatch.setattr(SL, "PRIOR_WORKERS", n)
        _Recorded.made = []
        bias = SL.fit_prior(log.append)
        for head in SL.HEADS:
            torch.testing.assert_close(bias[head], serial[head], atol=1e-6, rtol=0)
        if n:
            (pool,) = _Recorded.made
            assert pool.n == n and pool.window == SL.PRIOR_PREFETCH and pool.closed and pool.procs == []
        else:
            assert _Recorded.made == []
        assert not multiprocessing.active_children()
