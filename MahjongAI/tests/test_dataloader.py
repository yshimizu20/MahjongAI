"""DataLoader skip handling (plan T-005): an expected exclusion is counted (and logged with its file when asked); any
other skip raises UnexpectedSkip after the loader has moved past the game; skip counts survive state_dict.
Read order (plan T-045): epoch 0 is the pre-T-045 order, each later epoch has its own order, the cluster mixes its
loaders by expected accepted games left, and every state (also a pre-T-045 one) resumes to the same batches."""
import functools
import hashlib
import io
import math
import os
import random
from collections import Counter

import pytest
import torch

from MahjongAI.supervised_learning import EXPECTED_SKIPS, LOGGED_SKIPS, train_files_sha256
from MahjongAI.utils import dataloader as dataloader_module
from MahjongAI.utils import gates
from MahjongAI.utils.dataloader import DataLoader, DataLoaderCluster, UnexpectedSkip, epoch_order

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")
KEY = "invalid:Own-turn action after own discard"
DAY, NAME, _ = gates.AFTER_OWN_DISCARD_LOGS[0]
DAY_DIR = os.path.join(GATE_GAMES, DAY[:4], DAY)
PATH = os.path.join(DAY_DIR, f"{DAY}_00", NAME)


def _drain(loader):
    for _ in loader:
        pass


def test_expected_skip_is_counted_and_logged():
    lines = []
    loader = DataLoader(DAY_DIR, 1, lines.append, expected_skips=EXPECTED_SKIPS, logged_skips=LOGGED_SKIPS)
    assert PATH in loader.file_list
    _drain(loader)
    assert loader.skipped[KEY] >= 1
    assert f"skipped {PATH}: {KEY}" in lines
    restored = DataLoader(DAY_DIR, 1, lines.append)
    restored.load_state_dict(loader.state_dict())
    assert restored.skipped == loader.skipped and restored.current_index == loader.current_index


def test_unexpected_skip_raises_past_the_game():
    loader = DataLoader(DAY_DIR, 1, lambda *_: None, expected_skips=EXPECTED_SKIPS - {KEY})
    with pytest.raises(UnexpectedSkip, match="Own-turn action after own discard"):
        _drain(loader)
    assert os.path.basename(loader.file_list[loader.current_index - 1]) in {g[1] for g in gates.AFTER_OWN_DISCARD_LOGS}
    assert loader.skipped[KEY] == 1


@functools.lru_cache(maxsize=None)
def _accepted_games(k):
    """k fixture games that parse (2021 gate games), sorted."""
    from MahjongAI.process_xml import process
    found = []
    for root, _, names in sorted(os.walk(os.path.join(GATE_GAMES, "2021"))):
        for name in sorted(names):
            path = os.path.join(root, name)
            try:
                process(path)
            except Exception:
                continue
            found.append(path)
            if len(found) == k:
                return found
    raise AssertionError(f"fewer than {k} parsable fixture games")


def test_unexpected_skip_keeps_the_good_games_of_its_batch():
    good1, good2, good3 = _accepted_games(3)
    loader = DataLoader(GATE_GAMES, 2, lambda *_: None, expected_skips=EXPECTED_SKIPS - {KEY})
    loader.file_list = [good1, PATH, good2, good3]
    with pytest.raises(UnexpectedSkip):
        next(loader)
    assert loader.current_index == 2 and loader.pending == [good1]  # past the offending game, good1 kept
    resumed = DataLoader(GATE_GAMES, 2, lambda *_: None, expected_skips=EXPECTED_SKIPS - {KEY})
    resumed.file_list = list(loader.file_list)
    resumed.load_state_dict(loader.state_dict())
    games, _ = next(resumed)
    assert len(games) == 2 and resumed.last_batch_files == [good1, good2]
    next(resumed)
    assert resumed.last_batch_files == [good3]
    with pytest.raises(StopIteration):
        next(resumed)


# ---- T-045: read order per epoch, mixed years, resume ----

QUIET = dict(log_func=lambda *_: None)


def _loader(files, batch_size=1, seed=0, expected_skips=EXPECTED_SKIPS):
    """A loader rooted at GATE_GAMES that reads exactly `files` in epoch 0 (in that order)."""
    dl = DataLoader(GATE_GAMES, batch_size, seed=seed, expected_skips=expected_skips, **QUIET)
    dl.file_list = list(files)
    return dl


def _walk(root):
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith(".log"))


@pytest.mark.parametrize("seed", [0, 1])
def test_epoch0_order_and_train_hash_are_unchanged(seed):
    expected = _walk(GATE_GAMES)
    random.Random(seed).shuffle(expected)  # the pre-T-045 order
    dl = DataLoader(GATE_GAMES, 2, seed=seed, **QUIET)
    assert dl.file_list == expected and dl.epoch == 0
    h = hashlib.sha256()  # the pre-T-045 train_files_sha256 formula
    for f in expected:
        h.update(os.path.relpath(f, GATE_GAMES).encode() + b"\n")
    assert train_files_sha256(DataLoaderCluster([dl], seed=seed)) == h.hexdigest()


def test_each_epoch_has_its_own_order():
    a, b = DataLoader(GATE_GAMES, seed=0, **QUIET), DataLoader(GATE_GAMES, seed=0, **QUIET)
    e0 = list(a.file_list)
    a.next_epoch()
    b.next_epoch()
    assert sorted(a.file_list) == sorted(e0) and a.file_list != e0 and a.file_list == b.file_list
    assert a.epoch == 1 and a.current_index == 0
    e1 = list(a.file_list)
    a.next_epoch()
    assert sorted(a.file_list) == sorted(e0) and a.file_list not in (e0, e1) and a.epoch == 2
    plain = DataLoader(GATE_GAMES, seed=None, **QUIET)
    plain.next_epoch()
    assert plain.file_list == _walk(GATE_GAMES)
    # a str seed is hashed with SHA-512: the same order in every process and Python version
    assert epoch_order(list("abcdefgh"), 0, 1) == ["e", "f", "h", "a", "g", "c", "d", "b"]
    assert epoch_order(list("abcdefgh"), 0, 2) == ["e", "a", "f", "g", "d", "c", "h", "b"]


def test_loader_state_keeps_epoch_and_checks_the_order():
    a = DataLoader(GATE_GAMES, 2, seed=0, **QUIET)
    a.next_epoch()
    a.next_epoch()
    a.current_index, a.accepted = 5, 7
    b = DataLoader(GATE_GAMES, 2, seed=0, **QUIET)
    b.load_state_dict(a.state_dict())
    assert (b.file_list, b.current_index, b.epoch, b.accepted) == (a.file_list, 5, 2, 7)
    wrong = dict(a.state_dict(), order_sha256="0" * 64)
    with pytest.raises(AssertionError, match="order"):
        DataLoader(GATE_GAMES, 2, seed=0, **QUIET).load_state_dict(wrong)


class _SimLoader:
    """A loader without parsing: file i is accepted iff floor((i + 1) p) > floor(i p); a batch reads files until it
    holds batch_size accepted games (or the files run out)."""

    def __init__(self, n, p, batch_size=2):
        self.path, self.file_list, self.p, self.batch_size = "sim", list(range(n)), p, batch_size
        self.current_index, self.pending, self.accepted, self.skipped = 0, [], 0, Counter()
        self.last_batch_files = []

    def left(self):
        return len(self.file_list) - self.current_index + len(self.pending)

    def __next__(self):
        assert self.left() > 0, "the cluster drew a loader with nothing left"
        got = 0
        while got < self.batch_size and self.current_index < len(self.file_list):
            i = self.current_index
            self.current_index += 1
            if math.floor((i + 1) * self.p) > math.floor(i * self.p):
                got += 1
            else:
                self.skipped["sim"] += 1
        self.accepted += got
        return [None] * got, []

    def next_epoch(self):
        raise AssertionError("the draw test stops at the epoch end")


def _simulate(sizes, rates, seed=0):
    """One epoch of a cluster over simulated loaders: (loader of each draw, draw count at which each finished)."""
    loaders = [_SimLoader(n, p) for n, p in zip(sizes, rates)]
    cluster = DataLoaderCluster(loaders, seed=seed)
    picks, finished = [], [None] * len(loaders)
    while any(dl.left() for dl in loaders):
        next(cluster)
        picks.append(cluster.idx)
        for i, dl in enumerate(loaders):
            if finished[i] is None and dl.left() == 0:
                finished[i] = len(picks)
    return picks, finished


def _window_devs(picks, shares):
    """Per 10 % window of the draws (the last one is the low-LR end of a run): max |loader share - its share|."""
    k = len(picks) // 10
    return [max(abs(picks[s:s + k].count(i) / k - share) for i, share in enumerate(shares))
            for s in range(0, len(picks) - k + 1, k)]


SHARE_TOL = 0.08  # measured, 10 seeds: worst window 0.023-0.049 with these weights, last window 0.137-0.186 raw


@pytest.mark.parametrize("sizes, rates", [((5000, 5000), (0.8, 0.6)), ((7500, 2500), (0.7, 0.7))])
def test_cluster_draw_keeps_each_loader_at_its_share(sizes, rates):
    picks, finished = _simulate(sizes, rates)
    assert min(finished) >= 0.95 * len(picks), f"a loader finished at draw {min(finished)} of {len(picks)}"
    accepted = [n * p for n, p in zip(sizes, rates)]
    devs = _window_devs(picks, [a / sum(accepted) for a in accepted])
    assert max(devs) <= SHARE_TOL, devs
    assert _simulate(sizes, rates)[0] == picks and _simulate(sizes, rates, seed=1)[0] != picks


def test_raw_file_weights_fail_the_share_check(monkeypatch):
    """The check above can fail: with weights by raw files left the lower-acceptance loader's share falls toward
    the epoch end (both loaders still finish together, so the finish check alone cannot see it)."""
    monkeypatch.setattr(dataloader_module, "expected_accepted_left", lambda left, accepted, skipped: left)
    picks, _ = _simulate((5000, 5000), (0.8, 0.6))
    assert _window_devs(picks, [4 / 7, 3 / 7])[-1] > SHARE_TOL


def _run(cluster, n):
    """n batches: (cluster epoch, loader epochs, loader index, files) after each."""
    out = []
    for _ in range(n):
        next(cluster)
        out.append((cluster.epoch, [dl.epoch for dl in cluster.dataloaders], cluster.idx,
                    list(cluster.last_batch_files)))
    return out


def test_cluster_epochs_mix_the_loaders_and_resume():
    goods = _accepted_games(6)
    a, b = goods[:3], goods[3:]

    def make():
        return DataLoaderCluster([_loader(a), _loader(list(b) + [PATH])], seed=0)

    cluster = make()
    rows, skips_at_end = [], {}
    while not rows or rows[-1][0] < 2:
        rows += _run(cluster, 1)
        if rows[-1][0] < 2:
            skips_at_end[rows[-1][0]] = sum(cluster.skipped().values())
    for e in (0, 1):
        files = [f for epoch, _, _, fs in rows if epoch == e for f in fs]
        assert sorted(files) == sorted(goods), f"epoch {e} read {files}"
    assert [r[0] for r in rows] == sorted(r[0] for r in rows) and all(r[1] == [r[0]] * 2 for r in rows)
    order = {e: [f for epoch, _, _, fs in rows if epoch == e for f in fs] for e in (0, 1)}
    assert order[0] != order[1]
    idx0 = [i for epoch, _, i, fs in rows if epoch == 0 and fs]
    assert idx0 != sorted(idx0), f"epoch 0 read the loaders one after the other: {idx0}"
    assert skips_at_end == {0: 1, 1: 2}  # the rejected file once per epoch

    # resume mid epoch 1 (its order is rebuilt on load) through a weights_only checkpoint
    cluster = make()
    head = _run(cluster, 9)
    assert head[-1][0] == 1
    buf = io.BytesIO()
    torch.save(cluster.state_dict(), buf)
    buf.seek(0)
    resumed = make()
    resumed.load_state_dict(torch.load(buf, weights_only=True))
    assert _run(resumed, 6) == _run(cluster, 6)
    assert resumed.skipped() == cluster.skipped()


def test_cluster_empty_and_rejected_loaders():
    good = _accepted_games(1)[0]
    cluster = DataLoaderCluster([_loader([]), _loader([good])])
    rows = _run(cluster, 3)
    assert [r[2] for r in rows] == [1, 1, 1] and cluster.epoch == 2  # one file, two epoch boundaries
    assert all(r[3] == [good] for r in rows)
    with pytest.raises(RuntimeError, match="all dataloaders are empty"):
        next(DataLoaderCluster([_loader([]), _loader([])]))
    rejected = DataLoaderCluster([_loader([PATH])])
    for _ in range(2):
        games, _ = next(rejected)
        assert games == []
    assert rejected.epoch == 1 and rejected.skipped()[KEY] == 2


def test_pending_games_keep_their_loader_in_the_epoch():
    good1, good2, good3 = _accepted_games(3)
    strict = EXPECTED_SKIPS - {KEY}

    def make():
        return DataLoaderCluster([_loader([good1, PATH], 2, expected_skips=strict),
                                  _loader([good2, good3], 2, expected_skips=strict)], seed=0)

    cluster = make()
    while True:
        try:
            next(cluster)
        except UnexpectedSkip:
            break
    dl0 = cluster.dataloaders[0]
    assert cluster.idx == 0 and dl0.current_index == len(dl0.file_list) and dl0.pending == [good1]
    resumed = make()
    resumed.load_state_dict(cluster.state_dict())
    for c in (cluster, resumed):
        seen = []
        while good1 not in seen:
            next(c)
            assert c.epoch == 0, "good1 was lost at the epoch boundary"
            seen += c.last_batch_files
        assert c.skipped()[KEY] == 1


def test_pre_t045_state_resumes_one_loader():
    good1, good2, good3, good4 = _accepted_games(4)

    def make(n=1):
        return DataLoaderCluster([_loader([good1, PATH, good2, good3, good4], 2) for _ in range(n)], seed=0)

    cluster = make()
    next(cluster)
    assert cluster.last_batch_files == [good1, good2]
    head_state = {"idx": 0, "loaders": [{"current_index": 3, "seed": 0, "n_files": 5, "skipped": {KEY: 1},
                                         "pending": []}]}  # literal pre-T-045 format
    st = cluster.dataloaders[0].state_dict()
    assert {k: st[k] for k in head_state["loaders"][0]} == head_state["loaders"][0]
    resumed = make()
    resumed.load_state_dict(head_state)
    assert resumed.skipped() == Counter({KEY: 1})
    assert _run(resumed, 3) == _run(cluster, 3)  # crosses the epoch boundary
    with pytest.raises(AssertionError, match="pre-T-045"):
        make(2).load_state_dict({"idx": 0, "loaders": head_state["loaders"] * 2})
