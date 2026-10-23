"""T-055 G1: PositionBatcher against an independent reference.

Synthetic corpora: each .log file holds a JSON spec of its rounds (rows per head), parsed by `synthetic_process` and
built by `SyntheticProcessor`, whose rows carry their identity (game, round, row, head) in the token columns. The
real DataLoader / DataLoaderCluster serve them (serial or worker processes), so epochs, skips, state and worker
failures are the real code. The reference is a plain-Python FIFO of row ids over the same loader order; every step
is compared row by row (provenance -> file, and the tensor contents -> the same ids). Negative controls prove the
comparison fails on a dropped, duplicated, reordered or corrupted row.
Real games: the batcher's tensors equal a FIFO over TransformerTensorProcessor.prepare_batches per game.
Failure boundaries: a stop inside a fill (before or after a draw, inside an admission, at an epoch change), an
UnexpectedSkip after admitted games, a dead worker, a failure while a game enters the buffers, a failure between peek
and commit. After each, a batcher and
cluster restored from their weights_only state (or the same objects, in-process) give the reference's steps and,
after every commit, the reference's cluster and batcher state."""
import collections
import functools
import io
import json
import os
import random
import signal
import time

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import TransformerTensorProcessor, build_game
from MahjongAI.process_xml import InvalidGameException
from MahjongAI.utils import dataloader as DL
from MahjongAI.utils.dataloader import DataLoader, DataLoaderCluster, UnexpectedSkip, epoch_order
from MahjongAI.utils.game_workers import DataWorkerError, GameWorkers
from MahjongAI.utils.position_batcher import HEADS, SIDE_CAP, PositionBatcher, RowBuffer, emit_counts, side_take

DIMS = {"during": 72, "discard": 37, "post": 154}
EXPECTED = SL.EXPECTED_SKIPS | {"invalid:synthetic"}


# ---------------------------------------------------------------- synthetic games
def synthetic_process(filename):
    """A synthetic game: (halfturns, tokens) with one (game id, round, n during, n discard, n post) per round."""
    with open(filename) as f:
        spec = json.load(f)
    if "invalid" in spec:
        raise InvalidGameException(spec["invalid"])
    return [(spec["id"], r, *n) for r, n in enumerate(spec["rounds"])], [None] * len(spec["rounds"])


class SyntheticProcessor(TransformerTensorProcessor):
    """Rows whose token columns 0-3 are (game id, round, row, head index); the mask allows actions 0 and 1; the
    target is row % 2."""

    def build_tensors(self, turns, tokens):
        gid, r, *counts = turns
        out = []
        for head_index, (head, n) in enumerate(zip(HEADS, counts)):
            ids = torch.zeros(n, 150, dtype=torch.int64)
            ids[:, 0], ids[:, 1], ids[:, 2], ids[:, 3] = gid, r, torch.arange(n), head_index
            mask = torch.zeros(n, DIMS[head])
            mask[:, :2] = 1.0
            padding = torch.zeros(n, 150, dtype=torch.bool)
            padding[:, :4] = True
            out.append((ids, (torch.zeros(n, 10, 37, 4), torch.zeros(n, 7, 4), torch.zeros(n, 7)), mask,
                        torch.arange(n) % 2, padding))
        return tuple(out)


def write_corpus(root, specs):
    """specs: per file, a list of (during, discard, post) rounds or an "invalid:<reason>" key."""
    d = root / "20210101" / "20210101_00"
    d.mkdir(parents=True)
    for i, spec in enumerate(specs):
        body = {"id": i, "invalid": spec[len("invalid:"):]} if isinstance(spec, str) else {"id": i, "rounds": spec}
        (d / f"game{i:03d}.log").write_text(json.dumps(body))
    return root


def make_cluster(roots, workers=None, expected=EXPECTED, seed=0):
    loaders = [DataLoader(str(r), 1, lambda *_: None, seed=seed, expected_skips=expected, workers=workers)
               for r in roots]
    return DataLoaderCluster(loaders, seed=0)


def synthetic_workers(n=2):
    return GameWorkers(n, functools.partial(build_game, processor_cls=SyntheticProcessor), window=4,
                       parse=synthetic_process)


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(DL, "process", synthetic_process)


# ---------------------------------------------------------------- the independent reference
def reference_stream(roots, expected=EXPECTED, seed=0):
    """(file, epoch, {head: [row id]}) of each admitted game, in the order of a fresh serial cluster; a row id is
    (<root>/<file>, epoch, round, row)."""
    cluster = make_cluster(roots, expected=expected, seed=seed)
    while True:
        games, _ = next(cluster)
        if not games:
            continue
        file = f"{roots[cluster.idx].name}/{os.path.basename(cluster.last_batch_files[0])}"
        epoch = cluster.dataloaders[cluster.idx].epoch
        rows = {h: [(file, epoch, r, i) for (_, r, *n) in games[0] for i in range(n[k])] for k, h in enumerate(HEADS)}
        yield file, epoch, rows


def reference_steps(stream, sizes, n_steps, peaks=None, tally=None):
    """Per step {head: [row id]}: the emit rule written out in plain Python over FIFO deques. peaks: a dict that
    gets the largest buffer per head (after a fill, before the take). tally: a list that gets, after each step, the
    games read per source epoch and the rows left in each buffer."""
    buf, steps, games = {h: collections.deque() for h in HEADS}, [], collections.Counter()
    for _ in range(n_steps):
        while len(buf["discard"]) < sizes["discard"]:
            _, epoch, rows = next(stream)
            games[epoch] += 1
            for h, ids in rows.items():
                buf[h].extend(ids)
        if peaks is not None:
            peaks.update({h: max(peaks.get(h, 0), len(buf[h])) for h in HEADS})
        step = {}
        for h in HEADS:
            n, size = len(buf[h]), sizes[h]
            k = size if h == "discard" else 2 * size if n >= 2 * size else size if n >= size else 0
            step[h] = [buf[h].popleft() for _ in range(k)]
        steps.append(step)
        if tally is not None:
            tally.append({"games": games.copy(), "buffered": {h: list(buf[h]) for h in HEADS}})
    return steps


def epoch_counter_problems(report, steps, tally) -> list:
    """Problems of a batcher report's per-source-epoch counters against the reference after `steps` (its tally entry
    of that step): games per epoch, and per head rows out (the epochs of the emitted row ids) and rows in (emitted +
    left in the buffer). Empty: equal."""
    def by_epoch(counter):
        return {str(e): n for e, n in sorted(counter.items()) if n}

    problems = []
    if report["games_by_epoch"] != by_epoch(tally["games"]):
        problems.append(f"games by epoch {report['games_by_epoch']}, the reference {by_epoch(tally['games'])}")
    for h in HEADS:
        out = collections.Counter(row[1] for step in steps for row in step[h])
        left = collections.Counter(row[1] for row in tally["buffered"][h])
        if report["rows_out"][h] != by_epoch(out):
            problems.append(f"{h}: rows out {report['rows_out'][h]}, the reference {by_epoch(out)}")
        if report["rows_in"][h] != by_epoch(out + left):
            problems.append(f"{h}: rows in {report['rows_in'][h]}, the reference {by_epoch(out + left)}")
        if report["buffered"][h] != len(tally["buffered"][h]):
            problems.append(f"{h}: {report['buffered'][h]} rows buffered, the reference {len(tally['buffered'][h])}")
    return problems


class Ids:
    """Maps a provenance row (loader, epoch, file index, round, row) to its row id through each loader's epoch
    order, computed here from the sorted file list (not from the loader object)."""

    def __init__(self, roots, seed=0):
        self.names = [r.name for r in roots]
        self.files = [sorted(os.path.join(dp, f) for dp, _, fs in os.walk(str(r)) for f in fs if f.endswith(".log"))
                      for r in roots]
        self.seed, self.orders = seed, {}

    def __call__(self, prov) -> list:
        out = []
        for loader, epoch, index, r, i in prov.tolist():
            key = (loader, epoch)
            if key not in self.orders:
                self.orders[key] = epoch_order(self.files[loader], self.seed, epoch)
            out.append((f"{self.names[loader]}/{os.path.basename(self.orders[key][index])}", epoch, r, i))
        return out


def compare_step(batches, provenance, expected, ids: Ids, sizes) -> list:
    """Problems of one emitted step against the reference step (empty: equal)."""
    problems = []
    for head_index, h in enumerate(HEADS):
        want = expected[h]
        if not want:
            if h in batches or h in provenance:
                problems.append(f"{h}: present, the reference has no {h} row")
            continue
        if h not in batches:
            problems.append(f"{h}: absent, the reference has {len(want)} rows")
            continue
        tokens, (x1, x2, x3), mask, y, padding = batches[h]
        lengths = {len(t) for t in (tokens, x1, x2, x3, mask, y, padding, provenance[h])}
        if lengths != {len(want)}:
            problems.append(f"{h}: {lengths} rows, the reference {len(want)}")
            continue
        if h == "discard" and len(want) != sizes[h] or h != "discard" and len(want) not in (sizes[h], 2 * sizes[h]):
            problems.append(f"{h}: {len(want)} rows is not a fixed step shape")
        got = ids(provenance[h])
        if got != want:
            problems.append(f"{h}: provenance differs from the reference")
        content = [(f"game{g:03d}.log", r, i, k) for g, r, i, k in tokens[:, :4].tolist()]
        if content != [(f.split("/")[1], r, i, head_index) for f, _, r, i in got]:
            problems.append(f"{h}: row contents differ from their provenance")
        if not torch.equal(y, torch.tensor([i % 2 for *_, i in got])):
            problems.append(f"{h}: targets differ from their provenance")
    return problems


def run(batcher, n_steps, ids=None, expected=None, sizes=None, start=0):
    """Peek / commit n_steps steps; with a reference, assert each step equals it. [(row ids per head, read, digest)]"""
    out = []
    for s in range(n_steps):
        batches = batcher.peek()
        prov = batcher.peeked()
        if expected is not None:
            problems = compare_step(batches, prov, expected[start + s], ids, sizes)
            assert not problems, f"step {start + s}: {problems}"
        read, digest = batcher.commit()
        out.append(({h: ids(p) if ids else p.tolist() for h, p in prov.items()}, read, digest))
    return out


def roundtrip(state):
    """A state through torch.save / torch.load(weights_only=True), as a checkpoint carries it."""
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=True)


def assert_state_equal(a, b, where=""):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b), where
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"{where}: keys {a.keys()} vs {b.keys()}"
        for k in a:
            assert_state_equal(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{where}: {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_state_equal(x, y, f"{where}[{i}]")
    else:
        assert a == b, f"{where}: {a!r} vs {b!r}"


def random_specs(n_files, seed, max_rounds=4):
    rng = random.Random(seed)
    return [[(rng.randint(0, 6), rng.randint(0, 30), rng.randint(0, 12)) for _ in range(rng.randint(1, max_rounds))]
            for _ in range(n_files)]


# ---------------------------------------------------------------- the emit rule
def test_side_take_and_emit_counts():
    assert [side_take(n, 10) for n in (0, 9, 10, 19, 20, 35)] == [0, 0, 10, 10, 20, 20]
    sizes = {"during": 4, "discard": 16, "post": 8}
    assert emit_counts({"during": 3, "discard": 16, "post": 17}, sizes) == {"during": 0, "discard": 16, "post": 16}
    with pytest.raises(AssertionError, match="discard buffer 15 < 16"):
        emit_counts({"during": 0, "discard": 15, "post": 0}, sizes)


PATTERNS = {
    # random games over 3 epochs; side heads often short (absent-head steps)
    "random": (random_specs(30, 1), {"during": 6, "discard": 40, "post": 12}, 60),
    # empty games, games of one row, games with side rows only, an all-empty round
    "tiny_and_empty": ([[], [(0, 1, 0)], [(3, 0, 5)], [(0, 0, 0), (0, 2, 1)], [(1, 1, 1)], [(0, 0, 0)]] * 3,
                       {"during": 2, "discard": 3, "post": 4}, 25),
    # one game larger than a step: steps without a read
    "big_game": ([[(5, 150, 30)], [(1, 10, 2)], [(0, 7, 0)]], {"during": 3, "discard": 40, "post": 8}, 20),
    # Codex round 1: 20 games of 200 post rows, then 20 of 36, 434 discard rows each (loader order = file order)
    "codex_burst": ([[(0, 434, 200)]] * 20 + [[(0, 434, 36)]] * 20, {"during": 96, "discard": 1024, "post": 320}, 16),
}


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
def test_steps_equal_the_reference(tmp_path, synthetic, pattern):
    specs, sizes, n_steps = PATTERNS[pattern]
    seed = None if pattern == "codex_burst" else 0
    root = write_corpus(tmp_path / "A", specs)
    peaks, tally = {}, []
    expected = reference_steps(reference_stream([root], seed=seed), sizes, n_steps, peaks, tally)
    batcher = PositionBatcher(make_cluster([root], seed=seed), sizes, SyntheticProcessor())
    ids = Ids([root], seed)
    max_seen = {h: 0 for h in HEADS}
    for s in range(n_steps):
        run(batcher, 1, ids, expected, sizes, start=s)
        max_seen = {h: max(max_seen[h], batcher.max_buffered[h]) for h in HEADS}
        assert all(batcher.buffered()[h] <= SIDE_CAP * sizes[h] for h in ("during", "post"))
        # conservation by source epoch: in = out + buffered, each against the reference's provenance
        problems = epoch_counter_problems(batcher.report(), expected[:s + 1], tally[s])
        assert not problems, f"step {s}: {problems}"
    assert max_seen == peaks, (max_seen, peaks)
    report = batcher.report()
    for h in HEADS:
        assert sum(report["taken"][h].values()) == n_steps
    if pattern == "random":
        assert len(report["games_by_epoch"]) >= 3 and report["taken"]["during"].get("0", 0) > 0
    if pattern == "big_game":
        reads = [r for _, r, _ in run(batcher, 3)]
        assert [] in reads  # a step served from the buffer alone
    if pattern == "codex_burst":
        # Codex's burst left a 1,240-row post backlog under a 1-chunk rule; the 2-chunk rule (640 rows a step while
        # the buffer holds them) peaks lower, well under the 8 x 320 cap
        assert max_seen["post"] < 1240 and max_seen["post"] <= SIDE_CAP * sizes["post"], max_seen
        assert report["taken"]["during"] == {"0": n_steps}


class EpochZero(PositionBatcher):
    """A mutant that credits every emitted row to source epoch 0 (totals unchanged)."""

    def commit(self):
        out = super().commit()
        self.rows_out = {h: {0: sum(self.rows_out[h].values())} if self.rows_out[h] else {} for h in HEADS}
        return out


def test_per_epoch_counters_fail_a_batcher_crediting_every_row_to_epoch_0(tmp_path, synthetic):
    """The mutant keeps every total (in - out = buffered, rows out = rows emitted): only the per-epoch comparison
    against the reference provenance catches it, once the run is past epoch 0."""
    specs, sizes, n_steps = PATTERNS["random"]
    root = write_corpus(tmp_path / "A", specs)
    tally = []
    expected = reference_steps(reference_stream([root]), sizes, n_steps, tally=tally)
    batcher = EpochZero(make_cluster([root]), sizes, SyntheticProcessor())
    run(batcher, n_steps, Ids([root]), expected, sizes)
    report = batcher.report()
    for h in HEADS:  # the totals hold
        n_in, n_out = sum(report["rows_in"][h].values()), sum(report["rows_out"][h].values())
        assert n_in - n_out == report["buffered"][h] and n_out == sum(len(e[h]) for e in expected)
    problems = epoch_counter_problems(report, expected, tally[-1])
    assert problems and all("rows out" in p for p in problems), problems


def test_negative_controls_fail_the_comparison(tmp_path, synthetic):
    """A dropped, duplicated, reordered or corrupted row (tensors and provenance changed together where a bug would)
    each fails compare_step; the unchanged step passes."""
    specs, sizes, _ = PATTERNS["random"]
    root = write_corpus(tmp_path / "A", specs)
    expected = reference_steps(reference_stream([root]), sizes, 1)[0]
    batcher = PositionBatcher(make_cluster([root]), sizes, SyntheticProcessor())
    batches, prov, ids = batcher.peek(), batcher.peeked(), Ids([root])
    assert compare_step(batches, prov, expected, ids, sizes) == []

    def changed(index_fn, corrupt=False):
        b = {h: (t[0].clone(), t[1], t[2], t[3], t[4]) for h, t in batches.items()}
        p = dict(prov)
        if corrupt:
            b["discard"][0][0, 2] += 1
            return b, p
        idx = index_fn(len(prov["discard"]))
        cols = [c[idx] for c in (b["discard"][0], *b["discard"][1], *b["discard"][2:])]
        b["discard"] = (cols[0], (cols[1], cols[2], cols[3]), cols[4], cols[5], cols[6])
        p["discard"] = prov["discard"][idx]
        return b, p

    controls = {
        "dropped": changed(lambda n: torch.arange(n - 1)),
        "duplicated": changed(lambda n: torch.cat([torch.tensor([0]), torch.arange(n - 1)])),
        "reordered": changed(lambda n: torch.cat([torch.tensor([1, 0]), torch.arange(2, n)])),
        "corrupted": changed(None, corrupt=True),
    }
    for name, (b, p) in controls.items():
        assert compare_step(b, p, expected, ids, sizes), f"{name}: not detected"


# ---------------------------------------------------------------- memory guard and progress
def test_side_buffer_above_the_cap_raises_and_drops_nothing(tmp_path, synthetic):
    sizes = {"during": 4, "discard": 40, "post": 10}
    root = write_corpus(tmp_path / "A", [[(0, 40, 50)]] * 10)
    batcher = PositionBatcher(make_cluster([root]), sizes, SyntheticProcessor())
    with pytest.raises(RuntimeError, match=r"position buffers \{.*'post': \d+\} exceed 8 x the step size .* on "
                                           r"\['post'\]; games admitted \d+"):
        for _ in range(20):
            batcher.peek()
            batcher.commit()
    buffered = batcher.buffered()["post"]
    assert buffered > SIDE_CAP * sizes["post"]
    assert sum(batcher.rows_in["post"].values()) - sum(batcher.rows_out["post"].values()) == buffered


def test_side_buffer_cap_is_checked_after_each_admission(tmp_path, synthetic):
    """Games of one discard row and 5 post rows: the post buffer passes 8 x 1 at the 2nd admission, long before the
    40-row discard fill ends. The cap raises there, not after the fill (Codex round 2)."""
    sizes = {"during": 1, "discard": 40, "post": 1}
    root = write_corpus(tmp_path / "A", [[(0, 1, 5)]] * 50)
    batcher = PositionBatcher(make_cluster([root]), sizes, SyntheticProcessor())
    with pytest.raises(RuntimeError, match=r"exceed 8 x the step size .* on \['post'\]; games admitted 2$"):
        batcher.peek()
    assert batcher.buffered() == {"during": 0, "discard": 2, "post": 10}


@pytest.mark.parametrize("specs", [["invalid:synthetic"] * 4, [[(1, 0, 1)]] * 4],  # side rows under the cap
                         ids=["every_game_skipped", "no_discard_row"])
def test_an_epoch_without_a_discard_row_raises(tmp_path, synthetic, specs):
    root = write_corpus(tmp_path / "A", specs)
    batcher = PositionBatcher(make_cluster([root]), {"during": 1, "discard": 4, "post": 1}, SyntheticProcessor())
    with pytest.raises(RuntimeError, match=r"a full pass over the training files \(epoch 0\) admitted no discard row"):
        batcher.peek()
    assert batcher.cluster.epoch == 1


def test_tiny_games_are_progress(tmp_path, synthetic):
    """One file with one discard row: a 5-row step spans 5 epochs, and the next ones keep going."""
    root = write_corpus(tmp_path / "A", [[(0, 1, 0)]])
    sizes = {"during": 1, "discard": 5, "post": 1}
    expected = reference_steps(reference_stream([root]), sizes, 3)
    batcher = PositionBatcher(make_cluster([root]), sizes, SyntheticProcessor())
    run(batcher, 3, Ids([root]), expected, sizes)
    assert batcher.report()["games_by_epoch"] == {str(e): 1 for e in range(15)}


# ---------------------------------------------------------------- real games, several loaders, workers
def _prepare_reference(roots, sizes, n_steps):
    """Per step {head: 7 tensors}: per game prepare_batches (a fresh serial cluster), every game's rows concatenated
    per head in admission order, then sliced by the reference emit rule."""
    cluster = make_cluster(roots, expected=SL.EXPECTED_SKIPS)
    processor = TransformerTensorProcessor()
    cols = {h: [] for h in HEADS}
    taken = {h: 0 for h in HEADS}
    have = {h: 0 for h in HEADS}
    steps = []
    for _ in range(n_steps):
        while have["discard"] - taken["discard"] < sizes["discard"]:
            games, tokens = next(cluster)
            if not games:
                continue
            for h, t in zip(HEADS, processor.prepare_batches(games, tokens)):
                cols[h].append((t[0], *t[1], t[2], t[3], t[4]))
                have[h] += len(t[3])
        step = {}
        for h in HEADS:
            left = have[h] - taken[h]
            k = sizes[h] if h == "discard" else 2 * sizes[h] if left >= 2 * sizes[h] else sizes[h] if left >= sizes[h] else 0
            if k:
                full = [torch.cat([c[i] for c in cols[h]]) for i in range(7)]
                step[h] = [c[taken[h]:taken[h] + k] for c in full]
            taken[h] += k
        steps.append(step)
    return steps


def test_real_games_equal_a_prepare_batches_fifo_serial_and_workers(small_roots):
    """Two loaders (mixed draws, expected skips, epoch changes): every step's tensors equal the reference's; the
    worker path gives the serial path's tensors, provenance, digests and state."""
    sizes = {"during": 16, "discard": 300, "post": 60}
    n_steps = 22
    expected = _prepare_reference(small_roots, sizes, n_steps)
    serial = PositionBatcher(make_cluster(small_roots, expected=SL.EXPECTED_SKIPS), sizes)
    with GameWorkers(2, build_game, window=4) as w:
        fast = PositionBatcher(make_cluster(small_roots, w, expected=SL.EXPECTED_SKIPS), sizes)
        for s in range(n_steps):
            a, b = serial.peek(), fast.peek()
            assert a.keys() == b.keys() == {h for h in HEADS if h in expected[s]}, s
            for h in a:
                flat_a = [a[h][0], *a[h][1], *a[h][2:]]
                flat_b = [b[h][0], *b[h][1], *b[h][2:]]
                for i, (x, y, z) in enumerate(zip(flat_a, flat_b, expected[s][h])):
                    assert x.dtype == z.dtype and torch.equal(x, z) and torch.equal(y, z), (s, h, i)
            pa, pb = serial.peeked(), fast.peeked()
            assert pa.keys() == pb.keys() and all(torch.equal(pa[h], pb[h]) for h in pa)
            assert serial.commit() == fast.commit()
            assert_state_equal(serial.state_dict(), fast.state_dict(), f"batcher after step {s}")
            assert serial.cluster.state_dict() == fast.cluster.state_dict()
    report = serial.report()
    assert len(report["games_by_epoch"]) >= 2 and serial.cluster.draws > 0
    loaders = {int(l) for p in serial.state_dict()["buffers"].values() if len(p["provenance"])
               for l in p["provenance"][:, 0]} | {0}
    assert sum(serial.cluster.skipped().values()) > 0 and loaders <= {0, 1}


def test_synthetic_two_loaders_equal_the_reference(tmp_path, synthetic):
    a = write_corpus(tmp_path / "A", random_specs(12, 2))
    b = write_corpus(tmp_path / "B", random_specs(9, 3))
    sizes = {"during": 5, "discard": 30, "post": 10}
    expected_ids = reference_steps(reference_stream([a, b]), sizes, 45)
    batcher = PositionBatcher(make_cluster([a, b]), sizes, SyntheticProcessor())
    steps = run(batcher, 45, Ids([a, b]), expected_ids, sizes)
    roots_seen = {row[0].split("/")[0] for ids, _, _ in steps for rows in ids.values() for row in rows}
    assert roots_seen == {"A", "B"} and batcher.cluster.epoch >= 1


# ---------------------------------------------------------------- failure boundaries
SIZES = {"during": 3, "discard": 30, "post": 8}  # ~2-3 synthetic games per step: a fill spans several draws


def _reference_run(roots, n_steps, expected=EXPECTED):
    """The unbroken run: per step (row ids, read, digest) and the (cluster, batcher) state after its commit. Each step
    and the per-epoch counters after it equal the independent reference (reference_steps)."""
    tally = []
    want = reference_steps(reference_stream(roots, expected=expected), SIZES, n_steps, tally=tally)
    batcher = PositionBatcher(make_cluster(roots, expected=expected), SIZES, SyntheticProcessor())
    ids, out = Ids(roots), []
    for s in range(n_steps):
        out.append((run(batcher, 1, ids, want, SIZES, start=s)[0], roundtrip(batcher.cluster.state_dict()),
                    roundtrip(batcher.state_dict())))
        problems = epoch_counter_problems(batcher.report(), want[:s + 1], tally[s])
        assert not problems, f"reference run, step {s}: {problems}"
    return out


def _continue_and_compare(batcher, reference, ids, start):
    for s in range(start, len(reference)):
        step = run(batcher, 1, ids)[0]
        assert step == reference[s][0], f"step {s}"
        assert batcher.cluster.state_dict() == reference[s][1], f"cluster after step {s}"
        assert_state_equal(roundtrip(batcher.state_dict()), reference[s][2], f"batcher after step {s}")


def _restored(roots, state, workers=None, expected=EXPECTED):
    batcher = PositionBatcher(make_cluster(roots, workers, expected=expected), SIZES, SyntheticProcessor())
    batcher.cluster.load_state_dict(state[0])
    batcher.load_state_dict(state[1])
    return batcher


class Interrupting:
    """Raises KeyboardInterrupt at the k-th call of a wrapped function (before or after the real call)."""

    def __init__(self, fn, k, after=False):
        self.fn, self.k, self.after, self.calls = fn, k, after, 0

    def __call__(self, *a, **kw):
        self.calls += 1
        if self.calls == self.k and not self.after:
            raise KeyboardInterrupt(f"stop at call {self.k}")
        out = self.fn(*a, **kw)
        if self.calls == self.k:
            raise KeyboardInterrupt(f"stop after call {self.k}")
        return out


@pytest.mark.parametrize("where", ["before_draw", "after_draw", "inside_admission", "epoch_change"])
@pytest.mark.parametrize("resume", ["restored", "in_process"])
def test_a_stop_inside_a_fill_continues_as_the_unbroken_run(tmp_path, synthetic, monkeypatch, where, resume):
    specs = random_specs(14, 4, max_rounds=2)
    root = write_corpus(tmp_path / "A", specs)
    reference = _reference_run([root], 16)
    batcher = PositionBatcher(make_cluster([root]), SIZES, SyntheticProcessor())
    ids = Ids([root])
    run(batcher, 2, ids)  # two steps done: the buffers carry rows
    # the draw that starts epoch 1, or the 2nd draw from here; interrupted before or after the real draw
    k = batcher.cluster.dataloaders[0].left() + 1 if where == "epoch_change" else 2
    with monkeypatch.context() as m:
        if where == "inside_admission":
            processor = batcher.processor
            m.setattr(processor, "build_tensors", Interrupting(processor.build_tensors, 1, after=True))
        else:
            cls = type(batcher.cluster)
            wrapped = Interrupting(functools.partial(cls.__next__, batcher.cluster), k, where != "before_draw")
            m.setattr(batcher.cluster, "__class__", type("Stopping", (cls,), {"__next__": lambda self: wrapped()}))
        with pytest.raises(KeyboardInterrupt):
            for _ in range(10):
                run(batcher, 1, ids)
    if where == "epoch_change":  # the epoch change was undone with its draw
        assert batcher.cluster.epoch == 0 and batcher.cluster.dataloaders[0].left() == 0
    state = (roundtrip(batcher.cluster.state_dict()), roundtrip(batcher.state_dict()))
    done = sum(state[1]["taken"]["discard"].values())
    assert done >= 2 and len(batcher.buffers["discard"]) < SIZES["discard"]  # stopped inside a fill
    if resume == "restored":
        batcher = _restored([root], state)
    _continue_and_compare(batcher, reference, ids, done)


def test_unexpected_skip_after_admitted_games_keeps_them(tmp_path, synthetic):
    """The 6th file of epoch 0 is an unexpected skip: the fill raises after admitting earlier games of the step. The
    reference expects that skip (counted, continues). Restored from the stop's state, the run gives the reference's
    steps and states."""
    specs = random_specs(30, 5, max_rounds=2)
    root = write_corpus(tmp_path / "A", specs)
    order = epoch_order(sorted(str(p) for p in root.glob("*/*/*.log")), 0, 0)
    bad = int(os.path.basename(order[5])[4:7])
    specs[bad] = "invalid:unexpected"
    (root / "20210101" / "20210101_00" / f"game{bad:03d}.log").write_text(json.dumps({"id": bad, "invalid": "unexpected"}))
    reference = _reference_run([root], 8, expected=EXPECTED | {"invalid:unexpected"})
    batcher = PositionBatcher(make_cluster([root]), SIZES, SyntheticProcessor())
    ids = Ids([root])
    with pytest.raises(UnexpectedSkip, match="unexpected skip 'invalid:unexpected'"):
        for _ in range(8):
            run(batcher, 1, ids)
    done = sum(batcher.taken["discard"].values())
    assert sum(batcher.games_admitted.values()) > 0 and len(batcher.buffers["discard"]) < SIZES["discard"]
    state = (roundtrip(batcher.cluster.state_dict()), roundtrip(batcher.state_dict()))
    assert state[1]["buffers"]["discard"]["cols"] is not None  # admitted rows of the cut fill are in the state
    _continue_and_compare(_restored([root], state), reference, ids, done)


def test_a_dead_worker_inside_a_fill_continues_as_the_unbroken_run(tmp_path, synthetic):
    specs = random_specs(20, 6, max_rounds=2)
    root = write_corpus(tmp_path / "A", specs)
    reference = _reference_run([root], 10)
    ids = Ids([root])
    w = synthetic_workers()
    batcher = PositionBatcher(make_cluster([root], w), SIZES, SyntheticProcessor())
    run(batcher, 3, ids)
    os.kill(w.pids()[0], signal.SIGKILL)
    time.sleep(0.5)
    with pytest.raises(DataWorkerError, match="died"):
        for _ in range(5):
            run(batcher, 1, ids)
    done = sum(batcher.taken["discard"].values())
    state = (roundtrip(batcher.cluster.state_dict()), roundtrip(batcher.state_dict()))
    w.close()
    with synthetic_workers() as w2:
        _continue_and_compare(_restored([root], state, w2), reference, ids, done)


@pytest.mark.parametrize("resume", ["restored", "in_process"])
def test_a_failure_while_a_game_enters_the_buffers_undoes_its_draw(tmp_path, synthetic, monkeypatch, resume):
    """A MemoryError at the discard buffer's append (the 2nd head, after the during rows went in) in step 3's 2nd
    admission: the batcher and cluster states equal those before that draw (no partial game, the loader not past the
    file), and the file is drawn again: the run continues as the unbroken run (Codex round 2)."""
    root = write_corpus(tmp_path / "A", [[(1 + i % 3, 8 + i % 5, 2 + i % 4)] for i in range(20)])
    reference = _reference_run([root], 10)
    batcher = PositionBatcher(make_cluster([root]), SIZES, SyntheticProcessor())
    ids = Ids([root])
    run(batcher, 2, ids)
    append, calls, before = RowBuffer.append, [0], []

    def failing_append(self, cols, provenance):
        if self is batcher.buffers["discard"]:
            calls[0] += 1
            if calls[0] == 2:
                raise MemoryError("forced allocation failure at the discard append")
        append(self, cols, provenance)

    draw = batcher._draw

    def recording_draw():
        before.append((roundtrip(batcher.cluster.state_dict()), roundtrip(batcher.state_dict())))
        draw()

    with monkeypatch.context() as m:
        m.setattr(RowBuffer, "append", failing_append)
        m.setattr(batcher, "_draw", recording_draw)
        with pytest.raises(MemoryError, match="forced allocation failure"):
            run(batcher, 1, ids)
    assert calls[0] == 2
    state = (roundtrip(batcher.cluster.state_dict()), roundtrip(batcher.state_dict()))
    assert state[0] == before[-1][0], "the cluster is past the file that failed to enter"
    assert_state_equal(state[1], before[-1][1], "batcher after the failed admission")
    if resume == "restored":
        batcher = _restored([root], state)
    _continue_and_compare(batcher, reference, ids, 2)


@pytest.mark.parametrize("resume", ["restored", "in_process"])
def test_a_failure_between_peek_and_commit_keeps_the_step(tmp_path, synthetic, resume):
    root = write_corpus(tmp_path / "A", random_specs(20, 7, max_rounds=2))
    reference = _reference_run([root], 10)
    ids = Ids([root])
    batcher = PositionBatcher(make_cluster([root]), SIZES, SyntheticProcessor())
    run(batcher, 3, ids)
    first = batcher.peek()
    prov = batcher.peeked()
    state = (roundtrip(batcher.cluster.state_dict()), roundtrip(batcher.state_dict()))  # a stop before commit
    if resume == "restored":
        batcher = _restored([root], state)
    again = batcher.peek()
    assert again.keys() == first.keys() and all(torch.equal(batcher.peeked()[h], prov[h]) for h in prov)
    assert all(torch.equal(again[h][0], first[h][0]) for h in first)
    _continue_and_compare(batcher, reference, ids, 3)


def test_state_without_a_peek_and_commit_without_peek(tmp_path, synthetic):
    root = write_corpus(tmp_path / "A", random_specs(5, 8))
    batcher = PositionBatcher(make_cluster([root]), SIZES, SyntheticProcessor())
    with pytest.raises(AssertionError, match="commit without peek"):
        batcher.commit()
    empty = roundtrip(batcher.state_dict())
    assert empty["buffers"]["discard"]["cols"] is None and empty["read_since_update"] == []
    with pytest.raises(AssertionError, match="checkpoint position sizes"):
        PositionBatcher(make_cluster([root]), dict(SIZES, post=9), SyntheticProcessor()).load_state_dict(empty)
