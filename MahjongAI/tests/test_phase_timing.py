"""T-057 --phase-timing and --measure-window, on the CPU (fixture games, positions mode).

Trainer runs (one module fixture): B (timing + window 2,3), A (window only), P5 (neither, stop 5) and P2 (neither,
stop 2). B, A and P5 train bit for bit alike (per-step lines, metrics rows, checkpoints); A builds no probe or timer,
P5 no window either, and P5 / P2 build their data workers without the T-057 clocks (no instrumentation at all).
Accounting (C1): B's phase sum over the window is within 5 % of its independent outer time, and time outside every
span (a sleep in the trainer's code between two spans, or an operation whose span begin is missing) fails C1. The
window's endpoint time is taken right after the scheduler step of updates W and W+M. Denominators: the window's rows
per head equal the batcher counters at step 5 (metrics row) minus those at step 2 (P2's checkpoint). Unit level: the
timer's spans, chains and wait split, the window's endpoint times and memory samples, the window line's own row check
and its round trip through the parser, the data workers' exposed wait (only the blocked waits) and busy time, and the
gate arithmetic (C1, C2, D1 with its repeat band, the epoch projection, the window hashes)."""
import time

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.tests.conftest import build_train_data_root
from MahjongAI.tests.test_positions_train import _run, step_lines
from MahjongAI.tests.test_resume_rows import _assert_equal
from MahjongAI.utils import phase_probe as PP
from MahjongAI.utils import phase_timing as PT
from MahjongAI.utils.game_workers import GameWorkers

STEPS, LAST, WINDOW = 12, 5, (2, 3)  # eval every 5 (_run); save every 5: one eval + save at the window's last step


def run(tmp_path, monkeypatch, root, name, stop=LAST, **kw):
    return _run(tmp_path, monkeypatch, root, name, steps=STEPS, save_every=LAST, stop_after_step=stop, **kw)


def parsed(tmp_path, name) -> dict:
    return PT.parse_run(open(tmp_path / f"{name}.log"))


class _Workers(GameWorkers):
    """The training pool of a run, kept for its timing flag and clocks after the run."""
    made = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _Workers.made.append(self)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The four trainer runs; B and A must not build what their flags leave off (a fail on construction). Returns
    (tmp, root, {run: its training pool}, B's (mark / window endpoint) events)."""
    tmp = tmp_path_factory.mktemp("timing")
    root = build_train_data_root(tmp / "data")
    pools, events = {}, []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SL, "GameWorkers", _Workers)
        mark, update_done = PP.PhaseProbe.mark, PT.MeasureWindow.update_done

        def mark_spy(self, name, step=None, **k):
            events.append(("mark", name, self.step if step is None else step))
            return mark(self, name, step=step, **k)

        def update_done_spy(self, step):
            events.append(("window", step))
            return update_done(self, step)
        with pytest.MonkeyPatch.context() as spies:
            spies.setattr(PP.PhaseProbe, "mark", mark_spy)
            spies.setattr(PT.MeasureWindow, "update_done", update_done_spy)
            for name, kw in (("B", {"phase_timing": True, "measure_window": WINDOW}),):
                _Workers.made = []
                run(tmp, mp, root, name, **kw)
                pools[name] = _Workers.made[0]  # the training pool is made first
        mp.setattr(SL, "PhaseProbe", lambda *a, **k: pytest.fail("a run without --phase-timing built a probe"))
        mp.setattr(SL, "PhaseTimer", lambda *a, **k: pytest.fail("a run without --phase-timing built a timer"))
        for name, kw in (("A", {"measure_window": WINDOW}), ("P5", {}), ("P2", {"stop": WINDOW[0]})):
            if name == "P5":
                mp.setattr(SL, "MeasureWindow", lambda *a, **k: pytest.fail("a run without --measure-window built one"))
            _Workers.made = []
            run(tmp, mp, root, name, **kw)
            pools[name] = _Workers.made[0]
    return tmp, root, pools, events


def test_workers_time_only_with_a_flag(runs):
    """P5 / P2 (both flags off): workers without the T-057 clocks; B / A (a flag on): with them."""
    _, _, pools, _ = runs
    assert {n: p.timing for n, p in pools.items()} == {"B": True, "A": True, "P5": False, "P2": False}
    for n in ("P5", "P2"):
        assert pools[n].busy == [] and pools[n].exposed_wait_s == 0.0 and pools[n].busy_seconds() == [], n
    for n in ("B", "A"):
        assert len(pools[n].busy) == SL.DATA_WORKERS and pools[n].exposed_wait_s > 0, n


def test_timing_and_window_train_exactly_as_off(runs):
    tmp, _, _, _ = runs
    lines = {n: step_lines(tmp / f"{n}.log") for n in ("B", "A", "P5")}
    assert sorted(lines["P5"]) == list(range(1, LAST + 1)) and lines["B"] == lines["A"] == lines["P5"]
    rows = {n: SL.Metrics(str(tmp / n / "metrics.jsonl")).read()[1] for n in ("B", "A", "P5")}
    assert [r["step"] for r in rows["P5"]] == [0, LAST]
    _assert_equal(rows["B"], rows["P5"], "rows B")
    _assert_equal(rows["A"], rows["P5"], "rows A")
    saved = sorted(p.name for p in (tmp / "P5" / "checkpoints").iterdir())
    assert saved == [f"step_{LAST}.pt"]
    for n in ("B", "A"):
        assert sorted(p.name for p in (tmp / n / "checkpoints").iterdir()) == saved
        _assert_equal(torch.load(tmp / n / "checkpoints" / saved[0], weights_only=True),
                      torch.load(tmp / "P5" / "checkpoints" / saved[0], weights_only=True), f"{n} {saved[0]}")
    b, a, p = parsed(tmp, "B"), parsed(tmp, "A"), parsed(tmp, "P5")
    assert sorted(b["steps"]) == list(range(1, LAST + 1)) and not a["steps"] and not p["steps"]
    assert a["end"] is not None and p["end"] is None and set(a["timed"]) == set(b["timed"]) == {"eval", "save"}
    assert PT.window_hashes(b) == PT.window_hashes(a)


def test_phase_sum_matches_the_outer_timer(runs):
    tmp, _, _, _ = runs
    b = parsed(tmp, "B")
    totals = PT.phase_totals(b)
    ratio = PT.check_c1(totals["sum"], b["end"]["outer"])
    for k in PT.window_steps(b):
        s = b["steps"][k]
        assert s["other"] == 0 and min(s[p] for p in PT.PHASES) >= 0, (k, s)  # no eval / save inside the window
        assert abs(s["sum"] - sum(s[p] for p in PT.PHASES)) < 1e-5, (k, s)
        assert abs(s["wall"] - s["sum"] - s["other"] - s["unattributed"]) < 1e-5 and s["unattributed"] >= 0, (k, s)
    assert b["steps"][1]["other"] > 0  # step 1's wall holds the step-0 eval (init -> eval_weights -> eval -> boundary)
    print(f"C1 ratio {ratio:.4f}, unattributed {totals['unattributed']:.6f} s of wall {totals['wall']:.6f} s")


@pytest.mark.parametrize("where", ["between_spans", "no_begin"])
def test_c1_detects_unattributed_time(tmp_path, monkeypatch, train_data_root, where):
    """between_spans: a 0.2 s sleep in the trainer's code after the backward span and before the clip span starts.
    no_begin: a 0.2 s clip whose span begin is missing (forgotten instrumentation). Either way the time is in the outer
    timer and in no phase: C1 fails."""
    begin = PT.PhaseTimer.begin
    if where == "between_spans":
        def spy(self, name):
            if name == "clip":
                time.sleep(0.2)  # runs before begin's sync and clock: outside every span
            return begin(self, name)
    else:
        clip = torch.nn.utils.clip_grad_norm_

        def slow_clip(*a, **k):
            time.sleep(0.2)
            return clip(*a, **k)
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", slow_clip)

        def spy(self, name):
            return None if name == "clip" else begin(self, name)
    monkeypatch.setattr(PT.PhaseTimer, "begin", spy)
    run(tmp_path, monkeypatch, train_data_root, "U", phase_timing=True, measure_window=WINDOW)
    u = parsed(tmp_path, "U")
    totals = PT.phase_totals(u)
    with pytest.raises(PT.CheckError, match="C1"):
        PT.check_c1(totals["sum"], u["end"]["outer"])
    assert totals["unattributed"] >= 0.2 * WINDOW[1] and totals["update"] < 0.2 * WINDOW[1]


def test_window_endpoints_are_the_update_completions(runs):
    """B: the window's endpoint times are taken right after each update's scheduler step (sched_step mark), before
    its commit, at every update; and the memory samples come from updates W and W+M."""
    tmp, _, _, events = runs
    for k in range(1, LAST + 1):
        assert events.count(("window", k)) == 1, k
        i = events.index(("window", k))
        around = events[i - 2:i + 3]
        assert events[i - 1] == ("mark", "sched_step", k) and events[i + 1] == ("mark", "commit", k), around
    for n in ("A", "B"):
        mem = parsed(tmp, n)["memory"]
        assert {e: (m["step"], sorted(m["mb"])) for e, m in mem.items()} == {
            "start": (WINDOW[0], ["rss"]), "end": (LAST, ["rss"])}, mem
        assert min(m["mb"]["rss"] for m in mem.values()) > 0


def test_window_rows_match_the_batcher_counters(runs):
    tmp, _, _, _ = runs
    end = parsed(tmp, "A")["end"]
    at_last = SL.Metrics(str(tmp / "A" / "metrics.jsonl")).read()[1][-1]["positions"]
    at_warmup = torch.load(tmp / "P2" / "checkpoints" / f"step_{WINDOW[0]}.pt", weights_only=True)["batcher_state"]
    for h in PT.HEADS:
        assert end["rows"][h] == (sum(at_last["rows_out"][h].values()) - sum(at_warmup["rows_out"][h].values())), h
        assert end["admitted"][h] == (sum(at_last["rows_in"][h].values()) - sum(at_warmup["rows_in"][h].values())), h
    assert end["games"] == at_last["games_admitted"] - sum(at_warmup["games_admitted"].values())
    assert end["rows"]["total"] == sum(end["rows"][h] for h in PT.HEADS) > 0
    assert end["ms_per_1k"] == pytest.approx(end["outer"] / end["rows"]["total"] * 1e6, rel=1e-3)
    assert (end["first"], end["last"]) == (WINDOW[0] + 1, LAST) and end["workers"] == SL.DATA_WORKERS
    assert len(end["busy"]) == len(end["files"]) == SL.DATA_WORKERS and min(end["busy"]) >= 0


def test_window_needs_its_protocol(tmp_path, monkeypatch, train_data_root):
    with pytest.raises(AssertionError, match="--stop-after-step 5"):
        run(tmp_path, monkeypatch, train_data_root, "bad", stop=6, measure_window=WINDOW)


# ---------------------------------------------------------------- unit level (no trainer run)
def marks(seq):
    """[PP.Mark] from [(step, name, t)]."""
    return [PP.Mark(s, n, 0, 0, t) for s, n, t in seq]


def _drive(timer, events, clock, wait=None):
    """Feed the timer [("begin", name, t) | (step, name, t)] as the trainer and the probe would; wait: {mark name:
    the workers' wait added just before that mark}."""
    prev = None
    for e in events:
        if e[0] == "begin":
            clock[0] = e[2]
            timer.begin(e[1])
            continue
        m = PP.Mark(e[0], e[1], 0, 0, e[2])
        clock[1] += (wait or {}).get(e[1], 0.0)
        if prev is None:
            timer.start(m)
        else:
            timer.interval(prev, m)
        prev = m


def test_timer_times_spans_and_leaves_the_gaps_unattributed():
    lines, clock = [], [0.0, 0.0]  # [span start clock, workers' wait clock]
    timer = PT.PhaseTimer(lines.append, lambda: clock[1], clock=lambda: clock[0])
    step1 = [(0, "init", 0.0), (0, "eval_weights", 1.0), (0, "eval", 2.0), (0, "boundary", 2.5), (1, "top", 2.6),
             ("begin", "peek", 2.7), (1, "peek", 3.6), ("begin", "rng", 3.65), (1, "rng", 3.7),
             (1, "discard:transfer", 3.9), (1, "discard:forward", 4.4), (1, "results", 4.5),
             ("begin", "loss", 4.55), (1, "loss", 4.6), ("begin", "finite", 4.7), (1, "finite", 4.8),
             ("begin", "zero_grad", 4.8), (1, "zero_grad", 4.9), ("begin", "backward", 4.9), (1, "backward", 5.9),
             ("begin", "clip", 5.95), (1, "clip", 6.0), ("begin", "opt_step", 6.0), (1, "opt_step", 6.2),
             ("begin", "sched_step", 6.2), (1, "sched_step", 6.3), ("begin", "commit", 6.35), (1, "commit", 6.5),
             ("begin", "sums", 6.5), (1, "sums", 6.6), ("begin", "log", 6.6), (1, "log", 6.8)]
    step2 = [(2, "top", 6.9), (2, "peek", 7.8), ("begin", "rng", 7.8), (2, "rng", 7.9), (2, "post:transfer", 8.0),
             (2, "post:forward", 8.3), (2, "results", 8.3), ("begin", "log", 8.5), (2, "log", 9.0)]  # peek: no begin
    _drive(timer, step1 + step2, clock, wait={"peek": 0.75})
    assert len(lines) == 2
    got = [PT.STEP_TIME_RE.match(f"[ts] {ln}") for ln in lines]
    want = {"wait": 0.75, "admit": 0.15 + 0.15, "transfer": 0.2, "forward": 0.5 + 0.1 + 0.05, "backward": 1.0,
            "update": 0.1 + 0.05 + 0.2 + 0.1, "overhead": 0.05 + 0.1 + 0.1 + 0.2, "other": 2.5, "sum": 3.8,
            "unattributed": 0.2 + 0.05 + 0.05 + 0.1 + 0.05 + 0.05, "wall": 6.8}
    for k, v in want.items():
        assert float(got[0][k]) == pytest.approx(v, abs=1e-6), (k, lines[0])
    # step 2: the peek without its begin (its wait clock change included) is unattributed; the wall starts at log 1
    want = {"wait": 0.0, "admit": 0.0, "transfer": 0.1, "forward": 0.3 + 0.0, "overhead": 0.1 + 0.5, "sum": 1.0,
            "unattributed": 1.2, "wall": 2.2}
    for k, v in want.items():
        assert float(got[1][k]) == pytest.approx(v, abs=1e-6), (k, lines[1])


def test_timer_rejects_a_crossed_span():
    timer, clock = PT.PhaseTimer(lambda line: None, clock=lambda: 0.0), [0.0, 0.0]
    with pytest.raises(AssertionError, match="while the span 'clip' is open"):
        _drive(timer, [(1, "init", 0.0), ("begin", "clip", 0.1), (1, "opt_step", 0.2)], clock)
    with pytest.raises(AssertionError, match="begin 'opt_step' while the span 'clip' is open"):
        timer.begin("opt_step")


def test_every_step_mark_has_a_phase():
    for name in PP.step_marks(PP.HEADS):
        assert PT.phase_of(name) in PT.PHASES, name
    assert all(PT.phase_of(n) is None for n in PT.OUTSIDE_STEP)


class _Batcher:
    def __init__(self, out, inn, games):
        self.rows_out = {h: {0: v} for h, v in out.items()}
        self.rows_in = {h: {0: v} for h, v in inn.items()}
        self.games_admitted = {0: games}


def _reading(t, out, inn, games, steps, wait=0.0, busy=()):
    return PT.Reading(t, out, inn, games, steps, wait, list(busy))


def test_window_line_checks_its_rows_and_parses_back():
    zero = dict.fromkeys(PT.HEADS, 0)
    a = _reading(10.0, zero, zero, 0, zero, 1.0, [(2.0, 3), (1.0, 2)])
    rows = {"during": 192, "discard": 3072, "post": 960}
    b = _reading(12.5, rows, {"during": 200, "discard": 3100, "post": 1000}, 7, rows, 1.5, [(4.0, 6), (3.5, 5)])
    line = PT.window_line(3, 6, a, b)
    end = PT.parse_run([f"[2026-10-24 10:00:00.000] {line}"])["end"]
    assert end["rows"] == {**rows, "total": 4224} and end["games"] == 7 and end["outer"] == 2.5
    assert end["ms_per_1k"] == pytest.approx(2.5 / 4224 * 1e6, rel=1e-6) and end["wait"] == pytest.approx(0.5)
    assert end["busy"] == [2.0, 2.5] and end["files"] == [3, 3] and (end["first"], end["last"]) == (4, 6)
    with pytest.raises(AssertionError, match="batcher counters"):
        PT.window_line(3, 6, a, b._replace(step_rows={**rows, "post": 959}))


def test_measure_window_times_the_update_completions():
    """The outer time runs from update W's completion (update_done) to update W+M's, whatever step_end costs; the sync
    comes before the clock; the counters are read at step_end; one memory sample per end."""
    logs, rows, now, order = [], {"n": 0}, [0.0], []
    batcher = _Batcher(dict.fromkeys(PT.HEADS, 0), dict.fromkeys(PT.HEADS, 0), 0)

    def clock():
        order.append("clock")
        return now[0]
    w = PT.MeasureWindow(logs.append, 2, 3, lambda: order.append("sync"), batcher,
                         lambda: dict.fromkeys(PT.HEADS, rows["n"]), memory=lambda: {"rss": 100 + now[0]}, clock=clock)
    for step in range(1, 6):
        now[0] = 10.0 * step  # the update completes
        w.update_done(step)
        for h in PT.HEADS:  # its commit, then its log line, 7 s later
            batcher.rows_out[h][0] += 10
            batcher.rows_in[h][0] += 12
        rows["n"] += 10
        now[0] += 7.0
        w.step_end(step)
    assert order == ["sync", "clock"] * 2  # updates 2 and 5 only
    assert w.timed("eval", 4, lambda: "x") == "x" and len(logs) == 4  # not the last step: not timed
    assert w.timed("eval", 5, lambda: "y") == "y" and logs[-1].startswith("measure-window eval step 5 | ")
    p = PT.parse_run([f"[ts] {ln}" for ln in logs[:4]])
    assert p["end"]["outer"] == 30.0 and p["end"]["ms_per_1k"] == pytest.approx(30.0 / 90 * 1e6, rel=1e-6)
    assert p["end"]["rows"] == {**dict.fromkeys(PT.HEADS, 30), "total": 90} and p["end"]["admitted"]["total"] == 108
    assert {e: (m["step"], m["mb"]) for e, m in p["memory"].items()} == {"start": (2, {"rss": 120.0}),
                                                                         "end": (5, {"rss": 150.0})}


def _parse_slow(filename):
    time.sleep(0.3)
    return filename, None


def _build_small(halfturns, tokens):
    return halfturns


def test_workers_report_exposed_wait_and_busy_time():
    with GameWorkers(2, _build_small, window=2, timeout_s=60, parse=_parse_slow, timing=True) as w:
        assert w.exposed_wait_s == 0.0
        w.fetch("a.log")  # nothing submitted ahead: the fetch waits for the worker's start and the 0.3 s parse
        first = w.exposed_wait_s
        assert first >= 0.3
        w.prefetch(["b.log"])
        deadline = time.monotonic() + 30
        while "b.log" not in w.done and time.monotonic() < deadline:
            time.sleep(0.05)
        assert "b.log" in w.done
        for p in w.procs:  # a slow liveness check: foreground work, no wait for a result
            p.is_alive = (lambda alive: lambda: (time.sleep(0.05), alive())[1])(p.is_alive)
        w.fetch("b.log")  # cached: no blocked wait at all
        assert w.exposed_wait_s == first
        busy = w.busy_seconds()
        assert sum(f for _, f in busy) == 2 and sum(s for s, _ in busy) >= 0.6


def test_workers_without_timing_have_no_clocks():
    with GameWorkers(2, _build_small, window=2, timeout_s=60, parse=_parse_slow) as w:
        w.fetch("a.log")
        assert w.timing is False and w.exposed_wait_s == 0.0 and w.busy == [] and w.busy_seconds() == []


# ---------------------------------------------------------------- gate arithmetic
def test_c1_accounting():
    assert PT.check_c1(96.0, 100.0) == pytest.approx(0.96)
    assert PT.check_c1(104.9, 100.0) == pytest.approx(1.049)
    for bad in (94.9, 105.1):
        with pytest.raises(PT.CheckError, match="C1"):
            PT.check_c1(bad, 100.0)
    with pytest.raises(PT.CheckError):
        PT.check_c1(1.0, 0.0)


def test_c2_labels_both_directions():
    assert PT.c2_label(120.0, 100.0) == (pytest.approx(1.2), "ok")
    assert PT.c2_label(151.0, 100.0)[1] == "distorted by sync"
    assert PT.c2_label(89.0, 100.0)[1] == "distorted by sync"
    assert PT.c2_label(150.0, 100.0)[1] == PT.c2_label(90.0, 100.0)[1] == "ok"


def test_d1_with_its_repeat_band():
    assert PT.d1_size([100.0], [60.0]) == (pytest.approx(0.6), "pass")
    assert PT.d1_size([100.0], [95.0]) == (pytest.approx(0.95), "fail")
    for mps in (70.0, 80.0, 90.0):
        assert PT.d1_size([100.0], [mps])[1] == "repeat"
    assert PT.d1_size([100.0, 100.0], [78.0, 80.0]) == (pytest.approx(0.79), "pass")
    assert PT.d1_size([100.0, 100.0], [85.0, 80.0]) == (pytest.approx(0.825), "fail")
    assert PT.d1_verdict({"x1": "fail", "x2": "repeat", "x5": "pass"}) == "pass"
    assert PT.d1_verdict({"x1": "fail", "x2": "repeat"}) == "repeat"
    assert PT.d1_verdict({"x1": "fail", "x2": "repeat pending"}) == "repeat"
    assert PT.d1_verdict({"x1": "fail", "x2": "fail"}) == "fail"
    with pytest.raises(PT.CheckError):
        PT.d1_size([], [1.0])
    with pytest.raises(PT.CheckError):
        PT.d1_size([1.0, 1.0, 1.0], [1.0])


def test_d1_waits_for_both_repeats():
    """An original ratio in the band needs both repeats: one device's repeat alone decides nothing, however its mean
    lands (CPU [100, 200] / MPS [85] would be 0.567 on the means)."""
    assert PT.d1_size([100.0, 200.0], [85.0]) == (pytest.approx(0.85), "repeat pending")
    assert PT.d1_size([100.0], [85.0, 40.0]) == (pytest.approx(0.85), "repeat pending")
    assert PT.d1_size([100.0, 200.0], [85.0, 85.0]) == (pytest.approx(85.0 / 150.0), "pass")
    assert PT.d1_size([100.0, 100.0], [60.0, 95.0]) == (pytest.approx(0.6), "pass")  # out of band: the originals


def test_epoch_projection():
    # 1,000 admitted rows per game, 225,176 games; 2,000 rows a step; 100 ms per 1k positions; eval 30 s + save 2 s
    p = PT.epoch_projection(100.0, 50_000, 50, 2_000, 30.0, 2.0)
    assert p["rows_per_epoch"] == pytest.approx(225_176_000) and p["steps_per_epoch"] == pytest.approx(112_588)
    assert p["eval_save_events"] == 225
    assert p["train_h"] == pytest.approx(100 * 225_176_000 / 1e6 / 3600)
    assert p["eval_save_h"] == pytest.approx(225 * 32 / 3600)
    assert p["total_h"] == pytest.approx(p["train_h"] + p["eval_save_h"])


def test_window_hashes_need_every_window_step():
    p = {"end": {"first": 2, "last": 3}, "train": {2: ("t", "a" * 12, "b" * 12), 3: ("t", "c" * 12, "d" * 12)}}
    read, sample = PT.window_hashes(p)
    assert len(read) == len(sample) == 16 and read != sample
    q = {"end": p["end"], "train": {**p["train"], 3: ("t", "e" * 12, "d" * 12)}}
    assert PT.window_hashes(q)[0] != read and PT.window_hashes(q)[1] == sample
    with pytest.raises(PT.CheckError, match="without a step line"):
        PT.window_hashes({"end": p["end"], "train": {2: p["train"][2]}})
