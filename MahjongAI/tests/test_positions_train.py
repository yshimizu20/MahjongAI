"""T-055 G1, trainer level, positions mode on the CPU (fixture games; steps that cross an epoch).

Exact resume: a kill after an eval row and before the next save, and a stop inside an eval row's write, resume to the
unbroken run: metrics rows (with the batcher counters), per-step log lines (losses, read hash, sample digest) and
every checkpoint field (batcher buffers and provenance included).
Failure boundaries: a stop inside a fill (an interrupt after a cluster draw) and a failure after the forward and
before the commit (a forced clip_grad_norm_ error) each save a checkpoint from which the resume ends equal to the
unbroken run; the second one's stop checkpoint equals the unbroken run's checkpoint of that step except the fill
(RNG from before the failed forward; the buffers whole, with the rows the failed step's fill drew). An interrupt
after a commit and before the update's RNG snapshot and train sums are final saves no checkpoint; a failure after an
eval row is on disk resumes with that row's interval closed. Both resume to the unbroken run.
Stop bound (T-056): --stop-after-step stops after update K with a checkpoint of step K equal to the unbroken run's,
on the same LR horizon, and no worker left; the resume ends equal to the unbroken run.
Counters: every eval row's per-source-epoch counters (games, rows in / out per head) equal an independent reference
over the trainer's loader order (prepare_batches per game, the plain-Python emit rule).
Identity: batch mode, sizes and device are run-identity keys; a config without them reads as games mode on the CPU."""
import inspect
import multiprocessing
import os
import re
import shutil
import sys

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import TransformerTensorProcessor
from MahjongAI.tests.test_position_batcher import epoch_counter_problems, reference_steps
from MahjongAI.tests.test_resume_rows import _assert_equal

STEPS, EVAL_EVERY, SAVE_EVERY = 12, 5, 4  # rows 0, 5, 10, 12; checkpoints 4, 8, 12
SIZES = {"during": 60, "discard": 1600, "post": 400}  # ~3-4 fixture games a step: 12 steps cross epoch 0


def _run(tmp_path, monkeypatch, root, name, steps=STEPS, save_every=SAVE_EVERY, sizes=SIZES, device="cpu",
         batch_mode="positions", stop_after_step=None, **arch):
    """arch: embd_size / n_layers / row0_tol of SL.train (the defaults when absent)."""
    monkeypatch.setattr(SL, "DATA_ROOT", root)
    monkeypatch.setattr(SL, "VAL_GAMES", 2)
    monkeypatch.setattr(SL, "PRIOR_GAMES", 2)
    threads = torch.get_num_threads()
    try:
        SL.train("encoder_conv", steps, str(tmp_path / name), SL.RunLog(str(tmp_path / f"{name}.log")),
                 eval_every=EVAL_EVERY, save_every=save_every, batch_mode=batch_mode,
                 position_sizes=sizes if batch_mode == "positions" else None, device=device,
                 stop_after_step=stop_after_step, **arch)
    finally:
        torch.set_num_threads(threads)
    return tmp_path / name


def step_lines(path):
    """{step: the per-step log line without its timestamp}, the last one of each step (a resume re-logs it)."""
    out = {}
    for line in open(path):
        m = re.match(r"\[[^]]+\] (step (\d+) \| (?!val|positions).*)$", line.rstrip("\n"))
        if m:
            out[int(m.group(2))] = m.group(1)
    return out


def assert_runs_equal(tmp_path, a, b, checkpoints, ckpt_b=None):
    ha, rows_a = SL.Metrics(str(tmp_path / a / "metrics.jsonl")).read()
    hb, rows_b = SL.Metrics(str(tmp_path / b / "metrics.jsonl")).read()
    assert [r["step"] for r in rows_a] == [r["step"] for r in rows_b] == [0, 5, 10, 12]
    _assert_equal({k: v for k, v in ha.items() if k != "started"}, {k: v for k, v in hb.items() if k != "started"},
                  "header")
    _assert_equal(rows_a, rows_b, "rows")
    lines_a, lines_b = step_lines(tmp_path / f"{a}.log"), step_lines(tmp_path / f"{b}.log")
    assert sorted(lines_a) == list(range(1, STEPS + 1)) and lines_a == lines_b
    for step in checkpoints:
        _assert_equal(torch.load(tmp_path / a / "checkpoints" / f"step_{step}.pt", weights_only=True),
                      torch.load(tmp_path / (ckpt_b or b) / "checkpoints" / f"step_{step}.pt", weights_only=True),
                      f"step_{step}.pt")


def reference_epoch_counts(monkeypatch, root, n_steps):
    """(steps, tally) of reference_steps over the trainer's loader order (a fresh serial cluster), each game's rows
    counted per head by TransformerTensorProcessor.prepare_batches: a row id is (file, source epoch, row)."""
    monkeypatch.setattr(SL, "DATA_ROOT", root)
    cluster, processor = SL.train_cluster(lambda *_: None, 1), TransformerTensorProcessor()

    def stream():
        while True:
            games, tokens = next(cluster)
            if not games:
                continue
            file, epoch = cluster.last_batch_files[0], cluster.dataloaders[cluster.idx].epoch
            built = processor.prepare_batches(games, tokens)
            yield file, epoch, {h: [(file, epoch, i) for i in range(len(t[3]))] for h, t in zip(SL.HEADS, built)}

    tally = []
    return reference_steps(stream(), SIZES, n_steps, tally=tally), tally


@pytest.fixture
def unbroken(tmp_path, monkeypatch, train_data_root):
    """The unbroken positions-mode run "a" (save every 2 steps: a checkpoint for every even step)."""
    _run(tmp_path, monkeypatch, train_data_root, "a", save_every=2)
    return tmp_path / "a"


def test_unbroken_run_records_positions(tmp_path, monkeypatch, train_data_root, unbroken):
    header, rows = SL.Metrics(str(unbroken / "metrics.jsonl")).read()
    config = header["config"]
    assert (config["batch_mode"], config["position_sizes"], config["device"], config["batch_size"]) == (
        "positions", SIZES, "cpu", 1)
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", header["git_commit"])
    last = rows[-1]["positions"]
    assert last["games_admitted"] > 40 and set(last["games_by_epoch"]) >= {"0", "1"}  # crossed an epoch
    assert last["taken"]["discard"] == {str(SIZES["discard"]): STEPS}
    # per source epoch, every row's counters equal an independent reference over the same loader order
    steps, tally = reference_epoch_counts(monkeypatch, train_data_root, STEPS)
    for row in rows[1:]:
        s = row["step"]
        problems = epoch_counter_problems(row["positions"], steps[:s], tally[s - 1])
        assert not problems, f"row {s}: {problems}"
    zeroed = {**last, "rows_out": {h: {"0": sum(last["rows_out"][h].values())} for h in SL.HEADS}}
    assert epoch_counter_problems(zeroed, steps, tally[-1]), "crediting every row to epoch 0 must fail"
    for h in SL.HEADS:
        assert rows[-1]["heads"][h]["train_n"] == sum(int(k) * n for k, n in last["taken"][h].items()) - sum(
            int(k) * n for k, n in rows[-2]["positions"]["taken"][h].items())
    lines = step_lines(tmp_path / "a.log")
    assert all(re.search(r" \| read [0-9a-f]{12} \| sample [0-9a-f]{12}$", ln) for ln in lines.values())
    ckpt = torch.load(unbroken / "checkpoints" / "step_12.pt", weights_only=True)
    assert ckpt["batcher_state"]["sizes"] == SIZES and "mps_rng_state" not in ckpt


def test_kill_after_an_eval_row_before_the_save_resumes_to_the_unbroken_run(tmp_path, monkeypatch, train_data_root,
                                                                             unbroken):
    b = _run(tmp_path, monkeypatch, train_data_root, "b")
    os.remove(b / "checkpoints" / "step_12.pt")
    header, rows = SL.Metrics(str(b / "metrics.jsonl")).read()
    SL.Metrics(str(b / "metrics.jsonl"))._write([header] + [r for r in rows if r["step"] <= 10])
    resumed_from = torch.load(b / "checkpoints" / "step_8.pt", weights_only=True)
    assert resumed_from["batcher_state"]["buffers"]["discard"]["cols"] is not None  # carried rows in the state
    _run(tmp_path, monkeypatch, train_data_root, "b")
    assert "resuming at step 8;" in open(tmp_path / "b.log").read()
    assert_runs_equal(tmp_path, "a", "b", (4, 8, 12))


def test_stop_inside_an_eval_row_write_resumes_to_the_unbroken_run(tmp_path, monkeypatch, train_data_root, unbroken):
    append = SL.Metrics.append

    def stop_at_row_10(self, row):
        if row.get("type") == "eval" and row["step"] == 10:
            raise RuntimeError("stopped inside the row-10 write")
        append(self, row)

    with monkeypatch.context() as m:
        m.setattr(SL.Metrics, "append", stop_at_row_10)
        with pytest.raises(RuntimeError, match="stopped inside the row-10 write"):
            _run(tmp_path, monkeypatch, train_data_root, "b")
    _run(tmp_path, monkeypatch, train_data_root, "b")
    assert "resuming at step 10;" in open(tmp_path / "b.log").read()
    assert_runs_equal(tmp_path, "a", "b", (4, 8, 12))


def test_stop_inside_a_fill_resumes_to_the_unbroken_run(tmp_path, monkeypatch, train_data_root, unbroken):
    """An interrupt right after the 9th cluster draw (inside step 3's fill, its game not admitted yet): the draw is
    undone, the stop checkpoint keeps the admitted games, the resume ends equal to the unbroken run."""
    make = SL.train_cluster

    def stopping_cluster(*a, **k):
        cluster = make(*a, **k)
        cls, calls = type(cluster), [0]

        def next_then_stop(self):
            out = cls.__next__(self)
            calls[0] += 1
            if calls[0] == 9:
                raise KeyboardInterrupt("stop after draw 9")
            return out

        cluster.__class__ = type("Stopping", (cls,), {"__next__": next_then_stop})
        return cluster

    with monkeypatch.context() as m:
        m.setattr(SL, "train_cluster", stopping_cluster)
        with pytest.raises(KeyboardInterrupt):
            _run(tmp_path, monkeypatch, train_data_root, "b")
    stop = re.search(r"STOP at step (\d+): KeyboardInterrupt", open(tmp_path / "b.log").read())
    assert stop and 0 < int(stop.group(1)) < 6, stop
    _run(tmp_path, monkeypatch, train_data_root, "b")
    assert_runs_equal(tmp_path, "a", "b", (8, 12))


def test_failure_after_the_forward_before_the_commit_keeps_the_step(tmp_path, monkeypatch, train_data_root, unbroken):
    """Update 7 fails in clip_grad_norm_ (after its forward and backward). The stop checkpoint is step_6.pt: model,
    optimizer, scheduler, RNG (from before the failed forward), train sums and the committed counters equal the
    unbroken run's step_6.pt; the buffers stay whole [R2] (the unbroken step_6 rows, then the rows of the games step
    7's fill drew, in order). The resume ends equal to the unbroken run."""
    clip, calls = torch.nn.utils.clip_grad_norm_, [0]

    def fail_once(*a, **k):
        calls[0] += 1
        if calls[0] == 7:
            raise RuntimeError("forced failure in update 7")
        return clip(*a, **k)

    with monkeypatch.context() as m:
        m.setattr(torch.nn.utils, "clip_grad_norm_", fail_once)
        with pytest.raises(RuntimeError, match="forced failure in update 7"):
            _run(tmp_path, monkeypatch, train_data_root, "b")
    b = tmp_path / "b" / "checkpoints"
    assert sorted(os.listdir(b)) == ["step_4.pt", "step_6.pt"]
    ref = torch.load(unbroken / "checkpoints" / "step_6.pt", weights_only=True)
    stop = torch.load(b / "step_6.pt", weights_only=True)
    filled = ("loader_state", "batcher_state")
    _assert_equal({k: v for k, v in ref.items() if k not in filled}, {k: v for k, v in stop.items() if k not in filled},
                  "stop checkpoint step_6.pt")
    rb, sb = ref["batcher_state"], stop["batcher_state"]
    for key in ("sizes", "rows_out", "taken"):
        _assert_equal(rb[key], sb[key], f"batcher_state.{key}")
    assert sb["games_admitted"] != rb["games_admitted"] and sb["read_since_update"]  # step 7's fill admitted games
    for h in SL.HEADS:
        n, extra = len(rb["buffers"][h]["provenance"]), sum(sb["rows_in"][h].values()) - sum(rb["rows_in"][h].values())
        assert len(sb["buffers"][h]["provenance"]) == n + extra, h
        _assert_equal(rb["buffers"][h]["provenance"], sb["buffers"][h]["provenance"][:n], f"{h} provenance prefix")
        if n:
            _assert_equal(rb["buffers"][h]["cols"], [c[:n] for c in sb["buffers"][h]["cols"]], f"{h} rows prefix")
    _run(tmp_path, monkeypatch, train_data_root, "b")
    assert_runs_equal(tmp_path, "a", "b", (8, 12))


def test_stop_after_step_saves_step_k_and_resumes_to_the_unbroken_run(tmp_path, monkeypatch, train_data_root,
                                                                       unbroken):
    """--stop-after-step 6 (not an eval or save step of run b): train returns normally, step_6.pt equals the
    unbroken run's step_6.pt in every field (scheduler and LR on the 12-step horizon, train sums of the open
    interval, batcher state, RNG), no worker process is left; a start without the bound resumes at step 6 and ends
    equal to the unbroken run. A second start with the same bound stops at once and saves nothing new."""
    b = _run(tmp_path, monkeypatch, train_data_root, "b", stop_after_step=6)
    assert not multiprocessing.active_children(), multiprocessing.active_children()
    log = open(tmp_path / "b.log").read()
    assert "stopped after update 6 (--stop-after-step 6, total_steps 12)" in log and "Training completed." not in log
    assert sorted(os.listdir(b / "checkpoints")) == ["step_4.pt", "step_6.pt"]
    assert [r["step"] for r in SL.Metrics(str(b / "metrics.jsonl")).read()[1]] == [0, 5]
    _assert_equal(torch.load(unbroken / "checkpoints" / "step_6.pt", weights_only=True),
                  torch.load(b / "checkpoints" / "step_6.pt", weights_only=True), "stop checkpoint step_6.pt")
    mtime = os.path.getmtime(b / "checkpoints" / "step_6.pt")
    _run(tmp_path, monkeypatch, train_data_root, "b", stop_after_step=6)
    assert os.path.getmtime(b / "checkpoints" / "step_6.pt") == mtime
    assert len(step_lines(tmp_path / "b.log")) == 6  # no update after the bound
    _run(tmp_path, monkeypatch, train_data_root, "b")
    assert "resuming at step 6;" in open(tmp_path / "b.log").read()
    assert "Training completed." in open(tmp_path / "b.log").read()
    assert_runs_equal(tmp_path, "a", "b", (4, 8, 12))


class InterruptAtLine:
    """A KeyboardInterrupt (a signal's stand-in) raised by a trace function just before the first execution of the
    source line `at` of SL._train after the line `after` ran n times: a window between two plain statements, which
    no wrapped call reaches."""

    def __init__(self, after: str, n: int, at: str):
        lines, start = inspect.getsourcelines(SL._train)

        def line_of(text):
            found = [start + i for i, ln in enumerate(lines) if ln.strip() == text]
            assert len(found) == 1, (text, found)
            return found[0]

        self.after, self.n, self.at, self.seen, self.fired = line_of(after), n, line_of(at), 0, False

    def __enter__(self):
        sys.settrace(self._call)
        return self

    def __exit__(self, *exc):
        sys.settrace(None)

    def _call(self, frame, event, arg):
        return self._line if frame.f_code is SL._train.__code__ else None

    def _line(self, frame, event, arg):
        if event == "line" and not self.fired:
            if frame.f_lineno == self.after:
                self.seen += 1
            elif frame.f_lineno == self.at and self.seen == self.n:
                self.fired = True
                raise KeyboardInterrupt(f"interrupt before line {self.at}")
        return self._line


@pytest.mark.parametrize("at", ["step_rng = None", "train_sums[h][1] += out.y.numel()"],
                         ids=["before_the_rng_snapshot_is_cleared", "inside_the_train_sums_update"])
def test_an_interrupt_after_the_commit_never_saves_a_half_finished_update(tmp_path, monkeypatch, train_data_root,
                                                                          unbroken, at):
    """An interrupt after update 7's commit, before the step's RNG snapshot is cleared or inside its train-sums update
    (Codex round 2): no stop checkpoint holds the previous step's RNG or half-added train sums. The resume (from
    step_4.pt) ends equal to the unbroken run."""
    with InterruptAtLine("current_step += 1", 7, at) as trace:
        with pytest.raises(KeyboardInterrupt, match="interrupt before line"):
            _run(tmp_path, monkeypatch, train_data_root, "b")
    assert trace.fired
    assert "no checkpoint: the stop came inside an optimizer update" in open(tmp_path / "b.log").read()
    assert sorted(os.listdir(tmp_path / "b" / "checkpoints")) == ["step_4.pt"]
    _run(tmp_path, monkeypatch, train_data_root, "b")
    assert "resuming at step 4;" in open(tmp_path / "b.log").read()
    assert_runs_equal(tmp_path, "a", "b", (8, 12))


@pytest.mark.parametrize("where", ["after_the_row_append", "inside_the_peak_reset"])
def test_a_failure_after_an_eval_row_is_on_disk_resumes_to_the_unbroken_run(tmp_path, monkeypatch, train_data_root,
                                                                            unbroken, where):
    """Row 10 is on disk, then a RuntimeError: right after Metrics.append returns (before the interval resets), or in
    the buffer-peak reset (train sums already reset) (Codex round 2). The stop checkpoint step_10.pt holds the closed
    interval's accounting; the resume keeps row 10, closes that interval (train sums, buffer peaks) and ends equal to
    the unbroken run."""
    if where == "after_the_row_append":
        append = SL.Metrics.append

        def fail_after_row_10(self, row):
            append(self, row)
            if row.get("type") == "eval" and row["step"] == 10:
                raise RuntimeError("failure after the row-10 append")

        patch = (SL.Metrics, "append", fail_after_row_10)
    else:
        reset, calls = SL.PositionBatcher.reset_max_buffered, [0]

        def fail_at_row_10(self):
            calls[0] += 1
            if calls[0] == 3:  # rows 0, 5, 10
                raise RuntimeError("failure after the row-10 append")
            reset(self)

        patch = (SL.PositionBatcher, "reset_max_buffered", fail_at_row_10)
    with monkeypatch.context() as m:
        m.setattr(*patch)
        with pytest.raises(RuntimeError, match="failure after the row-10 append"):
            _run(tmp_path, monkeypatch, train_data_root, "b")
    assert [r["step"] for r in SL.Metrics(str(tmp_path / "b" / "metrics.jsonl")).read()[1]] == [0, 5, 10]
    assert os.path.exists(tmp_path / "b" / "checkpoints" / "step_10.pt")
    _run(tmp_path, monkeypatch, train_data_root, "b")
    assert "resuming at step 10;" in open(tmp_path / "b.log").read()
    assert_runs_equal(tmp_path, "a", "b", (4, 8, 12))


# ---------------------------------------------------------------- run identity
def test_config_mismatch_reads_a_config_without_t055_keys_as_games_on_the_cpu():
    legacy = {"model": "encoder_conv", "total_steps": 100}
    games = dict(legacy, batch_mode="games", position_sizes=None, device="cpu")
    assert SL.config_mismatch(legacy, games) == {} and SL.config_mismatch(games, legacy) == {}
    assert SL.config_mismatch(legacy, dict(games, batch_mode="positions", position_sizes=SIZES)) == {
        "batch_mode": ("games", "positions"), "position_sizes": (None, SIZES)}
    assert SL.config_mismatch(games, dict(games, device="mps")) == {"device": ("cpu", "mps")}


def _strip_t055(run_dir, step):
    path = run_dir / "checkpoints" / f"step_{step}.pt"
    ckpt = torch.load(path, weights_only=True)
    ckpt["config"] = {k: v for k, v in ckpt["config"].items() if k not in SL.LEGACY_RUN}
    torch.save(ckpt, path)
    metrics = SL.Metrics(str(run_dir / "metrics.jsonl"))
    header, rows = metrics.read()
    header["config"] = {k: v for k, v in header["config"].items() if k not in SL.LEGACY_RUN}
    metrics._write([header] + rows)


def test_a_legacy_games_run_resumes_and_other_settings_are_rejected(tmp_path, monkeypatch, train_data_root):
    run = _run(tmp_path, monkeypatch, train_data_root, "g", steps=1, batch_mode="games")
    _strip_t055(run, 1)  # a run dir written before T-055
    with pytest.raises(AssertionError, match=r"step_1.pt is from another run config: batch_mode 'games' -> "
                                             r"'positions', batch_size 2 -> 1, position_sizes None -> "):
        _run(tmp_path, monkeypatch, train_data_root, "g", steps=2)
    _run(tmp_path, monkeypatch, train_data_root, "g", steps=2, batch_mode="games")
    assert "resuming at step 1;" in open(tmp_path / "g.log").read()


@pytest.mark.mps
def test_a_legacy_games_run_rejects_the_mps_device(tmp_path, monkeypatch, train_data_root):
    """The device case of the legacy resume: it needs MPS (the trainer moves the model there before the config
    check), so it carries the mps marker and Linux CI leaves it out."""
    run = _run(tmp_path, monkeypatch, train_data_root, "g", steps=1, batch_mode="games")
    _strip_t055(run, 1)  # a run dir written before T-055
    with pytest.raises(AssertionError, match=r"step_1.pt is from another run config: device 'cpu' -> 'mps'"):
        _run(tmp_path, monkeypatch, train_data_root, "g", steps=2, batch_mode="games", device="mps")


def test_a_positions_run_rejects_other_sizes(tmp_path, monkeypatch, train_data_root):
    _run(tmp_path, monkeypatch, train_data_root, "p", steps=1)
    with pytest.raises(AssertionError, match=r"another run config: position_sizes .*'post': 400.* -> .*'post': 300"):
        _run(tmp_path, monkeypatch, train_data_root, "p", steps=2, sizes=dict(SIZES, post=300))
    with pytest.raises(AssertionError, match="position sizes are a positions-mode setting"):
        SL.train("encoder_conv", 1, str(tmp_path / "x"), SL.RunLog(str(tmp_path / "x.log")), position_sizes=SIZES)
