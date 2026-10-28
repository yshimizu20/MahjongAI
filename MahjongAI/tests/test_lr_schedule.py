"""T-067: the LR schedule keyed to games read (games_cosine_v1), its run identity and the migration of a steps_cosine
run (CPU, fixture games).

T1 lr_at at its boundaries (warmup, quarter points of the horizon, its end, past it) to 1e-12, and the corpus guard.
T2 in the trainer, both batch modes: the LR of every optimizer step (an optimizer step pre-hook) equals lr_at of a
   games count kept by the test (the games the cluster returned), across the warmup, the cosine, the horizon end, the
   floor and an epoch boundary; games mode with forced no-sample batches. Step log lines and eval rows carry the same
   values.
T3 a resume with another total_steps (12 -> 24) gives the step lines (LR, games, losses, read and sample hashes) and
   the final checkpoint of the unbroken 24-update run.
T5 migrate_lr_schedule: a steps_cosine run switched in a new run dir resumes without any flag, the LR of every new
   update = lr_at of its logged games; a stop inside the copy or between the checkpoint and header writes is refused
   by the trainer and completed by a re-run; an unrelated mismatch still fails; a games-mode state without its games
   counter and a run already on games_cosine_v1 are refused; the source is never written (read_only_run). The
   migration never removes a folder it did not stage (a source at the staging path, an unrelated folder there, another
   migration's staging folder), refuses a new run dir migrated from another source (or from a source checkpoint
   changed since), and completes an interrupted migration from its snapshot after the source changed.
T6 a corpus other than the recorded one is refused without --lr-cycle-games; with its sha recorded, the default runs.
Also: a steps_cosine run dir (no lr_schedule) resumes on its old formula.
Negative controls (each must fail the test's check): the LR read from total_steps (T2, T3), g counted from the resume
(T4 in test_positions_train, T5).
The test runs use small constants (test_positions_train: LR_WARMUP 3, TEST_LR_CYCLE_GAMES 16: horizon 32 games)."""
import hashlib
import math
import os
import re
import shutil

import pytest
import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

from MahjongAI import migrate_lr_schedule as MIG
from MahjongAI import supervised_learning as SL
from MahjongAI.tests.conftest import TEST_LR_CYCLE_GAMES, read_only_run, run_log, tree_digest
from MahjongAI.tests.test_positions_train import LR_WARMUP, _run, step_lines
from MahjongAI.tests.test_resume_rows import _assert_equal

TOL = 1e-12
LR_AT = SL.lr_at  # the shipped function: the reference of every check, whatever a negative control patches
LINE_LR = re.compile(r" \| lr (\S+) games (\d+) \| ")
TEST_SCHEDULE = {"name": SL.LR_SCHEDULE, "cycle_games": TEST_LR_CYCLE_GAMES, "cycles": SL.LR_CYCLES,
                 "peak": SL.LR_PEAK, "floor": SL.LR_FLOOR, "warmup": LR_WARMUP}
HORIZON = TEST_LR_CYCLE_GAMES * SL.LR_CYCLES


# ---------------------------------------------------------------- T1: lr_at and the corpus guard
def test_lr_at_at_its_boundaries():
    s = SL.games_cosine_schedule(SL.LR_CYCLE_CORPUS, SL.LR_CYCLE_SKIPS)
    assert s == {"name": "games_cosine_v1", "cycle_games": 225_176, "cycles": 2, "peak": 1e-3, "floor": 1e-4,
                 "warmup": 1000}
    h = 450_352
    assert h % 4 == 0
    r = math.sqrt(2) / 2
    expect = {  # (g, k): the closed form, no cosine call
        (0, 0): 0.0, (0, 1): 1e-6, (10**6, 999): 0.999e-3,  # warmup ignores g
        (0, 1000): 1e-3, (h // 4, 1000): 1e-4 + 9e-4 * 0.5 * (1 + r), (h // 2, 5000): 5.5e-4,
        (3 * h // 4, 5000): 1e-4 + 9e-4 * 0.5 * (1 - r), (h, 5000): 1e-4, (h + 1, 5000): 1e-4, (3 * h, 10**6): 1e-4,
        (226_096, 94_000): 5.471120136464902e-4,  # the T-061 step_94000.pt coordinate (Codex round 1)
    }
    got = {gk: SL.lr_at(*gk, s) for gk in expect}
    bad = {gk: (got[gk], v) for gk, v in expect.items() if abs(got[gk] - v) > TOL}
    assert not bad, bad
    quarters = [(0, 1000), (h // 4, 1000), (h // 2, 5000), (3 * h // 4, 5000), (h, 5000)]
    assert [round(got[gk], 6) for gk in quarters] == [1e-3, 8.68e-4, 5.5e-4, 2.32e-4, 1e-4]  # the plan's table
    series = [SL.lr_at(g, 1000, s) for g in range(0, h + 5000, 997)]  # past warmup: non-increasing in g
    assert all(a >= b for a, b in zip(series, series[1:])) and series[-1] == 1e-4


def test_the_default_cycle_needs_the_recorded_corpus():
    assert SL.LR_CYCLE_SKIPS == sorted(SL.EXPECTED_SKIPS)  # the acceptance policy LR_CYCLE_GAMES was counted under
    other = "0" * 64
    with pytest.raises(AssertionError, match=rf"train_files_sha256 {other} .* give --lr-cycle-games N"):
        SL.games_cosine_schedule(other, SL.EXPECTED_SKIPS)
    with pytest.raises(AssertionError, match="is not the corpus LR_CYCLE_GAMES 225176 was counted on"):
        SL.games_cosine_schedule(SL.LR_CYCLE_CORPUS, sorted(SL.EXPECTED_SKIPS)[1:])
    for sha in (SL.LR_CYCLE_CORPUS, other):  # an explicit count: any corpus, saved in the schedule
        assert SL.games_cosine_schedule(sha, [], 1234)["cycle_games"] == 1234
    with pytest.raises(AssertionError, match="must be >= 1"):
        SL.games_cosine_schedule(other, [], 0)


def test_a_config_without_lr_schedule_reads_as_steps_cosine():
    legacy = {"model": "encoder_conv", "total_steps": 100}
    assert SL.config_mismatch(legacy, dict(legacy, lr_schedule={"name": "steps_cosine"})) == {}
    assert SL.config_mismatch(legacy, dict(legacy, lr_schedule=TEST_SCHEDULE)) == {
        "lr_schedule": ({"name": "steps_cosine"}, TEST_SCHEDULE)}
    assert SL.config_mismatch(dict(legacy, lr_schedule=TEST_SCHEDULE),
                              dict(legacy, lr_schedule=dict(TEST_SCHEDULE, cycle_games=17))) != {}


# ---------------------------------------------------------------- T6: the corpus guard in the trainer
def test_an_unknown_corpus_is_refused_without_lr_cycle_games(tmp_path, monkeypatch, train_data_root):
    with pytest.raises(AssertionError, match=r"is not the corpus LR_CYCLE_GAMES 225176 was counted on .* give "
                                             r"--lr-cycle-games N"):
        _run(tmp_path, monkeypatch, train_data_root, "x", steps=1, lr_cycle_games=None)
    assert not os.path.exists(tmp_path / "x" / "metrics.jsonl")  # refused before any startup work
    assert "Validation:" not in open(tmp_path / "x.log").read()
    # the same corpus recorded as LR_CYCLE_CORPUS: the default count, saved in the run identity
    sha = SL.train_files_sha256(SL.train_cluster(lambda *_: None, 1))
    monkeypatch.setattr(SL, "LR_CYCLE_CORPUS", sha)
    run = _run(tmp_path, monkeypatch, train_data_root, "y", steps=1, lr_cycle_games=None)
    header, _ = SL.Metrics(str(run / "metrics.jsonl")).read()
    assert header["config"]["lr_schedule"] == dict(TEST_SCHEDULE, cycle_games=SL.LR_CYCLE_GAMES)
    # a resume with another count is another run
    with pytest.raises(AssertionError, match=r"step_1.pt is from another run config: lr_schedule .*'cycle_games': "
                                             r"225176.* -> .*'cycle_games': 16"):
        _run(tmp_path, monkeypatch, train_data_root, "y", steps=2)


# ---------------------------------------------------------------- the trainer's LR, observed
class Observed:
    """While active: every optimizer step's LRs (a global optimizer step pre-hook) with the games the training cluster
    has returned so far, counted here from the batches (not from any trainer or batcher counter)."""

    def __init__(self, monkeypatch):
        self.games, self.steps = 0, []  # steps: [(lrs of the param groups, games)]
        make = SL.train_cluster

        def counting_cluster(*a, **k):
            cluster = make(*a, **k)
            cls = type(cluster)

            def next_counted(c):
                out = cls.__next__(c)
                self.games += len(out[0])
                return out

            cluster.__class__ = type("Counting", (cls,), {"__next__": next_counted})
            return cluster

        monkeypatch.setattr(SL, "train_cluster", counting_cluster)

    def __enter__(self):
        def pre_step(optimizer, args, kwargs):
            if isinstance(optimizer, torch.optim.AdamW):  # the trainer's (the prior's fit has its own)
                self.steps.append(([g["lr"] for g in optimizer.param_groups], self.games))
        self.handle = register_optimizer_step_pre_hook(pre_step)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def lr_from_total_steps(total_steps):
    """Negative control: the LR's progress read from the updates over total_steps (the steps_cosine idea), not g."""
    return lambda g, k, s: LR_AT(k * s["cycles"] * s["cycle_games"] / total_steps, k, s)


def lr_problems(observed, schedule, first_k=0) -> dict:
    """{k: (lrs, games, lr_at)} of the observed steps (update k + 1, k = first_k + i) whose LR is not lr_at(games,
    k)."""
    out = {}
    for i, (lrs, g) in enumerate(observed):
        k = first_k + i
        if max(abs(lr - LR_AT(g, k, schedule)) for lr in lrs) > TOL:
            out[k] = (lrs, g, LR_AT(g, k, schedule))
    return out


def logged_lr(log_path) -> dict:
    """{step: (lr, games)} of the step log lines."""
    out = {}
    for s, line in step_lines(log_path).items():
        m = LINE_LR.search(line)
        assert m, line
        out[s] = (float(m.group(1)), int(m.group(2)))
    return out


def assert_lr_follows_games(observed, log_path, run_dir, schedule, first_k=0):
    """The T2 check: every observed LR = lr_at(the test's games count, k); the log lines and eval rows carry the same
    (LR, games); the steps cross every part of the schedule."""
    problems = lr_problems(observed, schedule, first_k)
    assert not problems, f"LR != lr_at(games, k): {problems}"
    logged = logged_lr(log_path)
    assert {first_k + i + 1: (lrs[0], g) for i, (lrs, g) in enumerate(observed)} == {
        s: v for s, v in logged.items() if s > first_k}
    _, rows = SL.Metrics(str(run_dir / "metrics.jsonl")).read()
    for row in rows:
        if row["step"] > first_k:
            assert (row["lr"], row["games_read"]) == logged[row["step"]], row["step"]
    ks = [first_k + i for i in range(len(observed))]
    games = [g for _, g in observed]
    if first_k == 0:
        assert rows[0]["step"] == 0 and (rows[0]["lr"], rows[0]["games_read"]) == (None, 0)
        assert min(ks) < schedule["warmup"] <= max(ks), "the run never left the warmup"
    lrs = [lrs[0] for lrs, _ in observed]
    assert any(k >= schedule["warmup"] and schedule["floor"] < lr < schedule["peak"] for k, lr in zip(ks, lrs)), (
        "no update inside the cosine")
    assert max(games) >= HORIZON and lrs[-1] == schedule["floor"], "the run never reached the horizon end"


def _epoch_of(run_dir, step) -> int:
    return torch.load(run_dir / "checkpoints" / f"step_{step}.pt", weights_only=True)["loader_state"]["epoch"]


@pytest.mark.parametrize("bug", [None, "lr_from_total_steps"])
def test_positions_mode_lr_is_lr_at_of_the_games_read(tmp_path, monkeypatch, module_data_root, bug):
    with Observed(monkeypatch) as obs:
        if bug:
            monkeypatch.setattr(SL, "lr_at", lr_from_total_steps(12))
        run = _run(tmp_path, monkeypatch, module_data_root, "p")
    assert len(obs.steps) == 12 and _epoch_of(run, 12) >= 1  # crossed an epoch
    if bug:
        with pytest.raises(AssertionError, match=r"LR != lr_at\(games, k\)"):
            assert_lr_follows_games(obs.steps, tmp_path / "p.log", run, TEST_SCHEDULE)
    else:
        assert_lr_follows_games(obs.steps, tmp_path / "p.log", run, TEST_SCHEDULE)


@pytest.mark.parametrize("bug", [None, "lr_from_total_steps"])
def test_games_mode_lr_is_lr_at_of_the_games_read_with_no_sample_batches(tmp_path, monkeypatch, module_data_root,
                                                                         bug):
    """Batches 2, 5 and 9 give no training sample (forced): their games count, no update runs for them."""
    forward, batches = SL.EncoderConvModel.forward, [0]

    def some_without_samples(self, *a, **k):
        out = forward(self, *a, **k)
        if k.get("sample"):  # a training batch (validation calls sample=False)
            batches[0] += 1
            if batches[0] in (2, 5, 9):
                return {}
        return out

    monkeypatch.setattr(SL.EncoderConvModel, "forward", some_without_samples)
    steps = 22
    with Observed(monkeypatch) as obs:
        if bug:
            monkeypatch.setattr(SL, "lr_at", lr_from_total_steps(steps))
        run = _run(tmp_path, monkeypatch, module_data_root, "g", steps=steps, save_every=steps, batch_mode="games")
    assert len(obs.steps) == steps and batches[0] == steps + 3 and _epoch_of(run, steps) >= 1
    assert obs.steps[1][1] - obs.steps[0][1] == 4  # update 2 ran after a no-sample batch: its games count
    if bug:
        with pytest.raises(AssertionError, match=r"LR != lr_at\(games, k\)"):
            assert_lr_follows_games(obs.steps, tmp_path / "g.log", run, TEST_SCHEDULE)
    else:
        assert_lr_follows_games(obs.steps, tmp_path / "g.log", run, TEST_SCHEDULE)


# ---------------------------------------------------------------- T3: total_steps changed on a resume
def assert_same_24_steps(a, b):
    """Step lines 1-24, the step-24 checkpoint and rows 0, 5, 10, 20, 24 (without total_steps) equal. b's 12-step leg
    adds row 12, which closes an interval: b's row 15 counts its train sums and buffer peaks from there."""
    lines_a, lines_b = step_lines(run_log(a)), step_lines(run_log(b))
    assert sorted(lines_a) == list(range(1, 25)) and lines_a == lines_b, "step lines differ"

    def rows(run):
        return [{k: v for k, v in r.items() if k != "total_steps"}
                for r in SL.Metrics(str(run / "metrics.jsonl")).read()[1] if r["step"] in (0, 5, 10, 20, 24)]
    _assert_equal(rows(a), rows(b), "rows")
    assert len(rows(a)) == 5
    _assert_equal(torch.load(a / "checkpoints" / "step_24.pt", weights_only=True),
                  torch.load(b / "checkpoints" / "step_24.pt", weights_only=True), "step_24.pt")


@pytest.mark.parametrize("bug", [None, "lr_from_total_steps"])
def test_a_longer_total_steps_on_resume_keeps_the_lr_series(tmp_path, monkeypatch, module_data_root, bug):
    """Run b: 12 updates, then resumed with total_steps 24; run a: 24 unbroken updates (the bug, when set, in both)."""
    def run(name, steps):
        with monkeypatch.context() as m:
            if bug:
                m.setattr(SL, "lr_at", lr_from_total_steps(steps))
            return _run(tmp_path, monkeypatch, module_data_root, name, steps=steps)

    a = run("a", 24)
    run("b", 12)
    b = run("b", 24)
    log = open(run_log(b)).read()
    assert "resuming at step 12;" in log and "total_steps 12 -> 24 (the stop point only" in log
    if bug:
        with pytest.raises(AssertionError, match="step lines differ"):
            assert_same_24_steps(a, b)
    else:
        assert_same_24_steps(a, b)


# ---------------------------------------------------------------- a steps_cosine run (before T-067)
def steps_cosine_lr(k: int, total_steps: int) -> float:
    """The LR of update k + 1 of a steps_cosine run (the pre-T-067 lr_lambda, restated)."""
    warmup = min(1000, total_steps // 10)
    if k < warmup:
        return SL.learning_rate * k / max(1, warmup)
    return SL.learning_rate * 0.5 * (1 + math.cos(math.pi * (k - warmup) / max(1, total_steps - warmup)))


def _as_steps_cosine(run_dir, total_steps):
    """Rewrite a run dir as one written before T-067: no lr_schedule in any config, the LambdaLR state of the old
    schedule (and its LR in the optimizer state) in every checkpoint, no T-067 checkpoint keys or row fields."""
    for path in sorted((run_dir / "checkpoints").glob("step_*.pt")):
        ckpt = torch.load(path, weights_only=True)
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([param], lr=SL.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: steps_cosine_lr(s, total_steps) / SL.learning_rate)
        for _ in range(ckpt["current_step"]):
            optimizer.step()
            scheduler.step()
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
        for group in ckpt["optimizer_state_dict"]["param_groups"]:
            group["lr"], group["initial_lr"] = optimizer.param_groups[0]["lr"], SL.learning_rate
        for key in ("last_update", "schedule_history"):
            ckpt.pop(key)
        ckpt["config"].pop("lr_schedule")
        torch.save(ckpt, path)
    metrics = SL.Metrics(str(run_dir / "metrics.jsonl"))
    header, rows = metrics.read()
    header["config"].pop("lr_schedule")
    metrics._write([header] + [{k: v for k, v in r.items() if k != "games_read"} for r in rows])


@pytest.fixture(scope="module")
def legacy(module_data_root):
    """A steps_cosine positions-mode run "L" (12-step horizon) stopped after update 6 (checkpoints 4 and 6), with a
    train.log inside as the CLI writes it. Read only: the migration must never write it."""
    with pytest.MonkeyPatch.context() as m:
        run = _run(module_data_root.parent, m, module_data_root, "L", stop_after_step=6)
    _as_steps_cosine(run, 12)
    shutil.copy(run_log(run), run / "train.log")
    with read_only_run(run) as run:
        yield run


def _copy_run(src, tmp_path, name):
    shutil.copytree(src, tmp_path / name)
    shutil.copy(run_log(src), tmp_path / f"{name}.log")
    return tmp_path / name


def test_a_steps_cosine_run_resumes_on_its_old_formula(tmp_path, monkeypatch, module_data_root, legacy):
    run = _copy_run(legacy, tmp_path, "old")
    with pytest.raises(AssertionError, match=r"step_6.pt is a steps_cosine run \(before T-067\): --lr-cycle-games"):
        _run(tmp_path, monkeypatch, module_data_root, "old")
    _run(tmp_path, monkeypatch, module_data_root, "old", lr_cycle_games=None)
    log = open(tmp_path / "old.log").read()
    assert "resuming at step 6;" in log and "LR schedule {'name': 'steps_cosine'}" in log
    logged = logged_lr(tmp_path / "old.log")
    bad = {s: (logged[s][0], steps_cosine_lr(s - 1, 12)) for s in range(7, 13)
           if abs(logged[s][0] - steps_cosine_lr(s - 1, 12)) > TOL}
    assert not bad, bad
    ckpt = torch.load(run / "checkpoints" / "step_12.pt", weights_only=True)
    assert ckpt["config"]["lr_schedule"] == SL.LEGACY_LR_SCHEDULE and ckpt["scheduler_state_dict"]["last_epoch"] == 12


# ---------------------------------------------------------------- T5: the migration
def _stop(*a, **k):
    raise KeyboardInterrupt("stop")


def _entry(legacy_dir) -> dict:
    ckpt = torch.load(legacy_dir / "checkpoints" / "step_6.pt", weights_only=True)
    return {"at_update": 6, "games": SL.games_read_of_state(ckpt), "from": "steps_cosine", "to": "games_cosine_v1"}


def assert_migrated(new, legacy_dir):
    entry = _entry(legacy_dir)
    assert sorted(os.listdir(new / "checkpoints")) == ["step_6.pt"]
    ckpt = torch.load(new / "checkpoints" / "step_6.pt", weights_only=True)
    header, rows = SL.Metrics(str(new / "metrics.jsonl")).read()
    for config in (ckpt["config"], header["config"]):
        assert config["lr_schedule"] == TEST_SCHEDULE
    assert ckpt["schedule_history"] == header["schedule_history"] == [entry]
    assert "scheduler_state_dict" not in ckpt
    src = torch.load(legacy_dir / "checkpoints" / "step_6.pt", weights_only=True)
    same = {k: v for k, v in src.items() if k not in ("config", "scheduler_state_dict")}
    _assert_equal(same, {k: v for k, v in ckpt.items() if k not in ("config", "schedule_history")}, "step_6.pt")
    assert rows == SL.Metrics(str(legacy_dir / "metrics.jsonl")).read()[1]
    assert open(new / "train.log").read().startswith(open(legacy_dir / "train.log").read())
    return entry


def games_reset_at(base):
    """Negative control: g counted from the resume (the games of the checkpoint lost)."""
    return lambda g, k, s: LR_AT(g - base, k, s)


@pytest.mark.parametrize("bug", [None, "games_reset_on_resume"])
def test_a_migrated_run_resumes_without_any_flag_on_the_games_schedule(tmp_path, monkeypatch, module_data_root,
                                                                       legacy, bug):
    """Switch at step 6, resume to step 9 (a stop and a save), resume again to 12: no flag. Every new update's LR
    equals lr_at of its logged games, the first one past the checkpoint's games (its own fill admitted more)."""
    monkeypatch.setattr(SL, "LR_WARMUP", LR_WARMUP)  # the migration writes the schedule the test runs use
    new = tmp_path / "N"
    assert MIG.migrate(str(legacy), str(new), TEST_LR_CYCLE_GAMES) == "copied the checkpoint and migrated it; " \
                                                                      "migrated the header"
    entry = assert_migrated(new, legacy)
    with monkeypatch.context() as m:
        if bug:
            m.setattr(SL, "lr_at", games_reset_at(entry["games"]))
        with Observed(m) as first:
            _run(tmp_path, monkeypatch, module_data_root, "N", stop_after_step=9)
        with Observed(m) as second:
            _run(tmp_path, monkeypatch, module_data_root, "N")
    log = open(tmp_path / "N.log").read()
    assert "resuming at step 6;" in log and "resuming at step 9;" in log and "Training completed." in log
    logged = logged_lr(tmp_path / "N.log")
    assert sorted(logged) == [7, 8, 9, 10, 11, 12] and logged[7][1] > entry["games"]
    observed = [([lr], g) for _, (lr, g) in sorted(logged.items())]  # the trainer's own g, logged
    assert [lrs[0] for lrs, _ in first.steps + second.steps] == [lr for (lr,), _ in observed]
    final = torch.load(new / "checkpoints" / "step_12.pt", weights_only=True)
    header, rows = SL.Metrics(str(new / "metrics.jsonl")).read()
    assert final["schedule_history"] == header["schedule_history"] == [entry]
    assert final["last_update"] == {"lr": logged[12][0], "games": logged[12][1]}
    assert (rows[-1]["step"], rows[-1]["lr"], rows[-1]["games_read"]) == (12, *logged[12])
    if bug:
        with pytest.raises(AssertionError, match=r"LR != lr_at\(games, k\)"):
            assert not lr_problems(observed, TEST_SCHEDULE, first_k=6), "LR != lr_at(games, k)"
    else:
        assert not lr_problems(observed, TEST_SCHEDULE, first_k=6), "LR != lr_at(games, k)"
        lrs = [lr for (lr,), _ in observed]
        assert all(a >= b for a, b in zip(lrs, lrs[1:])) and lrs[0] > SL.LR_FLOOR  # inside the cosine at the switch


def test_an_interrupted_migration_is_refused_by_the_trainer_and_completed_by_a_rerun(tmp_path, monkeypatch,
                                                                                    module_data_root, legacy):
    monkeypatch.setattr(SL, "LR_WARMUP", LR_WARMUP)
    new = tmp_path / "N"
    # a stop inside the copy (step 2): the new run dir does not exist yet, its staging folder does
    with monkeypatch.context() as m:
        m.setattr(MIG.Migration, "_migrate_checkpoint", _stop)
        with pytest.raises(KeyboardInterrupt):
            MIG.migrate(str(legacy), str(new), TEST_LR_CYCLE_GAMES)
    assert not new.exists() and os.path.exists(SL.migration_staging(str(new)))
    with pytest.raises(AssertionError, match=r"N.migrating exists: an interrupted lr-schedule migration into"):
        _run(tmp_path, monkeypatch, module_data_root, "N")
    assert not new.exists()
    # the re-run, stopped between the checkpoint write and the header write
    with monkeypatch.context() as m:
        m.setattr(MIG.Migration, "_migrate_header", _stop)
        with pytest.raises(KeyboardInterrupt):
            MIG.migrate(str(legacy), str(new), TEST_LR_CYCLE_GAMES)
    assert not os.path.exists(SL.migration_staging(str(new)))
    header, _ = SL.Metrics(str(new / "metrics.jsonl")).read()
    assert "lr_schedule" not in header["config"]
    with pytest.raises(AssertionError, match=r"header is steps_cosine, .*step_6.pt games_cosine_v1: an interrupted "
                                             r"lr-schedule migration; re-run python -m MahjongAI.migrate_lr_schedule"):
        _run(tmp_path, monkeypatch, module_data_root, "N")
    # an unrelated mismatch in that state is named as such, and the migration does not touch it
    metrics = SL.Metrics(str(new / "metrics.jsonl"))
    saved = open(metrics.path).read()
    header, rows = metrics.read()
    metrics._write([dict(header, config=dict(header["config"], embd_size=128))] + rows)
    with pytest.raises(AssertionError, match=r"metrics.jsonl header is from another run: embd_size 128 -> 64"):
        _run(tmp_path, monkeypatch, module_data_root, "N")
    with pytest.raises(AssertionError, match="is neither the migrated header nor the snapshot of the source's file"):
        MIG.migrate(str(legacy), str(new), TEST_LR_CYCLE_GAMES)
    open(metrics.path, "w").write(saved)
    # the re-run completes it, a third run does nothing, the trainer resumes without any flag
    assert MIG.migrate(str(legacy), str(new), TEST_LR_CYCLE_GAMES) == "migrated the header"
    assert_migrated(new, legacy)
    assert MIG.migrate(str(legacy), str(new), TEST_LR_CYCLE_GAMES) == "already migrated: nothing to do"
    _run(tmp_path, monkeypatch, module_data_root, "N", stop_after_step=7)
    assert "resuming at step 6;" in open(tmp_path / "N.log").read()
    assert MIG.migrate(str(legacy), str(new), TEST_LR_CYCLE_GAMES) == "already migrated: nothing to do"
    # the migrated run under another identity: refused as usual
    with pytest.raises(AssertionError, match=r"step_7.pt is from another run config: .*'cycle_games': 16.* -> "
                                             r".*'cycle_games': 17"):
        _run(tmp_path, monkeypatch, module_data_root, "N", lr_cycle_games=17)


def test_the_migration_refuses_what_it_cannot_switch(tmp_path, monkeypatch, train_data_root, legacy):
    # a source whose checkpoint and header disagree on an identity key
    bad = _copy_run(legacy, tmp_path, "bad")
    metrics = SL.Metrics(str(bad / "metrics.jsonl"))
    header, rows = metrics.read()
    metrics._write([dict(header, config=dict(header["config"], embd_size=128))] + rows)
    with pytest.raises(AssertionError, match=r"step_6.pt and .*metrics.jsonl header disagree: embd_size 64 -> 128"):
        MIG.migrate(str(bad), str(tmp_path / "bad_new"), TEST_LR_CYCLE_GAMES)
    # the fixture corpus with the default count (the corpus guard)
    with pytest.raises(AssertionError, match="give --lr-cycle-games N"):
        MIG.migrate(str(legacy), str(tmp_path / "default_new"))
    # an existing folder that is no migration of the source, and the source itself
    (tmp_path / "other" / "checkpoints").mkdir(parents=True)
    with pytest.raises(AssertionError, match="exists without lr_migration.json: not a migration made by this tool"):
        MIG.migrate(str(legacy), str(tmp_path / "other"), TEST_LR_CYCLE_GAMES)
    with pytest.raises(AssertionError, match="overlap: the new run dir and its staging folder must be outside"):
        MIG.migrate(str(legacy), str(legacy) + "/", TEST_LR_CYCLE_GAMES)
    # a run already on games_cosine_v1
    on_games = _run(tmp_path, monkeypatch, train_data_root, "on_games", steps=1)
    with pytest.raises(AssertionError, match=r"step_1.pt is lr_schedule .*games_cosine_v1.*, not steps_cosine"):
        MIG.migrate(str(on_games), str(tmp_path / "on_games_new"), TEST_LR_CYCLE_GAMES)
    # a games-mode steps_cosine state without its games counter (loader states before T-045)
    games = _run(tmp_path, monkeypatch, train_data_root, "games", steps=1, batch_mode="games")
    _as_steps_cosine(games, 1)
    MIG.migrate(str(games), str(tmp_path / "games_ok"), TEST_LR_CYCLE_GAMES)  # with the counter: accepted
    path = games / "checkpoints" / "step_1.pt"
    ckpt = torch.load(path, weights_only=True)
    for state in ckpt["loader_state"]["loaders"]:
        state.pop("accepted")
    torch.save(ckpt, path)
    with pytest.raises(AssertionError, match=r"loader states \[0\] have no `accepted` counter"):
        MIG.migrate(str(games), str(tmp_path / "games_new"), TEST_LR_CYCLE_GAMES)
    # a refusal leaves nothing: no new run dir, no staging folder
    assert not any((tmp_path / f"{n}{x}").exists() for n in ("bad_new", "default_new", "on_games_new", "games_new")
                   for x in ("", ".migrating"))


def _set_total_steps(run_dir, total_steps):
    """The run with another total_steps in its checkpoints and header (config_mismatch skips that key)."""
    for path in (run_dir / "checkpoints").glob("step_*.pt"):
        ckpt = torch.load(path, weights_only=True)
        ckpt["config"]["total_steps"] = total_steps
        torch.save(ckpt, path)
    metrics = SL.Metrics(str(run_dir / "metrics.jsonl"))
    header, rows = metrics.read()
    metrics._write([dict(header, config=dict(header["config"], total_steps=total_steps))] + rows)


def test_the_migration_never_removes_a_folder_it_did_not_stage(tmp_path, legacy):
    # a source at the staging path of the new run dir (<new>.migrating): refused before anything is written
    src = _copy_run(legacy, tmp_path, "S.migrating")
    before = tree_digest(src)
    with pytest.raises(AssertionError, match="overlap: the new run dir and its staging folder must be outside"):
        MIG.migrate(str(src), str(tmp_path / "S"), TEST_LR_CYCLE_GAMES)
    assert tree_digest(src) == before and not (tmp_path / "S").exists()
    # an unrelated folder at the staging path
    other = tmp_path / "U.migrating"
    other.mkdir()
    (other / "keep.txt").write_text("not a migration")
    with pytest.raises(AssertionError, match=r"U.migrating exists and is not a staging folder of the migration"):
        MIG.migrate(str(legacy), str(tmp_path / "U"), TEST_LR_CYCLE_GAMES)
    assert tree_digest(other) == {"U.migrating/keep.txt": hashlib.sha256(b"not a migration").hexdigest()}
    assert not (tmp_path / "U").exists()
    # the staging folder of a migration of another source into the same new run dir
    w = _copy_run(legacy, tmp_path, "W")
    with pytest.MonkeyPatch.context() as m:
        m.setattr(MIG.Migration, "_migrate_checkpoint", _stop)
        with pytest.raises(KeyboardInterrupt):
            MIG.migrate(str(w), str(tmp_path / "V"), TEST_LR_CYCLE_GAMES)
    staged = tree_digest(tmp_path / "V.migrating")
    with pytest.raises(AssertionError, match=r"V.migrating exists and is not a staging folder of the migration"):
        MIG.migrate(str(legacy), str(tmp_path / "V"), TEST_LR_CYCLE_GAMES)
    assert tree_digest(tmp_path / "V.migrating") == staged and not (tmp_path / "V").exists()


def test_a_new_run_dir_migrated_from_another_source_is_refused(tmp_path, legacy):
    # L2 differs from the fixture run only in total_steps, which config_mismatch skips
    l2 = _copy_run(legacy, tmp_path, "L2")
    _set_total_steps(l2, 24)
    new = tmp_path / "N"
    MIG.migrate(str(legacy), str(new), TEST_LR_CYCLE_GAMES)
    migrated = tree_digest(new)
    with pytest.raises(AssertionError, match=r"N is the migration of .*, not of .*L2"):
        MIG.migrate(str(l2), str(new), TEST_LR_CYCLE_GAMES)
    # the same source path, its recorded checkpoint changed since the migration
    src = _copy_run(legacy, tmp_path, "S")
    new2 = tmp_path / "N2"
    MIG.migrate(str(src), str(new2), TEST_LR_CYCLE_GAMES)
    _set_total_steps(src, 24)
    with pytest.raises(AssertionError, match=r"S/checkpoints/step_6.pt changed since .*N2 was migrated from it"):
        MIG.migrate(str(src), str(new2), TEST_LR_CYCLE_GAMES)
    assert tree_digest(new) == migrated


@pytest.mark.parametrize("change", ["rows_appended", "source_resumed"])
def test_an_interrupted_migration_completes_from_its_snapshot_after_the_source_changed(tmp_path, monkeypatch,
                                                                                      module_data_root, legacy, change):
    """Stopped between the checkpoint and header writes; then the source changes: a row appended to its metrics, or
    the source run resumed on its own schedule to step 8 (a newer checkpoint, its log and rows). The re-run completes
    the migration of step 6 from the snapshot it copied."""
    monkeypatch.setattr(SL, "LR_WARMUP", LR_WARMUP)
    src = _copy_run(legacy, tmp_path, "S")
    new = tmp_path / "N"
    with monkeypatch.context() as m:
        m.setattr(MIG.Migration, "_migrate_header", _stop)
        with pytest.raises(KeyboardInterrupt):
            MIG.migrate(str(src), str(new), TEST_LR_CYCLE_GAMES)
    if change == "rows_appended":
        metrics = SL.Metrics(str(src / "metrics.jsonl"))
        header, rows = metrics.read()
        metrics._write([header] + rows + [dict(rows[-1], step=rows[-1]["step"] + 1)])
    else:
        _run(tmp_path, monkeypatch, module_data_root, "S", lr_cycle_games=None, stop_after_step=8)
        assert SL.find_latest_checkpoint(str(src / "checkpoints"))[0] == 8
    assert MIG.migrate(str(src), str(new), TEST_LR_CYCLE_GAMES) == "migrated the header"
    assert_migrated(new, legacy)
    assert MIG.migrate(str(src), str(new), TEST_LR_CYCLE_GAMES) == "already migrated: nothing to do"
