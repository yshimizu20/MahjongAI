"""A kill after an eval row and before the next save (T-047 final runs): the run resumes from the earlier checkpoint
(a non-eval step, train_sums nonzero), drops the stale rows written after it, recomputes them, and ends equal to an
unbroken run: metrics rows, per-step log lines (losses, consumed-file hashes) and every checkpoint field.
Also a stop inside an eval row's write: the resume starts at that eval step's checkpoint and recomputes its row.

The 64 x 1 unbroken run "a" is built once per module and only read (T-063, conftest.read_only_run).
The 128 x 2 case runs a shorter schedule (T-065, SCHEDULES): two of the 64 x 1 schedule's features do not depend on
the architecture (an intermediate eval row kept by the resume, two checkpoints on disk with the latest chosen), so the
64 x 1 case alone shows them. Every run builds its validation set and prior in the trainer (PRIOR_WORKERS 0, T-065):
test_startup_pool proves that path equal to the startup pool's, and the claims here are about the resume."""
import os
import re
import shutil
from typing import NamedTuple

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.tests.conftest import read_only_run, run_log


class Schedule(NamedTuple):
    """A run's schedule and its kill: the disk of a kill after row `kill_row` and before the next save holds the
    checkpoints <= `resume` and the rows <= `kill_row`; the resume keeps rows `kept` and recomputes the rest."""
    steps: int
    eval_every: int
    save_every: int
    rows: list
    checkpoints: list
    kill_row: int
    resume: int
    kept: list


SCHEDULES = {
    (64, 1): Schedule(12, 5, 4, rows=[0, 5, 10, 12], checkpoints=[4, 8, 12], kill_row=10, resume=8, kept=[0, 5]),
    (128, 2): Schedule(6, 3, 2, rows=[0, 3, 6], checkpoints=[2, 4, 6], kill_row=3, resume=2, kept=[0]),
}
STEPS, EVAL_EVERY, SAVE_EVERY = SCHEDULES[(64, 1)][:3]  # test 2 and the shared run "a"


def _run(tmp_path, monkeypatch, root, name, arch=(64, 1)):
    monkeypatch.setattr(SL, "DATA_ROOT", root)
    monkeypatch.setattr(SL, "VAL_GAMES", 2)
    monkeypatch.setattr(SL, "PRIOR_GAMES", 2)
    monkeypatch.setattr(SL, "PRIOR_WORKERS", 0)  # validation set and prior built in the trainer (T-065)
    run_dir = tmp_path / name
    log = SL.RunLog(str(tmp_path / f"{name}.log"))
    sc = SCHEDULES[arch]
    # the 128 x 2 init model sits 0.059 below uniform-legal on the fixture's post head: the T-010 run's --row0-tol 0.1
    SL.train("encoder_conv", sc.steps, str(run_dir), log, eval_every=sc.eval_every, save_every=sc.save_every,
             embd_size=arch[0], n_layers=arch[1], row0_tol=0.1 if arch != (64, 1) else None)
    return run_dir


def _step_lines(path):
    """{step: the per-step log line without its timestamp}, the last line of each step (a resume re-logs it)."""
    out = {}
    for line in open(path):
        m = re.match(r"\[[^]]+\] (step (\d+) \| (?!val).*)$", line.rstrip("\n"))
        if m:
            out[int(m.group(2))] = m.group(1)
    return out


def _assert_equal(a, b, where):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b), where
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), where
        for k in a:
            _assert_equal(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_equal(x, y, f"{where}[{i}]")
    else:
        assert a == b, f"{where}: {a!r} vs {b!r}"


@pytest.fixture(scope="module")
def unbroken_64(module_data_root):
    """The unbroken 64 x 1 run "a", built once per module on module_data_root; tests only read it."""
    threads = torch.get_num_threads()
    try:
        with pytest.MonkeyPatch.context() as m:
            a = _run(module_data_root.parent, m, module_data_root, "a")
    finally:
        torch.set_num_threads(threads)
    with read_only_run(a) as a:
        yield a


@pytest.mark.parametrize("arch", [(64, 1), (128, 2)])
def test_kill_after_an_eval_row_before_the_save_resumes_to_the_unbroken_run(tmp_path, monkeypatch, request, arch):
    """64 x 1: "a" is the module's unbroken run, and b's disk before the kill is a copy of it (run dir and log): b's
    first run is the same command on the same root, so the copy is the disk that run leaves (T-063). What the copy no
    longer shows here, that two fresh 64 x 1 runs agree over steps 1-8, the next test shows (a fresh b up to row 10),
    as do test_positions_train and test_worker_sweep. 128 x 2 keeps two fresh runs: the only fresh-run comparison at
    that architecture, on its shorter schedule (SCHEDULES: kill after row 3, resume at checkpoint 2)."""
    sc = SCHEDULES[arch]
    # the kill is after an eval row that is no save step, and the resume checkpoint is no eval step (train sums open)
    assert sc.kill_row in sc.rows and sc.kill_row % sc.save_every and sc.resume % sc.eval_every
    assert sc.resume == max(c for c in sc.checkpoints if c <= sc.kill_row)
    assert sc.kept == [r for r in sc.rows if r <= sc.resume] and sc.rows[-1] == sc.checkpoints[-1] == sc.steps
    threads = torch.get_num_threads()
    try:
        if arch == (64, 1):
            root, a = request.getfixturevalue("module_data_root"), request.getfixturevalue("unbroken_64")
            b = tmp_path / "b"
            shutil.copytree(a, b)
            shutil.copy(run_log(a), tmp_path / "b.log")
        else:
            root = request.getfixturevalue("train_data_root")
            a = _run(tmp_path, monkeypatch, root, "a", arch)
            b = _run(tmp_path, monkeypatch, root, "b", arch)
        # the disk of a kill -9 after row kill_row and before the next save (64 x 1: after row 10, before the step-12
        # save: checkpoints 4 and 8; rows 0, 5, 10. 128 x 2: after row 3, before the step-4 save: checkpoint 2; rows 0, 3)
        for step in sc.checkpoints:
            if step > sc.resume:
                os.remove(b / "checkpoints" / f"step_{step}.pt")
        header, rows = SL.Metrics(str(b / "metrics.jsonl")).read()
        SL.Metrics(str(b / "metrics.jsonl"))._write([header] + [r for r in rows if r["step"] <= sc.kill_row])
        resumed_from = torch.load(b / "checkpoints" / f"step_{sc.resume}.pt", weights_only=True)
        assert all(n > 0 for _, n in resumed_from["train_sums"].values())  # the steps since the last row before it
        shutil.copy(tmp_path / "b.log", tmp_path / "b_first.log")
        _run(tmp_path, monkeypatch, root, "b", arch)
    finally:
        torch.set_num_threads(threads)
    log_b = open(tmp_path / "b.log").read()
    assert f"resuming at step {sc.resume};" in log_b and f"metrics rows kept {sc.kept}" in log_b, log_b[-2000:]
    ha, rows_a = SL.Metrics(str(a / "metrics.jsonl")).read()
    hb, rows_b = SL.Metrics(str(b / "metrics.jsonl")).read()
    assert [r["step"] for r in rows_a] == [r["step"] for r in rows_b] == sc.rows
    _assert_equal({k: v for k, v in ha.items() if k != "started"}, {k: v for k, v in hb.items() if k != "started"},
                  "header")
    _assert_equal(rows_a, rows_b, "rows")
    lines_a, lines_b = _step_lines(run_log(a)), _step_lines(tmp_path / "b.log")
    assert sorted(lines_a) == list(range(1, sc.steps + 1)) and lines_a == lines_b
    for step in sc.checkpoints:
        _assert_equal(torch.load(a / "checkpoints" / f"step_{step}.pt", weights_only=True),
                      torch.load(b / "checkpoints" / f"step_{step}.pt", weights_only=True), f"step_{step}.pt")


def test_stop_inside_an_eval_row_write_resumes_from_that_steps_checkpoint_and_recomputes_its_row(
        tmp_path, monkeypatch, module_data_root, unbroken_64):
    """A stop while row 10 is written (an eval step): the stop path saves checkpoint 10 (train_sums of steps 6..10,
    not reset), whose own row is absent. The resume starts at step 10, recomputes row 10 before any update, and ends
    equal to an unbroken run: rows, step lines and checkpoints 4, 8, 12."""
    append = SL.Metrics.append

    def stop_at_row_10(self, row):
        if row.get("type") == "eval" and row["step"] == 10:
            raise RuntimeError("stopped inside the row-10 write")
        append(self, row)

    threads = torch.get_num_threads()
    try:
        a = unbroken_64
        with monkeypatch.context() as m:
            m.setattr(SL.Metrics, "append", stop_at_row_10)
            try:
                _run(tmp_path, monkeypatch, module_data_root, "b")
                raise AssertionError("the stop at row 10 did not happen")
            except RuntimeError as e:
                assert "stopped inside the row-10 write" in str(e), e
        b = tmp_path / "b"
        assert sorted(os.listdir(b / "checkpoints")) == ["step_10.pt", "step_4.pt", "step_8.pt"]
        _, rows = SL.Metrics(str(b / "metrics.jsonl")).read()
        assert [r["step"] for r in rows] == [0, 5]
        stopped = torch.load(b / "checkpoints" / "step_10.pt", weights_only=True)
        assert stopped["current_step"] == 10 and all(n > 0 for _, n in stopped["train_sums"].values())
        _run(tmp_path, monkeypatch, module_data_root, "b")
    finally:
        torch.set_num_threads(threads)
    log_b = open(tmp_path / "b.log").read()
    assert "resuming at step 10;" in log_b and "metrics rows kept [0, 5]" in log_b, log_b[-2000:]
    ha, rows_a = SL.Metrics(str(a / "metrics.jsonl")).read()
    hb, rows_b = SL.Metrics(str(b / "metrics.jsonl")).read()
    assert [r["step"] for r in rows_a] == [r["step"] for r in rows_b] == [0, 5, 10, 12]
    _assert_equal(rows_a, rows_b, "rows")
    lines_a, lines_b = _step_lines(run_log(a)), _step_lines(tmp_path / "b.log")
    assert sorted(lines_a) == list(range(1, STEPS + 1)) and lines_a == lines_b
    for step in (4, 8, 12):
        _assert_equal(torch.load(a / "checkpoints" / f"step_{step}.pt", weights_only=True),
                      torch.load(b / "checkpoints" / f"step_{step}.pt", weights_only=True), f"step_{step}.pt")
