"""A kill after an eval row and before the next save (T-047 final runs): the run resumes from the earlier checkpoint
(a non-eval step, train_sums nonzero), drops the stale rows written after it, recomputes them, and ends equal to an
unbroken run: metrics rows, per-step log lines (losses, consumed-file hashes) and every checkpoint field."""
import os
import re
import shutil

import torch

from MahjongAI import supervised_learning as SL

STEPS, EVAL_EVERY, SAVE_EVERY = 12, 5, 4  # rows 0, 5, 10, 12; checkpoints 4, 8, 12


def _run(tmp_path, monkeypatch, root, name):
    monkeypatch.setattr(SL, "DATA_ROOT", root)
    monkeypatch.setattr(SL, "VAL_GAMES", 2)
    monkeypatch.setattr(SL, "PRIOR_GAMES", 2)
    run_dir = tmp_path / name
    log = SL.RunLog(str(tmp_path / f"{name}.log"))
    SL.train("encoder_conv", STEPS, str(run_dir), log, eval_every=EVAL_EVERY, save_every=SAVE_EVERY)
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


def test_kill_after_an_eval_row_before_the_save_resumes_to_the_unbroken_run(tmp_path, monkeypatch, train_data_root):
    threads = torch.get_num_threads()
    try:
        a = _run(tmp_path, monkeypatch, train_data_root, "a")
        b = _run(tmp_path, monkeypatch, train_data_root, "b")
        # the disk of a kill -9 after row 10 and before the step-12 save: checkpoints 4 and 8; rows 0, 5, 10
        os.remove(b / "checkpoints" / "step_12.pt")
        header, rows = SL.Metrics(str(b / "metrics.jsonl")).read()
        SL.Metrics(str(b / "metrics.jsonl"))._write([header] + [r for r in rows if r["step"] <= 10])
        resumed_from = torch.load(b / "checkpoints" / "step_8.pt", weights_only=True)
        assert all(n > 0 for _, n in resumed_from["train_sums"].values())  # steps 6..8 since row 5
        shutil.copy(tmp_path / "b.log", tmp_path / "b_first.log")
        _run(tmp_path, monkeypatch, train_data_root, "b")
    finally:
        torch.set_num_threads(threads)
    log_b = open(tmp_path / "b.log").read()
    assert "resuming at step 8;" in log_b and "metrics rows kept [0, 5]" in log_b, log_b[-2000:]
    ha, rows_a = SL.Metrics(str(a / "metrics.jsonl")).read()
    hb, rows_b = SL.Metrics(str(b / "metrics.jsonl")).read()
    assert [r["step"] for r in rows_a] == [r["step"] for r in rows_b] == [0, 5, 10, 12]
    _assert_equal({k: v for k, v in ha.items() if k != "started"}, {k: v for k, v in hb.items() if k != "started"},
                  "header")
    _assert_equal(rows_a, rows_b, "rows")
    lines_a, lines_b = _step_lines(tmp_path / "a.log"), _step_lines(tmp_path / "b.log")
    assert sorted(lines_a) == list(range(1, STEPS + 1)) and lines_a == lines_b
    for step in (4, 8, 12):
        _assert_equal(torch.load(a / "checkpoints" / f"step_{step}.pt", weights_only=True),
                      torch.load(b / "checkpoints" / f"step_{step}.pt", weights_only=True), f"step_{step}.pt")
