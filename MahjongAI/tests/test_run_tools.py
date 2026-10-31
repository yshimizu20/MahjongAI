"""T-068 run tools: the self-tests of utils/fit_gap.py, utils/compare_runs.py and utils/mem_guard.py, mem_guard's log
tail on a real file, and fit_gap end to end on a 1-update fixture run (CPU): its val CE equals the logged row's, and a
logged row changed by more than VAL_TOL raises."""
import pytest

from MahjongAI import supervised_learning as SL
from MahjongAI.tests.test_positions_train import _run
from MahjongAI.utils import compare_runs, fit_gap, mem_guard
from MahjongAI.utils.phase_timing import CheckError


def test_fit_gap_self_test():
    fit_gap.self_test()


def test_compare_runs_self_test():
    compare_runs.self_test()


def test_mem_guard_self_test():
    mem_guard.self_test()


def test_mem_guard_reads_only_complete_lines_appended_after_its_start(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(mem_guard._mps(0, 8, " (post-eval/save)") + "\n")
    tail = mem_guard.LogTail(str(log))
    assert tail() == []
    with open(log, "a") as f:
        f.write(mem_guard._mps(100, 8952) + "\n" + mem_guard._mps(200, 8964)[:-20])  # the last line not finished
    assert mem_guard.driver_readings(tail()) == [(100, "", 8952)]
    with open(log, "a") as f:
        f.write(mem_guard._mps(200, 8964)[-20:] + "\n")
    assert mem_guard.driver_readings(tail()) == [(200, "", 8964)] and tail() == []
    assert mem_guard.LogTail(str(tmp_path / "missing.log"))() == []


def test_fit_gap_matches_the_trainers_logged_val_ce(tmp_path, monkeypatch, train_data_root, capsys):
    run = _run(tmp_path, monkeypatch, train_data_root, "fg", steps=1)  # rows 0 and 1, checkpoint step_1.pt
    out = fit_gap.fit_gap(str(run / "checkpoints" / "step_1.pt"))
    _, rows = SL.Metrics(str(run / "metrics.jsonl")).read()
    assert out["step"] == 1 and out["max_diff"] <= fit_gap.VAL_TOL
    assert all(abs(out["val"][h] - rows[-1]["heads"][h]["ce"]) <= fit_gap.VAL_TOL for h in SL.HEADS)
    assert all(out["train"][h] > 0 for h in SL.HEADS) and "gap (val - train)" in capsys.readouterr().out
    metrics = SL.Metrics(str(run / "metrics.jsonl"))
    header, rows = metrics.read()
    rows[-1]["heads"]["discard"]["ce"] += 2 * fit_gap.VAL_TOL
    metrics._write([header] + rows)
    with pytest.raises(CheckError, match="val CE differs from the logged row of step 1"):
        fit_gap.fit_gap(str(run / "checkpoints" / "step_1.pt"))


def test_a_trainer_started_as_a_background_job_still_stops_on_sigint():
    """A background job of a non-interactive shell starts with SIGINT ignored (the T-068 smoke: mem_guard's SIGINT
    did not stop the trainer); supervised_learning.restore_sigint gives it the KeyboardInterrupt handler back."""
    import subprocess
    import sys
    probe = ("import signal; from MahjongAI import supervised_learning as SL; before = signal.getsignal(signal.SIGINT); "
             "restored = SL.restore_sigint(); print(before, restored, signal.getsignal(signal.SIGINT) is "
             "signal.default_int_handler, SL.restore_sigint())")
    for cmd, expect in ((f"'{sys.executable}' -c '{probe}' & wait", "Handlers.SIG_IGN True True False"),
                        (f"'{sys.executable}' -c '{probe}'", "<built-in function default_int_handler> False True False")):
        out = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True, check=True).stdout.strip()
        assert out == expect, (cmd, out)
