"""The phase marks of --phase-timing (T-057 on the T-059 mark list), on the CPU (fixture games, positions mode).

Off path: every probe call in the trainer sits under `if probe is not None` (bit-for-bit equal runs with and without
the flag: test_phase_timing.py). Sequence: the check passes the mark list and fails a removed, reordered or
misplaced mark, a missing boundary group (also the last one before the stop), a skipped step, an unscheduled,
duplicate or backward cycle end and a trainer whose forward drops its head marks; a run with the flag on passes it
with one `phase-time` line per step."""
import ast
import inspect

import pytest

from MahjongAI import supervised_learning as SL
from MahjongAI.utils import phase_probe as PP
from MahjongAI.utils import phase_timing as PT
from MahjongAI.tests.test_positions_train import _run, step_lines

STEPS, SAVE_EVERY = 6, 3  # eval every 5 (_run): boundaries 0 (eval), 3 (save), 5 (eval), 6 (both)


def run(tmp_path, monkeypatch, root, name, stop=None, **kw):
    return _run(tmp_path, monkeypatch, root, name, steps=STEPS, save_every=SAVE_EVERY, stop_after_step=stop,
                phase_timing=True, **kw)


@pytest.mark.parametrize("stop", [None, 4])  # 4: a stop step that is no save step (the stop checkpoint)
def test_a_run_with_the_flag_passes_the_sequence_check(tmp_path, monkeypatch, train_data_root, stop):
    """Every cycle of the run passes the check (a SequenceError would stop the run) and each step has one
    phase-time line."""
    run(tmp_path, monkeypatch, train_data_root, "on", stop)
    last = stop or STEPS
    assert sorted(step_lines(tmp_path / "on.log")) == list(range(1, last + 1))
    assert sorted(PT.parse_run(open(tmp_path / "on.log"))["steps"]) == list(range(1, last + 1))


def unguarded_probe_calls(source: str) -> list:
    """Lines of the calls on `probe` (probe.mark(...)) outside the body of an `if probe is not None:`: work (a call,
    its argument tuple / dict) a run without --phase-timing would do."""
    tree = ast.parse(source)
    parent = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}

    def guarded(node):
        while node in parent:
            up = parent[node]
            if isinstance(up, ast.If) and node in up.body and ast.unparse(up.test) == "probe is not None":
                return True
            node = up
        return False
    return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "probe" and not guarded(n)]


def test_the_flag_off_path_makes_no_probe_call():
    source = inspect.getsource(SL)
    assert source.count("probe.mark(") >= 20 and unguarded_probe_calls(source) == []
    assert unguarded_probe_calls("if probe is not None:\n    probe.mark('a')\nprobe.mark('b', step=1)\n") == [3]


# ---------------------------------------------------------------- the sequence check on a scripted mark stream
def is_eval(s):
    return s % 5 == 0 or s == STEPS


def is_save(s):
    return s % SAVE_EVERY == 0 or s == STEPS


def script(start=0, stop=STEPS, heads=lambda s: PP.HEADS if s % 2 else ("discard", "post"), start_eval=True):
    """[(mark name, step or None, heads or None)]: the trainer's marks for a run from `start` to `stop` (a fresh run
    when start_eval)."""
    out = [("init", start, None)]
    if start_eval:
        out += [("eval_weights", None, None), ("eval", None, None), ("boundary", None, None)]
    for s in range(start + 1, stop + 1):
        names = PP.step_marks(heads(s))
        out += [(names[0], s, None), (names[1], None, heads(s))] + [(n, None, None) for n in names[2:]]
        if is_eval(s) or is_save(s):
            out += [(n, None, None) for n in (("eval_weights", "eval") if is_eval(s) else ())
                    + (("save",) if is_save(s) else ()) + ("boundary",)]
    if stop != start and not is_save(stop):
        out.append(("save", None, None))
    return out + [("stop", stop, None)]


class CycleCount:
    """A stand-in timer that counts the closed cycles (the probe hands it every mark)."""

    def __init__(self):
        self.ends = 0

    def start(self, mark):
        pass

    def interval(self, prev, mark):
        self.ends += mark.name in PP.CYCLE_ENDS


def play(marks) -> int:
    """Feeds the marks to a probe; returns the cycle ends it passed (the check runs at each)."""
    counter = CycleCount()
    probe = PP.PhaseProbe(lambda line: None, PP.cpu_no_op, is_eval, is_save, timer=counter)
    for name, step, heads in marks:
        probe.mark(name, step=step, heads=heads)
    return counter.ends


@pytest.mark.parametrize("start, stop, start_eval", [(0, STEPS, True), (0, 4, True), (3, 4, False), (5, 5, False)])
def test_the_sequence_check_passes_the_mark_list(start, stop, start_eval):
    assert play(script(start, stop, start_eval=start_eval)) == sum(
        is_eval(s) or is_save(s) for s in range(start + 1, stop + 1)) + start_eval + 1


def broken_scripts():
    good = script()
    names = [n for n, _, _ in good]
    out = {}
    for victim in ["top", "peek", "rng", "post:transfer", "discard:forward", "results", "loss", "finite", "zero_grad",
                   "backward", "clip", "opt_step", "sched_step", "commit", "sums", "log", "eval_weights", "eval",
                   "save", "init"]:
        i = 0 if victim == "init" else names.index(victim, names.index("top"))  # its first place in a step
        if victim == "top":  # the step id travels on top: the next mark carries it instead
            out["remove top"] = good[:i] + [(good[i + 1][0], good[i][1], good[i + 1][2])] + good[i + 2:]
        else:
            out[f"remove {victim}"] = good[:i] + good[i + 1:]
    i = names.index("zero_grad")
    out["swap zero_grad backward"] = good[:i] + [good[i + 1], good[i]] + good[i + 2:]
    i = names.index("during:transfer")
    out["swap heads"] = good[:i] + good[i + 2:i + 4] + good[i:i + 2] + good[i + 4:]
    first_boundary = names.index("boundary", names.index("top"))  # after step 3 (a save step)
    out["no boundary group at step 3"] = good[:first_boundary - 1] + good[first_boundary + 1:]
    i = names.index("top", names.index("top") + 1)  # step 2's top: give it step 3 (step 2 skipped in the ids)
    out["skip a step id"] = good[:i] + [("top", 3, None)] + good[i + 1:]
    i = names.index("peek")
    out["empty heads"] = good[:i] + [("peek", None, ())] + good[i + 1:]
    stop_at_4 = script(0, 4)
    assert stop_at_4[-2][0] == "save"
    out["no stop save"] = stop_at_4[:-2] + stop_at_4[-1:]
    return out


@pytest.mark.parametrize("broken", sorted(broken_scripts()))
def test_the_sequence_check_fails_a_broken_mark_stream(broken):
    with pytest.raises((PP.SequenceError, AssertionError)):
        play(broken_scripts()[broken])


def endpoint_scripts() -> dict:
    """{case: (marks, the error it must raise)}: a wrong cycle end. Each stream is cut right after that end, so the
    error comes from the end itself, not from a later cycle."""
    good = script()
    names = [n for n, _, _ in good]
    b3 = names.index("boundary", names.index("top"))  # the boundary after step 3 (save only)
    b5 = names.index("boundary", b3 + 1)  # after step 5 (eval only)
    assert [n for n, _, _ in good[-5:]] == ["eval_weights", "eval", "save", "boundary", "stop"]
    log1 = names.index("log")  # step 1: no eval / save step
    return {
        "stop without the last boundary group": (good[:-5] + good[-1:], "step 6 is an eval / save step"),
        "unscheduled boundary": (good[:log1 + 1] + [("boundary", None, None)], "no eval / save step"),
        "duplicate boundary": (good[:b3 + 1] + good[b3 - 1:b3 + 1], "must close later steps"),
        "backward boundary": (good[:b5 + 1] + [("save", 3, None), ("boundary", None, None)], "must close later steps"),
        "backward stop": (good[:-1] + [("stop", STEPS - 1, None)], "must close later steps"),
        "start boundary at no eval step": ([("init", 3, None), ("save", None, None), ("boundary", None, None)],
                                           "start step 3, which is no eval step"),
    }


@pytest.mark.parametrize("case", sorted(endpoint_scripts()))
def test_the_sequence_check_fails_a_wrong_cycle_end(case):
    marks, error = endpoint_scripts()[case]
    with pytest.raises(PP.SequenceError, match=error):
        play(marks)


def test_an_unknown_mark_fails():
    with pytest.raises(AssertionError, match="unknown mark 'discard:embed'"):
        play([("init", 3, None), ("top", 4, None), ("peek", None, ("discard",)), ("rng", None, None),
              ("discard:transfer", None, None), ("discard:embed", None, None)])


def test_a_trainer_whose_forward_drops_its_marks_stops(tmp_path, monkeypatch, train_data_root):
    real = SL.EncoderConvModel.forward_heads
    monkeypatch.setattr(SL.EncoderConvModel, "forward_heads", lambda self, batches, mark=None, cores=None: real(self, batches, cores=cores))
    with pytest.raises(PP.SequenceError, match="during:transfer|discard:transfer"):
        run(tmp_path, monkeypatch, train_data_root, "on")
