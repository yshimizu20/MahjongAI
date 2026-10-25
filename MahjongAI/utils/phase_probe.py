"""Phase marks of a positions-mode training step (the T-059 mark list), read by the T-057 phase timer.

The trainer calls mark(name) after each operation of the mark list (plan T-059 "Stage 4: mark list"). Each mark reads
(on MPS: torch.mps.synchronize(); the reading's memory counters are zeros, this probe reads no memory) and stores
time.perf_counter() after the reading; a PhaseTimer (utils/phase_timing.py, T-057 --phase-timing) reads those times.
The marks of a cycle (from the previous `boundary` / `init` to the next `boundary` / `stop`) are checked against the
expected sequence, then dropped: a mark the trainer lost or misplaced raises SequenceError. The probe writes no log
line; the timer writes the `phase-time` lines."""
import time
from typing import NamedTuple

import torch

HEADS = ("during", "discard", "post")
STEP_HEAD_START = ("top", "peek", "rng")  # then <head>:transfer, <head>:forward per present head
STEP_TAIL = ("results", "loss", "finite", "zero_grad", "backward", "clip", "opt_step", "sched_step", "commit", "sums",
             "log")
CATEGORY = {
    "top": "log/bookkeeping", "peek": "gap/data", "rng": "log/bookkeeping", "results": "forward", "loss": "forward",
    "finite": "log/bookkeeping", "zero_grad": "update", "backward": "backward", "clip": "update", "opt_step": "update",
    "sched_step": "update", "commit": "gap/data", "sums": "log/bookkeeping", "log": "log/bookkeeping",
    "eval_weights": "eval", "eval": "eval", "save": "save", "boundary": "log/bookkeeping", "stop": "log/bookkeeping",
    **{f"{h}:transfer": "transfer" for h in HEADS}, **{f"{h}:forward": "forward" for h in HEADS},
}
CYCLE_ENDS = ("boundary", "stop")


def head_marks(head: str) -> list:
    """The marks of one head's forward: transfer, forward."""
    return [f"{head}:transfer", f"{head}:forward"]


def step_marks(heads) -> list:
    """The marks of one training step with the given present heads, in order."""
    return [*STEP_HEAD_START, *(k for h in heads for k in head_marks(h)), *STEP_TAIL]


def category(name: str) -> str:
    return CATEGORY[name]


def is_mark(name: str) -> bool:
    return name in CATEGORY


def mps_sync_only():
    """The reading on MPS: the sync, no counters (zeros)."""
    torch.mps.synchronize()
    return 0, 0


def cpu_no_op():
    """The reading on the CPU: no sync, no counters (zeros; the probe still runs every mark and the sequence check)."""
    return 0, 0


def sync_reader_for(device: torch.device):
    """The reading function of the probe (T-057): the sync on MPS, a no-op on the CPU."""
    return mps_sync_only if device.type == "mps" else cpu_no_op


def sync_for(device: torch.device):
    """Waits for the device's queued work (MPS); a no-op on the CPU."""
    return torch.mps.synchronize if device.type == "mps" else (lambda: None)


class Mark(NamedTuple):
    step: int
    name: str
    driver: int
    current: int
    t: float  # time.perf_counter()
    heads: tuple = None  # `peek` only: the heads present in the step


class SequenceError(AssertionError):
    pass


class PhaseProbe:
    """Owns the marks of the current cycle and the sequence check.

    is_eval / is_save: the trainer's step predicates (which steps get a boundary group). reader: () -> (driver,
    current) after a sync (sync_reader_for: zeros). timer: a phase_timing.PhaseTimer (T-057) that reads the marks'
    times, or None. log: the run log (the probe itself writes no line)."""

    def __init__(self, log, reader, is_eval, is_save, timer=None):
        self.log, self.reader, self.is_eval, self.is_save = log, reader, is_eval, is_save
        self.timer = timer
        self.step = None
        self.start_step = None
        self.last = None  # the previous Mark
        self.prev_end = None  # the step of the previous cycle end
        self.cycle = []  # the Marks since the last cycle end (init included in the first cycle)

    def mark(self, name: str, step: int = None, heads=None):
        """Read, pass the interval that ends here to the timer; at a cycle end check the cycle's sequence."""
        assert is_mark(name) or name == "init", f"unknown mark {name!r}"
        if step is not None:
            self.step = step
        driver, current = self.reader()
        m = Mark(self.step, name, int(driver), int(current), time.perf_counter(),
                 None if heads is None else tuple(heads))
        if name == "init":
            assert self.last is None, "init after other marks"
            self.start_step, self.last, self.cycle = m.step, m, [m]
            if self.timer is not None:
                self.timer.start(m)
            return
        assert self.last is not None, f"mark {name!r} before init"
        if self.timer is not None:
            self.timer.interval(self.last, m)
        self.last = m
        self.cycle.append(m)
        if name in CYCLE_ENDS:
            self._close_cycle(m)

    def _close_cycle(self, end: Mark):
        check_sequence(self.cycle, self.prev_end, self.start_step, self.is_eval, self.is_save)
        self.cycle, self.prev_end = [], end.step


def expected_cycle(cycle, prev_end, start_step: int, is_eval, is_save) -> list:
    """[(step, mark name)] the cycle must have: built from the previous cycle end's step (None: the cycle starts with
    init), its last mark (the cycle end and its step) and the heads its `peek` marks recorded. Raises when the cycle
    end itself is wrong: a step before the previous end (or the start), a boundary that closes no step (duplicate),
    a boundary at a step that is no eval / save step, or a stop that follows an eval / save step of its own cycle
    (that step's boundary group is missing)."""
    assert cycle and cycle[-1].name in CYCLE_ENDS, f"a cycle ends with {CYCLE_ENDS}: {cycle[-1:]}"
    end = cycle[-1]
    heads = {m.step: m.heads for m in cycle if m.name == "peek"}
    out = []
    last = start_step if prev_end is None else prev_end
    if end.step < last or (end.name == "boundary" and end.step == last and prev_end is not None):
        raise SequenceError(f"{end.name} at step {end.step} after the cycle end / start at step {last}: a cycle end "
                            "must close later steps (a boundary at least one)")
    if end.name == "boundary" and not (is_eval(end.step) or is_save(end.step)):
        raise SequenceError(f"boundary at step {end.step}, which is no eval / save step")
    if prev_end is None:
        first = start_step + 1
        out.append((start_step, "init"))
        if end.name == "boundary" and end.step == start_step:  # the start step writes its row (no save)
            if not is_eval(start_step):
                raise SequenceError(f"boundary at the start step {start_step}, which is no eval step")
            return out + [(end.step, n) for n in ("eval_weights", "eval", "boundary")]
    else:
        first = prev_end + 1
    for s in range(first, end.step + 1):
        out += [(s, n) for n in step_marks(heads.get(s) or ())]
        if (s < end.step or end.name == "stop") and (is_eval(s) or is_save(s)):
            raise SequenceError(f"step {s} is an eval / save step but its cycle runs on to the {end.name} at step "
                                f"{end.step}: its boundary marks are missing")
    if end.name == "boundary":
        out += [(end.step, n) for n in (("eval_weights", "eval") if is_eval(end.step) else ())
                + (("save",) if is_save(end.step) else ()) + ("boundary",)]
    else:
        out += [(end.step, n) for n in (("save",) if end.step != start_step and not is_save(end.step) else ())
                + ("stop",)]
    return out


def check_sequence(cycle, prev_end, start_step: int, is_eval, is_save):
    """Raises SequenceError at the first mark that differs from the expected cycle (step ids included), or when a
    step's heads are not a non-empty subsequence of HEADS; returns (the cycle's training steps, {head: steps it
    was present in})."""
    for m in cycle:
        if m.name == "peek" and (not m.heads or list(m.heads) != [h for h in HEADS if h in m.heads]):
            raise SequenceError(f"step {m.step}: heads {m.heads} are not a non-empty subsequence of {HEADS}")
    expected = expected_cycle(cycle, prev_end, start_step, is_eval, is_save)
    observed = [(m.step, m.name) for m in cycle]
    if observed != expected:
        i = next((i for i, (a, b) in enumerate(zip(observed, expected)) if a != b), min(len(observed), len(expected)))
        raise SequenceError(f"phase mark sequence: mark {i} is {observed[i] if i < len(observed) else '<none>'}, "
                            f"expected {expected[i] if i < len(expected) else '<none>'} ({len(observed)} marks, "
                            f"{len(expected)} expected; around: {observed[max(0, i - 3):i + 3]})")
    peeks = [m for m in cycle if m.name == "peek"]
    return [m.step for m in peeks], {h: sum(h in m.heads for m in peeks) for h in HEADS}
