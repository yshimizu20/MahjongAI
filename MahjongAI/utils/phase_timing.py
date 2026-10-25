"""Phase timing of a positions-mode training step (T-057, `--phase-timing`) and its measure window
(`--measure-window W,M`).

PhaseTimer reads the marks of the PhaseProbe (utils/phase_probe.py, the T-059 mark list). Each mark stores
time.perf_counter() after the probe's reading, and on MPS that reading synchronizes first. The timer times explicit
operation spans: the trainer calls timer.begin(<mark>) right before an operation (a sync on MPS, then the clock) and
the operation's mark ends the span, so a span holds the operation and its GPU work only. Inside model.forward_heads,
which has no code between its operations, a mark chains the next span (CHAIN_FROM: `rng` opens the first head's
transfer, `<head>:transfer` its forward, `<head>:forward` the next head's transfer or `results`). A span goes to the
phase of its end mark (PHASE_OF_CATEGORY). The time between spans (the loop's own code, a mark without its begin) goes
to no phase: `unattributed`, which check C1 catches. The intervals that end at an eval, save, boundary or stop mark are
no training-step work: "other". The exposed wait for the data workers' results (GameWorkers.exposed_wait_s, the
blocked condition waits of fetch) happens inside the `peek` span; the timer reads the workers' wait clock at each
span's start and end and moves the change to `wait`, so `admit` is the foreground data work only.

Plan T-057 "Phases":
    wait      exposed wait for worker results (GameWorkers.exposed_wait_s)
    admit     peek (draws, unpack, row checks, provenance, admission concat, buffer guards, batch build) minus the
              wait, and commit (row removal, counters, sample digest)
    transfer  <head>:transfer (the .to(device) of a head's batch)
    forward   <head>:forward, results, loss
    backward  backward
    update    zero_grad, clip, opt_step, sched_step
    overhead  rng (RNG snapshot), finite (finite check), sums (scalar reads), log (the step line)

Log lines (through the run log, after its timestamp; seconds):
    phase-time step <k> | wait <s> | admit <s> | transfer <s> | forward <s> | backward <s> | update <s> |
        overhead <s> | other <s> | sum <s> | unattributed <s> | wall <s>
        one per step, at its `log` mark. sum: the seven phases; wall: the time from the previous step's `log` mark (or
        the init mark) to this one, an independent clock; unattributed: wall - sum - other, the time outside every
        span. The line itself is written after that mark: its cost is the next step's unattributed time.
    measure-window start step <W> | warmup <W> measure <M>
    measure-window memory <start|end> step <k> | <counter> <MB> MB | ...
        the window's two memory samples, taken at its endpoints: MPS driver and current MB, CPU the process RSS MB
    measure-window end step <W+M> | steps <W+1>-<W+M> | outer <s> s | rows during <n> discard <n> post <n> total <n> |
        admitted games <g> during <n> discard <n> post <n> total <n> | ms per 1k positions <x> | exposed wait <s> s |
        workers <n> busy <s>,... s files <n>,...
        outer: from the completion of update W to the completion of update W+M (MeasureWindow.update_done, called
        right after the scheduler step of every update: a sync on MPS, then the clock; the workers' exposed wait and
        busy counters are read right after it); rows: rows trained in the window
        per head (the batcher's rows_out counters, read after each endpoint update's commit, checked against the steps'
        own row counts); admitted: games and rows admitted in the window; busy / files: per data worker, the seconds it
        spent on parse + build + serialize and the files it finished in the window (in parallel with the trainer: never
        part of the wall time)
    measure-window <eval|save> step <W+M> | <s> s
        the one eval and the one save after the window, each timed alone (on MPS between two syncs)

The pure functions at the end (log parsing, window hash, checks C1, C2, gate D1, the epoch projection) are shared by
the tests and the T-057 analysis script, so the trainer's format and the gates cannot drift apart.
"""
import hashlib
import math
import re
import time
from typing import NamedTuple

from MahjongAI.utils import phase_probe as PP

HEADS = PP.HEADS
PHASES = ("wait", "admit", "transfer", "forward", "backward", "update", "overhead")
PHASE_OF_CATEGORY = {"gap/data": "admit", "transfer": "transfer", "forward": "forward", "backward": "backward",
                     "update": "update", "log/bookkeeping": "overhead"}
OUTSIDE_STEP = ("eval_weights", "eval", "save", "boundary", "stop")  # their intervals: "other"


def phase_of(name: str):
    """The phase of the span that ends at mark `name`; None (other) for the eval / save / boundary / stop marks."""
    return None if name in OUTSIDE_STEP else PHASE_OF_CATEGORY[PP.category(name)]


def chains(name: str) -> bool:
    """True: mark `name` opens the next span itself (CHAIN_FROM, inside model.forward_heads)."""
    return name == "rng" or name.endswith((":transfer", ":forward"))


def step_line(step: int, acc: dict, wall: float) -> str:
    total = sum(acc[p] for p in PHASES)
    return (f"phase-time step {step} | " + " | ".join(f"{p} {acc[p]:.6f}" for p in PHASES + ("other",))
            + f" | sum {total:.6f} | unattributed {wall - total - acc['other']:.6f} | wall {wall:.6f}")


class PhaseTimer:
    """Owns the phase seconds of the current step and the open span. The trainer calls begin(name) right before an
    operation; the probe calls start(init mark) once and interval(previous mark, mark) for every later mark. A span
    closes at the mark of its name (a chained span: at the next mark); a mark with no open span
    adds nothing to a phase. At a `log` mark the step's line is written and the accumulators restart.
    wait_clock: () -> the data workers' cumulative exposed wait in seconds (None: no workers, the wait is 0).
    sync: () -> None, waits for the device's queued work (MPS) before a span's start time. clock: the span start's
    clock (a test's fake one; the marks' times are the probe's perf_counter)."""

    CHAINED = "<chained>"

    def __init__(self, log, wait_clock=None, sync=lambda: None, clock=time.perf_counter):
        self.log, self.sync, self.clock = log, sync, clock
        self.wait_clock = wait_clock if wait_clock is not None else (lambda: 0.0)
        self.acc = dict.fromkeys(PHASES + ("other",), 0.0)
        self.open = None  # (span name or CHAINED, start time, wait clock at the start)
        self.step_t = None  # the time of the previous `log` mark (or init): the start of the step's wall

    def start(self, mark):
        self.step_t = mark.t

    def begin(self, name: str):
        """The span of operation `name` starts now (after a sync); its mark ends it."""
        assert self.open is None, f"begin {name!r} while the span {self.open[0]!r} is open"
        self.sync()
        self.open = (name, self.clock(), self.wait_clock())

    def interval(self, prev, mark):
        assert self.step_t is not None, f"interval {prev.name}->{mark.name} before start"
        name = mark.name
        if self.open is not None:
            span, t0, w0 = self.open
            assert span in (name, self.CHAINED), f"mark {name!r} while the span {span!r} is open"
            waited = self.wait_clock() - w0
            self.acc["wait"] += waited
            self.acc[phase_of(name) or "other"] += (mark.t - t0) - waited
            self.open = None
        elif phase_of(name) is None:
            self.acc["other"] += mark.t - prev.t  # eval / save / boundary / stop: no training-step work
        if chains(name):
            self.open = (self.CHAINED, mark.t, self.wait_clock())
        if name == "log":
            self.log(step_line(mark.step, self.acc, mark.t - self.step_t))
            self.acc = dict.fromkeys(PHASES + ("other",), 0.0)
            self.step_t = mark.t


def memory_for(device):
    """The measure window's endpoint memory sample of the device: MPS driver and current MB (no sync), CPU the
    process RSS MB."""
    if device.type == "mps":
        import torch
        return lambda: {"driver": torch.mps.driver_allocated_memory() / 2 ** 20,
                        "current": torch.mps.current_allocated_memory() / 2 ** 20}
    import psutil
    proc = psutil.Process()
    return lambda: {"rss": proc.memory_info().rss / 2 ** 20}


class Reading(NamedTuple):
    t: float
    rows_out: dict  # {head: rows trained since the start of the run (batcher counters)}
    rows_in: dict  # {head: rows admitted}
    games: int  # games admitted
    step_rows: dict  # {head: rows the steps trained, by their own count (the trainer's train_sums n)}
    wait: float  # the workers' cumulative exposed wait
    busy: list  # [(seconds, files)] per data worker


def counts(d: dict) -> int:
    return sum(d.values())


class MeasureWindow:
    """--measure-window W,M: an outer timer over the M updates after a W-update warm-up, the rows trained and admitted
    in them, the workers' busy time, a memory sample at each end, and the one eval and save after them, timed apart.

    The trainer calls update_done(k) right after the scheduler step of update k (the update's completion: the same
    code point at both ends) and step_end(k) after the step's log line (the counters, after the commit). sync: () ->
    None, waits for the device's queued work (MPS) so the time closes a completed update. memory: () -> {counter: MB},
    the endpoint memory sample (MPS driver / current, CPU RSS). step_rows: () -> {head: rows trained so far by the
    steps' own count}, which must not reset inside the window (no eval inside it). workers: the run's GameWorkers or
    None. clock: a test's fake clock."""

    def __init__(self, log, warmup: int, measure: int, sync, batcher, step_rows, workers=None,
                 memory=lambda: {}, clock=time.perf_counter):
        assert warmup >= 1 and measure >= 1, f"measure window {warmup},{measure}: both must be >= 1"
        self.log, self.warmup, self.measure, self.sync, self.memory = log, warmup, measure, sync, memory
        self.clock = clock
        self.batcher, self.step_rows, self.workers = batcher, step_rows, workers
        self.times = {}  # {endpoint update: perf_counter time of its completion}
        self.counters = {}  # {endpoint update: (the workers' exposed wait, busy seconds), read right after the clock}
        self.mem = {}  # {endpoint update: its memory sample}
        self.start = None

    @property
    def last(self) -> int:
        return self.warmup + self.measure

    def _worker_counters(self) -> tuple:
        w = self.workers
        return (0.0 if w is None else w.exposed_wait_s, [] if w is None else w.busy_seconds())

    def update_done(self, step: int):
        """Right after the scheduler step of update `step`, in this exact order (plan T-061 "Window endpoint samples"):
        start (update W): the memory sample -> sync -> clock -> worker counters, so the sample's cost lands before the
        window's clock; end (update W+M): sync -> clock -> worker counters -> the memory sample, so no sample is inside
        the window or in the busy deltas."""
        if step == self.warmup:
            self.mem[step] = self.memory()
            self.sync()
            self.times[step] = self.clock()
            self.counters[step] = self._worker_counters()
        elif step == self.last:
            self.sync()
            self.times[step] = self.clock()
            self.counters[step] = self._worker_counters()
            self.mem[step] = self.memory()

    def _reading(self, step: int) -> Reading:
        assert step in self.times, f"measure window: no completion time of update {step} (update_done not called)"
        b = self.batcher
        wait, busy = self.counters[step]
        return Reading(self.times[step], {h: counts(b.rows_out[h]) for h in HEADS},
                       {h: counts(b.rows_in[h]) for h in HEADS}, counts(b.games_admitted), dict(self.step_rows()),
                       wait, busy)

    def _memory_line(self, edge: str, step: int) -> str:
        return (f"measure-window memory {edge} step {step} | "
                + (" | ".join(f"{k} {v:.0f} MB" for k, v in self.mem[step].items()) or "none"))

    def step_end(self, step: int):
        """Called once per completed update, after its log line."""
        if step == self.warmup:
            self.start = self._reading(step)
            self.log(f"measure-window start step {step} | warmup {self.warmup} measure {self.measure}")
            self.log(self._memory_line("start", step))
        elif step == self.last:
            assert self.start is not None, f"measure window end at step {step} without its start"
            self.log(self._memory_line("end", step))
            self.log(window_line(self.warmup, self.last, self.start, self._reading(step)))

    def timed(self, label: str, step: int, fn):
        """fn(); at the window's last step timed alone (a sync before and after) and logged."""
        if step != self.last:
            return fn()
        self.sync()
        t0 = self.clock()
        out = fn()
        self.sync()
        self.log(f"measure-window {label} step {step} | {self.clock() - t0:.6f} s")
        return out


def window_line(warmup: int, last: int, a: Reading, b: Reading) -> str:
    """The end line; raises when the batcher's rows trained differ from the steps' own row counts."""
    rows = {h: b.rows_out[h] - a.rows_out[h] for h in HEADS}
    by_steps = {h: b.step_rows[h] - a.step_rows[h] for h in HEADS}
    assert rows == by_steps, f"window rows trained: batcher counters {rows} != the steps' own counts {by_steps}"
    admitted = {h: b.rows_in[h] - a.rows_in[h] for h in HEADS}
    outer, total = b.t - a.t, counts(rows)
    busy = [(sb - sa, fb - fa) for (sa, fa), (sb, fb) in zip(a.busy, b.busy)]
    return (f"measure-window end step {last} | steps {warmup + 1}-{last} | outer {outer:.6f} s | rows "
            + " ".join(f"{h} {rows[h]}" for h in HEADS) + f" total {total} | admitted games {b.games - a.games} "
            + " ".join(f"{h} {admitted[h]}" for h in HEADS) + f" total {counts(admitted)} | ms per 1k positions "
            f"{outer / total * 1e6 if total else float('nan'):.4f} | exposed wait {b.wait - a.wait:.6f} s | workers "
            f"{len(busy)} busy {','.join(f'{s:.3f}' for s, _ in busy) or '-'} s files "
            f"{','.join(str(f) for _, f in busy) or '-'}")


# ---------------------------------------------------------------- log parsing (the tests and the T-057 analysis)
_TS = r"^\[(?P<ts>[^]]+)\] "
STEP_TIME_RE = re.compile(_TS + r"phase-time step (?P<step>\d+) \| " + " \\| ".join(
    f"{p} (?P<{p}>-?\\d+\\.\\d+)" for p in PHASES + ("other",)) + r" \| sum (?P<sum>-?\d+\.\d+) \| unattributed "
    r"(?P<unattributed>-?\d+\.\d+) \| wall (?P<wall>-?\d+\.\d+)$")
WINDOW_START_RE = re.compile(_TS + r"measure-window start step (?P<step>\d+) \| warmup (?P<warmup>\d+) measure "
                             r"(?P<measure>\d+)$")
_HEAD_COUNTS = " ".join(f"{h} (?P<{{p}}{h}>\\d+)" for h in HEADS) + " total (?P<{p}total>\\d+)"
WINDOW_END_RE = re.compile(
    _TS + r"measure-window end step (?P<step>\d+) \| steps (?P<first>\d+)-(?P<last>\d+) \| outer (?P<outer>\d+\.\d+) s"
    r" \| rows " + _HEAD_COUNTS.format(p="r_") + r" \| admitted games (?P<games>\d+) " + _HEAD_COUNTS.format(p="a_")
    + r" \| ms per 1k positions (?P<ms>\S+) \| exposed wait (?P<wait>-?\d+\.\d+) s \| workers (?P<workers>\d+) busy "
    r"(?P<busy>\S+) s files (?P<files>\S+)$")
WINDOW_MEMORY_RE = re.compile(_TS + r"measure-window memory (?P<edge>start|end) step (?P<step>\d+) \| (?P<body>.*)$")
WINDOW_TIMED_RE = re.compile(_TS + r"measure-window (?P<label>eval|save) step (?P<step>\d+) \| (?P<s>\d+\.\d+) s$")
TRAIN_STEP_RE = re.compile(_TS + r"step (?P<step>\d+) \| (?!val|positions).* \| read (?P<read>[0-9a-f]{12}) \| sample "
                           r"(?P<sample>[0-9a-f]{12})$")
MPS_MEMORY_RE = re.compile(_TS + r"mps memory at step (?P<step>\d+)(?P<when> \([^)]*\))?: driver (?P<driver>\d+) MB")


def parse_run(lines) -> dict:
    """{"steps": {k: {phase: s, "sum", "unattributed", "wall"}}, "start": {...} | None, "end": {...} | None,
    "memory": {edge: {"ts", "step", "mb": {counter: MB}}}, "timed": {label: s}, "train": {k: (ts, read, sample)} (the
    last line of each step), "mps": [(ts, step, when, driver MB)]}; raises on a phase-time or measure-window line that
    matches no format."""
    out = {"steps": {}, "start": None, "end": None, "memory": {}, "timed": {}, "train": {}, "mps": []}
    for line in lines:
        line = line.rstrip("\n")
        if "phase-time " in line:
            m = STEP_TIME_RE.match(line)
            if not m:
                raise ValueError(f"unparsed phase-time line: {line!r}")
            out["steps"][int(m["step"])] = {k: float(m[k]) for k in PHASES + ("other", "sum", "unattributed", "wall")}
        elif "measure-window " in line:
            if m := WINDOW_START_RE.match(line):
                out["start"] = {"ts": m["ts"], "step": int(m["step"]), "warmup": int(m["warmup"]),
                                "measure": int(m["measure"])}
            elif m := WINDOW_END_RE.match(line):
                out["end"] = {
                    "ts": m["ts"], "step": int(m["step"]), "first": int(m["first"]), "last": int(m["last"]),
                    "outer": float(m["outer"]), "rows": {k: int(m[f"r_{k}"]) for k in HEADS + ("total",)},
                    "games": int(m["games"]), "admitted": {k: int(m[f"a_{k}"]) for k in HEADS + ("total",)},
                    "ms_per_1k": float(m["ms"]), "wait": float(m["wait"]), "workers": int(m["workers"]),
                    "busy": [] if m["busy"] == "-" else [float(v) for v in m["busy"].split(",")],
                    "files": [] if m["files"] == "-" else [int(v) for v in m["files"].split(",")]}
            elif m := WINDOW_MEMORY_RE.match(line):
                parts = [] if m["body"] == "none" else m["body"].split(" | ")
                mb = {k: float(v) for k, v in (re.fullmatch(r"(\S+) (\d+) MB", p).groups() for p in parts)}
                out["memory"][m["edge"]] = {"ts": m["ts"], "step": int(m["step"]), "mb": mb}
            elif m := WINDOW_TIMED_RE.match(line):
                out["timed"][m["label"]] = float(m["s"])
            else:
                raise ValueError(f"unparsed measure-window line: {line!r}")
        elif m := TRAIN_STEP_RE.match(line):
            out["train"][int(m["step"])] = (m["ts"], m["read"], m["sample"])
        elif m := MPS_MEMORY_RE.match(line):
            out["mps"].append((m["ts"], int(m["step"]), (m["when"] or "").strip(), int(m["driver"])))
    return out


# ---------------------------------------------------------------- checks and gates (plan T-057 "Checks and gates")
C1_TOL = 0.05
C2_RANGE = (0.9, 1.5)  # B / A outside it: B's attribution is "distorted by sync"
D1_RATIO = 0.8  # MPS A time <= 0.8 x CPU A time at the same size: MPS is worth pursuing
D1_REPEAT_BAND = (0.7, 0.9)  # a single-run ratio inside it: repeat both A runs once, decide on the means
GAMES_PER_EPOCH = 225_176
PRODUCTION_EVERY = 500  # eval and save cadence of a production run


class CheckError(AssertionError):
    pass


def window_steps(parsed: dict) -> range:
    end = parsed["end"]
    if end is None:
        raise CheckError("no measure-window end line")
    return range(end["first"], end["last"] + 1)


def window_hashes(parsed: dict) -> tuple:
    """(read hash, sample hash) of the window's steps: 16 hex digits of sha256 over their per-step read hashes and
    sample digests in step order. Raises when a window step has no step line."""
    steps = window_steps(parsed)
    missing = [k for k in steps if k not in parsed["train"]]
    if missing:
        raise CheckError(f"window steps without a step line: {missing[:10]} ({len(missing)})")
    read = hashlib.sha256("\n".join(parsed["train"][k][1] for k in steps).encode()).hexdigest()[:16]
    sample = hashlib.sha256("\n".join(parsed["train"][k][2] for k in steps).encode()).hexdigest()[:16]
    return read, sample


def phase_totals(parsed: dict) -> dict:
    """{phase: seconds summed over the window's steps, "sum", "other", "unattributed", "wall"}; raises when a window
    step has no phase-time line."""
    steps = window_steps(parsed)
    missing = [k for k in steps if k not in parsed["steps"]]
    if missing:
        raise CheckError(f"window steps without a phase-time line: {missing[:10]} ({len(missing)})")
    return {k: sum(parsed["steps"][s][k] for s in steps) for k in PHASES + ("other", "sum", "unattributed", "wall")}


def check_c1(phase_sum: float, outer: float, tol: float = C1_TOL) -> float:
    """C1 accounting: the B run's phase sum over the window within tol of its independent outer time. Returns the
    ratio; raises with the numbers."""
    if not (math.isfinite(outer) and outer > 0 and math.isfinite(phase_sum)):
        raise CheckError(f"C1: outer time {outer} must be finite and positive, phase sum {phase_sum} finite")
    ratio = phase_sum / outer
    if abs(ratio - 1) > tol:
        raise CheckError(f"C1: phase sum {phase_sum:.6f} s vs outer {outer:.6f} s, ratio {ratio:.4f} outside "
                         f"1 +- {tol}: a span is not attributed (or attributed twice)")
    return ratio


def c2_label(b_outer: float, a_outer: float) -> tuple:
    """C2 sync cost: (B / A, label); the label is "distorted by sync" outside C2_RANGE, else "ok"."""
    if not all(math.isfinite(t) and t > 0 for t in (a_outer, b_outer)):
        raise CheckError(f"C2: outer times A {a_outer}, B {b_outer} must be finite and positive")
    ratio = b_outer / a_outer
    return ratio, ("distorted by sync" if ratio > C2_RANGE[1] or ratio < C2_RANGE[0] else "ok")


def d1_size(cpu_ms: list, mps_ms: list) -> tuple:
    """D1 at one batch size from the A runs' ms per 1k positions, [A, A2 (the repeat, if run)] per device: (ratio,
    status). The original ratio (A runs) inside D1_REPEAT_BAND needs both repeats: "repeat" (none yet), "repeat
    pending" (one device's only), then the ratio of the means decides. Outside the band the original ratio decides.
    status: "pass" (ratio <= D1_RATIO) or "fail"."""
    if (not cpu_ms or not mps_ms or not all(math.isfinite(t) and t > 0 for t in cpu_ms + mps_ms)
            or max(len(cpu_ms), len(mps_ms)) > 2):
        raise CheckError(f"D1: needs one or two finite positive A times per device (cpu {cpu_ms}, mps {mps_ms})")
    ratio = mps_ms[0] / cpu_ms[0]
    if D1_REPEAT_BAND[0] <= ratio <= D1_REPEAT_BAND[1]:
        repeats = (len(cpu_ms) == 2) + (len(mps_ms) == 2)
        if repeats < 2:
            return ratio, "repeat" if repeats == 0 else "repeat pending"
        ratio = (sum(mps_ms) / 2) / (sum(cpu_ms) / 2)
    return ratio, "pass" if ratio <= D1_RATIO else "fail"


def d1_verdict(statuses: dict) -> str:
    """{size: status} -> "pass" (some size passes: MPS is worth pursuing), "repeat" (none passes, some size needs
    its repeats or waits for one), "fail"."""
    values = set(statuses.values())
    return "pass" if "pass" in values else "repeat" if values & {"repeat", "repeat pending"} else "fail"


def epoch_projection(ms_per_1k: float, admitted_rows: int, games: int, rows_per_step: float, eval_s: float,
                     save_s: float, games_per_epoch: int = GAMES_PER_EPOCH, every: int = PRODUCTION_EVERY) -> dict:
    """Plan T-057: rows per epoch = admitted rows per admitted game x games_per_epoch; train hours = ms per 1k
    positions x rows per epoch; plus (eval + save) at every `every` steps of the epoch."""
    if not (games > 0 and rows_per_step > 0 and ms_per_1k > 0):
        raise CheckError(f"epoch projection: games {games}, rows per step {rows_per_step}, ms per 1k {ms_per_1k}")
    rows = admitted_rows / games * games_per_epoch
    steps = rows / rows_per_step
    events = math.floor(steps / every)
    train_h = ms_per_1k * rows / 1e6 / 3600
    eval_save_h = events * (eval_s + save_s) / 3600
    return {"rows_per_epoch": rows, "steps_per_epoch": steps, "eval_save_events": events, "train_h": train_h,
            "eval_save_h": eval_save_h, "total_h": train_h + eval_save_h}
