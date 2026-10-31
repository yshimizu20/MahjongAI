"""Memory guard of a long MPS training run (plan T-068 "Run safety"): stop the trainer before swap takes the Mac.

  python -m MahjongAI.utils.mem_guard <trainer pid> <run dir>/train.log
  python -m MahjongAI.utils.mem_guard --test

Every EVERY_S seconds the guard reads the swap used (`sysctl vm.swapusage`) and the `mps memory ... driver <N> MB`
lines the trainer appended to train.log since the guard started (phase_timing.MPS_MEMORY_RE). It sends SIGINT to the
trainer (which saves a checkpoint and stops; a resume is the same command, a fresh process) ONCE, when
  swap    swap used > its value at the guard's start + SWAP_LIMIT_MB, or
  driver  the latest driver MB > the first post-eval driver MB + DRIVER_LIMIT_MB.
The driver baseline is the first `(post-eval/save)` line of a step > 0: the row-0 line of a fresh run comes before any
update (~8 MB against ~8.9 GB in training), so no baseline and no driver trigger exist before it. Start the guard
right after the trainer: lines logged before the guard starts are not read (train.log keeps every earlier process).
One line per check on stdout (the readings, the limits and the decision); the guard exits when the pid is gone. MB:
2**20 bytes, as the trainer's log and sysctl print them.
"""
import argparse
import os
import re
import signal
import subprocess
import time
from datetime import datetime

from MahjongAI.utils.phase_timing import MPS_MEMORY_RE

EVERY_S = 60
SWAP_LIMIT_MB = 1024  # 1 GB over the start value
DRIVER_LIMIT_MB = 1536  # 1.5 GB over the first post-eval value
POST_EVAL = "(post-eval/save)"
SWAP_RE = re.compile(r"used = (?P<value>\d+(?:\.\d+)?)(?P<unit>[MG])")


def parse_swap_used(text: str) -> float:
    """MB used of a `sysctl vm.swapusage` output ("vm.swapusage: total = 2048.00M  used = 1023.25M  free = ...")."""
    m = SWAP_RE.search(text)
    if not m:
        raise ValueError(f"no swap used in {text!r}")
    return float(m["value"]) * (1024 if m["unit"] == "G" else 1)


def read_swap_used() -> float:
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, check=True).stdout
    return parse_swap_used(out)


def driver_readings(lines) -> list:
    """[(step, when, driver MB)] of the mps memory lines among `lines`, in order."""
    out = []
    for line in lines:
        if m := MPS_MEMORY_RE.match(line.rstrip("\n")):
            out.append((int(m["step"]), (m["when"] or "").strip(), int(m["driver"])))
    return out


class LogTail:
    """The complete lines appended to a file since the tail was made (a partial last line waits for its newline)."""

    def __init__(self, path: str):
        self.path = path
        self.offset = os.path.getsize(path) if os.path.exists(path) else 0

    def __call__(self) -> list:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        end = data.rfind(b"\n") + 1
        self.offset += end
        return data[:end].decode(errors="replace").splitlines()


class MemGuard:
    """The guard's state: swap at start, the driver baseline and latest reading, whether it has fired."""

    def __init__(self, swap_start: float, swap_limit: float = SWAP_LIMIT_MB, driver_limit: float = DRIVER_LIMIT_MB):
        self.swap_start, self.swap_limit, self.driver_limit = swap_start, swap_limit, driver_limit
        self.driver_base = self.driver_now = None
        self.fired = None  # the reason of the one SIGINT, once sent

    def observe(self, swap_used: float, new_lines) -> tuple:
        """(status line, fire): fire is True only on the first check where a limit is crossed."""
        for step, when, mb in driver_readings(new_lines):
            if self.driver_base is None and when == POST_EVAL and step > 0:
                self.driver_base = mb
            self.driver_now = mb
        swap_up = swap_used - self.swap_start
        reasons = []
        if swap_up > self.swap_limit:
            reasons.append(f"swap +{swap_up:.0f} MB > {self.swap_limit:.0f}")
        if self.driver_base is not None and self.driver_now - self.driver_base > self.driver_limit:
            reasons.append(f"driver +{self.driver_now - self.driver_base} MB > {self.driver_limit:.0f}")
        fire = bool(reasons) and self.fired is None
        if fire:
            self.fired = "; ".join(reasons)
        base = "-" if self.driver_base is None else f"{self.driver_base}, {self.driver_now - self.driver_base:+d}"
        driver = ("driver -" if self.driver_now is None
                  else f"driver {self.driver_now} MB (post-eval base {base} / limit +{self.driver_limit:.0f})")
        decision = (f"SIGINT ({self.fired})" if fire else f"already sent SIGINT ({self.fired}): waiting for the exit"
                    if self.fired else "ok")
        return (f"swap used {swap_used:.0f} MB (start {self.swap_start:.0f}, {swap_up:+.0f} / limit "
                f"+{self.swap_limit:.0f}) | {driver} | {decision}"), fire


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # alive, another user's
        return True
    return True


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run(pid: int, tail, read_swap=read_swap_used, alive=pid_alive, send=os.kill, sleep=time.sleep, now=now_text,
        out=print, every: float = EVERY_S) -> MemGuard:
    """The guard loop: a check right away, then every `every` s, until the pid is gone. Every reader, the signal, the
    clock and the sleep are arguments (the self-test drives them)."""
    guard = MemGuard(read_swap())
    out(f"[{now()}] mem_guard: pid {pid}, swap used at start {guard.swap_start:.0f} MB, limits swap "
        f"+{guard.swap_limit:.0f} MB, driver +{guard.driver_limit:.0f} MB over the first post-eval value; every "
        f"{every:g} s", flush=True)
    while alive(pid):
        line, fire = guard.observe(read_swap(), tail())
        if fire:
            send(pid, signal.SIGINT)
        out(f"[{now()}] {line}", flush=True)
        sleep(every)
    out(f"[{now()}] pid {pid} is gone: exit" + (f" (SIGINT sent: {guard.fired})" if guard.fired else ""), flush=True)
    return guard


# ----------------------------------------------------------------------------------------------- self-test


def _mps(step: int, mb: int, when: str = "") -> str:
    return f"[2026-10-30 12:00:00.000] mps memory at step {step}{when}: driver {mb} MB, current 21 MB"


def _drive(swaps: list, logs: list) -> tuple:
    """Run the loop on fake readings: swaps[0] at start, then one swap reading and one batch of new log lines per
    check; the pid exits after the last check. Returns (signals sent, status lines, the guard)."""
    swap_iter, log_iter = iter(swaps), iter(logs)
    checks = [0]
    sent, lines = [], []

    def alive(pid):
        return checks[0] < len(logs)

    def tail():
        checks[0] += 1
        return next(log_iter)

    guard = run(7, tail, read_swap=lambda: next(swap_iter), alive=alive, send=lambda p, s: sent.append((p, s)),
                sleep=lambda s: None, now=lambda: "t", out=lambda line, flush=False: lines.append(line))
    assert len(lines) == len(logs) + 2, lines
    return sent, lines, guard


def self_test():
    """No signal when flat (and at the limits, and on a row-0 line far below training); exactly one SIGINT for each
    trigger, however long it stays crossed; swap output and log tails parse."""
    assert parse_swap_used("vm.swapusage: total = 10240.00M  used = 9602.19M  free = 637.81M  (encrypted)") == 9602.19
    assert parse_swap_used("vm.swapusage: total = 2.00G  used = 1.50G  free = 0.50G") == 1536.0
    training = [[_mps(0, 8, " (post-eval/save)")], [_mps(100, 8952)], [_mps(5000, 8862, " (post-eval/save)")],
                [], [_mps(5100, 8964), "[t] step 5100 | loss"]]
    sent, lines, guard = _drive([9000.0] * 6, training)
    assert not sent and guard.driver_base == 8862 and guard.fired is None, (sent, lines)
    assert all(line.endswith("| ok") for line in lines[1:-1]), lines
    print("  flat: no signal (row-0 8 MB line is no baseline; driver +102 MB)")

    at_limits = training + [[_mps(5200, 8862 + DRIVER_LIMIT_MB)]]
    sent, _, _ = _drive([9000.0] * 6 + [9000.0 + SWAP_LIMIT_MB], at_limits)
    assert not sent
    print("  at the limits exactly: no signal")

    sent, lines, guard = _drive([9000.0] * 4 + [10100.0, 10200.0, 9000.0], training + [[]])
    assert sent == [(7, signal.SIGINT)] and "SIGINT (swap +1100 MB > 1024)" in lines[4], lines
    assert "already sent SIGINT" in lines[5] and "already sent SIGINT" in lines[6] and "SIGINT sent" in lines[-1]
    print("  swap trigger: one SIGINT, none after")

    driver_up = training + [[_mps(5200, 8862 + DRIVER_LIMIT_MB + 1)], [_mps(5300, 11000)]]
    sent, lines, _ = _drive([9000.0] * 8, driver_up)
    assert sent == [(7, signal.SIGINT)] and f"driver +{DRIVER_LIMIT_MB + 1} MB > {DRIVER_LIMIT_MB}" in lines[6], lines
    print("  driver trigger: one SIGINT, none after")

    no_base = [[_mps(0, 8, " (post-eval/save)")], [_mps(100, 8952)], [_mps(200, 12000)]]  # before any post-eval line
    sent, _, guard = _drive([9000.0] * 4, no_base)
    assert not sent and guard.driver_base is None
    print("  no post-eval baseline yet: no driver trigger")
    print("mem_guard self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pid", nargs="?", type=int, help="the trainer's pid")
    ap.add_argument("log", nargs="?", help="the run's train.log")
    ap.add_argument("--every", type=float, default=EVERY_S, help=f"seconds between checks (default {EVERY_S})")
    ap.add_argument("--test", action="store_true", help="run the self-test and exit")
    args = ap.parse_args()
    if args.test:
        self_test()
        raise SystemExit(0)
    assert args.pid and args.log, "give the trainer's pid and its train.log"
    run(args.pid, LogTail(args.log), every=args.every)
