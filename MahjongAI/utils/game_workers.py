"""Parse and tensor-build training games ahead of the trainer in spawn worker processes (plan T-011 D1).

The DataLoader / DataLoaderCluster stay the only authority on the file order, the batch composition, skips,
`pending`, UnexpectedSkip and the checkpoint state: a loader given `workers` asks them for the same file it would
parse (`fetch`) and submits its next files ahead (`prefetch`). A worker runs `process(file)` and then `build(halfturns,
tokens)` (encoder_conv.build_game). The two error phases stay apart:
  - a parse error travels back as the exception itself and is raised by `Fetched.game()` inside the loader's skip
    handler, so the skip key, the log line, UnexpectedSkip and `pending` are the serial path's;
  - a build error travels inside the built game and is raised by the trainer's prepare_batches, where the serial path
    builds the game (after the loader accepted the batch).
A dead, hung or broken worker is not a skip: `fetch` raises DataWorkerError naming the file, after terminating and
joining every worker. Every wait is bounded. Workers start on the first prefetch / fetch, ignore SIGINT (the trainer
owns the stop), run one torch thread each, and exit by themselves when the trainer process is gone (kill -9).
"""
import os
import pickle
import queue
import signal
import threading
import time

import torch.multiprocessing as mp

WORKER_TIMEOUT_S = 120.0  # the longest wait for one game (a game takes ~0.25 s; the first wait includes worker start)
POLL_S = 0.2  # liveness check interval while waiting


class DataWorkerError(RuntimeError):
    """A data worker died, hung or sent a broken result: fatal for the run (never a counted skip)."""


class _RemoteError(Exception):
    """Stand-in for an exception that does not survive pickling; renamed to the original type's name below."""


def _transportable(e: BaseException) -> BaseException:
    """`e` if it pickles back to the same type and text (so the loader's skip key and log line are unchanged), else an
    exception of a class with the same name, the same text and the same `msg` attribute when there is one."""
    try:
        back = pickle.loads(pickle.dumps(e))
        if type(back) is type(e) and str(back) == str(e) and getattr(back, "msg", None) == getattr(e, "msg", None):
            return e
    except Exception:
        pass
    stand_in = type(type(e).__name__, (_RemoteError,), {})(str(e))
    if hasattr(e, "msg"):
        stand_in.msg = e.msg
    return stand_in


def _watch_parent(parent_pid: int):
    while True:
        if os.getppid() != parent_pid:  # the trainer is gone (kill -9 runs no cleanup): exit, leave no stray
            os._exit(0)
        time.sleep(0.5)


def _worker_main(tasks, results, build, parent_pid: int):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch
    torch.set_num_threads(1)  # the trainer's threads keep their cores
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()
    from MahjongAI.process_xml import process
    while True:
        filename = tasks.get()
        if filename is None:
            return
        try:
            halfturns, tokens = process(filename)
        except Exception as e:
            payload = ("parse_error", _transportable(e))
        else:
            built = build(halfturns, tokens)
            if getattr(built, "build_error", None) is not None:
                built.build_error = _transportable(built.build_error)
            payload = ("ok", built)
        try:
            data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            data = pickle.dumps(("broken", f"{type(e).__name__}: {e}"))
        results.put((filename, data))


class Fetched:
    """One fetched game: `game()` returns (built game, None) for the loader's (halfturns, tokens) slots, or raises
    the game's parse error."""

    def __init__(self, status: str, value):
        self.status, self.value = status, value

    def game(self):
        if self.status == "parse_error":
            raise self.value
        return self.value, None


class GameWorkers:
    """`n` spawn worker processes shared by the loaders of one run. window: files a loader keeps submitted ahead."""

    def __init__(self, n: int, build, window: int = 8, timeout_s: float = WORKER_TIMEOUT_S):
        assert n >= 1 and window >= 1
        self.n, self.build, self.window, self.timeout_s = n, build, window, timeout_s
        self.procs = []
        self.in_flight = set()  # submitted, result not received
        self.done = {}  # file -> pickled result, not fetched yet
        self.closed = False

    def _start(self):
        if self.procs:
            return
        if self.closed:
            raise DataWorkerError("data workers used after close()")
        ctx = mp.get_context("spawn")
        self.tasks, self.results = ctx.Queue(), ctx.Queue()
        for _ in range(self.n):
            p = ctx.Process(target=_worker_main, args=(self.tasks, self.results, self.build, os.getpid()), daemon=True)
            p.start()
            self.procs.append(p)

    def prefetch(self, files):
        """Submit the files not submitted yet, in order (the workers take them first come, first served)."""
        self._start()
        for f in files:
            if f not in self.in_flight and f not in self.done:
                self.in_flight.add(f)
                self.tasks.put(f)

    def _fail(self, message: str):
        self.close()
        raise DataWorkerError(message)

    def fetch(self, filename: str) -> Fetched:
        """The game of `filename` (submitted now if it was not). Raises DataWorkerError, every worker stopped, if a
        worker is dead, the result does not come within timeout_s, or it cannot be read."""
        self.prefetch([filename])
        deadline = time.monotonic() + self.timeout_s
        while True:
            dead = [p for p in self.procs if not p.is_alive()]
            if dead:
                self._fail(f"data worker pid {dead[0].pid} died (exit code {dead[0].exitcode}) while fetching "
                           f"{filename}; {len(self.in_flight)} file(s) in flight")
            if filename in self.done:
                break
            try:
                f, data = self.results.get(timeout=min(POLL_S, max(0.0, deadline - time.monotonic())))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    self._fail(f"no result for {filename} from the data workers within {self.timeout_s:.0f} s")
                continue
            self.in_flight.discard(f)
            self.done[f] = data
        self.in_flight.discard(filename)
        try:
            status, value = pickle.loads(self.done.pop(filename))
        except Exception as e:
            self._fail(f"unreadable result for {filename}: {type(e).__name__}: {e}")
        if status == "broken":
            self._fail(f"the data worker could not send its result for {filename}: {value}")
        return Fetched(status, value)

    def close(self):
        """Terminate and join every worker (idempotent). Called on every stop path of the trainer."""
        self.closed = True
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        if self.procs:
            for q in (self.tasks, self.results):
                q.cancel_join_thread()
                q.close()
        self.procs = []
        self.in_flight.clear()
        self.done.clear()

    def pids(self) -> list:
        return [p.pid for p in self.procs]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
