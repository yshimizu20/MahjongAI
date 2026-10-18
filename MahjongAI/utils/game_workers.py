"""Parse and tensor-build training games ahead of the trainer in spawn worker processes (plan T-011 D1).

The DataLoader / DataLoaderCluster stay the only authority on the file order, the batch composition, skips,
`pending`, UnexpectedSkip and the checkpoint state: a loader given `workers` asks them for the same file it would
parse (`fetch`) and submits its next files ahead (`prefetch`). A worker runs `process(file)` and then `build(halfturns,
tokens)` (encoder_conv.build_game). The two error phases stay apart:
  - a parse error travels back as the exception itself (or, if it does not pickle, as a RemoteError record that keeps
    its type name, text and `msg`) and is raised by `Fetched.game()` inside the loader's skip handler, so the skip
    key, the log line, UnexpectedSkip and `pending` are the serial path's;
  - a build error travels inside the built game and is raised by the trainer's prepare_batches, where the serial path
    builds the game (after the loader accepted the batch).
A dead, hung or broken worker is not a skip: `fetch` raises DataWorkerError naming the file, after terminating and
joining every worker. Every wait is bounded: each worker sends its results on its own pipe (the parent holds only the
read end, so a dead writer gives EOF), as length-prefixed frames; a frame cut by a worker's death or stall is an EOF,
a dead process or the deadline, never a blocked read. A parent drain thread per pool (T-047 D-B) reads every worker's
pipe continuously into the in-memory `done` buffer, so a worker never waits for the trainer's next fetch to finish a
send (a macOS pipe holds 64 KB; a game's frame is 3-4 MB). The buffer is bounded: the thread stops reading while it
holds `window` whole frames and the trainer is not waiting, and reads on while a fetch waits (the awaited frame can be
behind a held one). The thread only records a broken pipe or frame; the next fetch raises it. Workers start on the
first prefetch / fetch, ignore SIGINT (the trainer owns the stop), run one torch thread each, and exit by themselves
when the trainer process is gone (kill -9).
"""
import os
import pickle
import signal
import struct
import threading
import time

import torch.multiprocessing as mp
from multiprocessing.connection import wait

WORKER_TIMEOUT_S = 120.0  # the longest wait for one game (a game takes ~0.25 s; the first wait includes worker start)
POLL_S = 0.2  # liveness check interval while waiting
_HEADER = struct.Struct("!Q")  # a result frame: body length, then the body (a pickled (filename, pickled payload))
_READ_CHUNK = 1 << 20


class DataWorkerError(RuntimeError):
    """A data worker died, hung or sent a broken result: fatal for the run (never a counted skip)."""


class RemoteError(Exception):
    """A worker's parse or build exception that does not survive pickling, as a pickleable record: the phase ("parse"
    or "build"), the original type's name, its text, its `msg` attribute (None if it had none) and whether it was an
    InvalidGameException. The loader takes the skip key from these, so the key and the log line are the serial path's.
    """

    def __init__(self, phase: str, type_name: str, message: str, msg=None, invalid_game: bool = False):
        super().__init__(phase, type_name, message, msg, invalid_game)
        self.phase, self.type_name, self.message, self.msg, self.invalid_game = (
            phase, type_name, message, msg, invalid_game)

    def __str__(self):
        return self.message


def _transportable(e: BaseException, phase: str) -> BaseException:
    """`e` if it pickles back to the same type and text (so the loader's skip key and log line are unchanged), else a
    RemoteError record of it."""
    try:
        back = pickle.loads(pickle.dumps(e))
        if type(back) is type(e) and str(back) == str(e) and getattr(back, "msg", None) == getattr(e, "msg", None):
            return e
    except Exception:
        pass
    from MahjongAI.process_xml import InvalidGameException
    return RemoteError(phase, type(e).__name__, str(e), getattr(e, "msg", None), isinstance(e, InvalidGameException))


def _watch_parent(parent_pid: int):
    while True:
        if os.getppid() != parent_pid:  # the trainer is gone (kill -9 runs no cleanup): exit, leave no stray
            os._exit(0)
        time.sleep(0.5)


def _send_frame(fd: int, body: bytes):
    """Write one length-prefixed frame to the worker's result pipe (blocking: the parent's drain thread reads it)."""
    view = memoryview(_HEADER.pack(len(body)) + body)
    while view:
        view = view[os.write(fd, view):]


def _worker_main(tasks, results, build, parent_pid: int, send=_send_frame, parse=None):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch
    torch.set_num_threads(1)  # the trainer's threads keep their cores
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()
    if parse is None:
        from MahjongAI.process_xml import process as parse
    while True:
        filename = tasks.get()
        if filename is None:
            return
        try:
            halfturns, tokens = parse(filename)
        except Exception as e:
            payload = ("parse_error", _transportable(e, "parse"))
        else:
            built = build(halfturns, tokens)
            if getattr(built, "build_error", None) is not None:
                built.build_error = _transportable(built.build_error, "build")
            payload = ("ok", built)
        try:
            data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            data = pickle.dumps(("broken", f"{type(e).__name__}: {e}"))
        send(results.fileno(), pickle.dumps((filename, data), protocol=pickle.HIGHEST_PROTOCOL))


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
    """`n` spawn worker processes shared by the loaders of one run. window: files a loader keeps submitted ahead.
    send, parse: test hooks, module-level functions: the worker's frame writer of (fd, body), and its parser of a
    file (default process_xml.process)."""

    def __init__(self, n: int, build, window: int = 8, timeout_s: float = WORKER_TIMEOUT_S, send=_send_frame,
                 parse=None):
        assert n >= 1 and window >= 1
        self.n, self.build, self.window, self.timeout_s = n, build, window, timeout_s
        self.send, self.parse = send, parse
        self.procs = []
        self.readers = []  # per worker: the read end of its result pipe (non-blocking)
        self.buffers = []  # per worker: bytes read and not yet a whole frame
        self.in_flight = set()  # submitted, result not received
        self.done = {}  # file -> pickled result, not fetched yet
        self.closed = False
        # the drain thread and the trainer share in_flight, done, buffers, _waiting and _error under _cond
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._drainer = None
        self._waiting = False  # a fetch waits: the thread reads past the bound
        self._error = None  # the first broken pipe or frame the thread saw, raised by the next fetch

    def _start(self):
        if self.procs:
            return
        if self.closed:
            raise DataWorkerError("data workers used after close()")
        ctx = mp.get_context("spawn")
        self.tasks = ctx.Queue()
        for _ in range(self.n):
            reader, writer = ctx.Pipe(duplex=False)
            args = (self.tasks, writer, self.build, os.getpid(), self.send, self.parse)
            p = ctx.Process(target=_worker_main, args=args, daemon=True)
            p.start()
            writer.close()  # the child holds the only write end: its death is an EOF here
            os.set_blocking(reader.fileno(), False)
            self.procs.append(p)
            self.readers.append(reader)
            self.buffers.append(bytearray())
        self._stop.clear()
        self._drainer = threading.Thread(target=self._drain_loop, name="game-workers-drain", daemon=True)
        self._drainer.start()

    def prefetch(self, files):
        """Submit the files not submitted yet, in order (the workers take them first come, first served)."""
        self._start()
        with self._cond:
            for f in files:
                if f not in self.in_flight and f not in self.done:
                    self.in_flight.add(f)
                    self.tasks.put(f)

    def _fail(self, message: str):
        self.close()
        raise DataWorkerError(message)

    def _drain_loop(self):
        """The drain thread: read every open pipe as data arrives; stop reading at the bound unless a fetch waits."""
        open_ = set(range(len(self.readers)))
        while not self._stop.is_set():
            with self._cond:
                while len(self.done) >= self.window and not self._waiting and not self._stop.is_set():
                    self._cond.wait(POLL_S)
            if self._stop.is_set():
                return
            if not open_:
                self._stop.wait(POLL_S)
                continue
            try:
                ready = wait([self.readers[i] for i in open_], timeout=POLL_S)
            except (OSError, ValueError):  # a reader closed under us: only close() does that
                return
            for i in sorted(open_):
                if self.readers[i] in ready and not self._read(i):
                    open_.discard(i)

    def _read(self, i: int) -> bool:
        """Read what worker i's pipe holds now (never blocks) and store each whole frame's result. False at EOF."""
        chunks, eof = [], False
        while True:
            try:
                chunk = os.read(self.readers[i].fileno(), _READ_CHUNK)
            except BlockingIOError:
                break
            except OSError as e:
                eof, error = True, f"data worker pid {self.procs[i].pid} result pipe read failed: {e}"
                break
            if not chunk:
                eof, error = True, None
                break
            chunks.append(chunk)
        with self._cond:
            buf = self.buffers[i]
            for chunk in chunks:
                buf += chunk
            while len(buf) >= _HEADER.size:
                (size,) = _HEADER.unpack_from(buf)
                if len(buf) < _HEADER.size + size:
                    break
                body = bytes(buf[_HEADER.size:_HEADER.size + size])
                del buf[:_HEADER.size + size]
                try:
                    f, data = pickle.loads(body)
                except Exception as e:
                    self._error = self._error or f"unreadable result frame: {type(e).__name__}: {e}"
                    break
                self.in_flight.discard(f)
                self.done[f] = data
            if eof:
                self._error = self._error or error or (
                    f"data worker pid {self.procs[i].pid} closed its result pipe ({len(buf)} byte(s) of an unfinished "
                    f"frame)")
            self._cond.notify_all()
        return not eof

    def _unfinished(self) -> str:
        """The workers holding part of a frame (call under _cond), for the timeout message."""
        parts = [f"data worker pid {p.pid} holds {len(b)} byte(s) of an unfinished frame"
                 for p, b in zip(self.procs, self.buffers) if b]
        return ("; " + "; ".join(parts)) if parts else ""

    def fetch(self, filename: str) -> Fetched:
        """The game of `filename` (submitted now if it was not). Raises DataWorkerError, every worker stopped, if a
        worker is dead, the result does not come within timeout_s, or it cannot be read."""
        self.prefetch([filename])
        deadline = time.monotonic() + self.timeout_s
        failure, data = None, None
        with self._cond:
            self._waiting = True
            self._cond.notify_all()
            try:
                while True:
                    dead = [p for p in self.procs if not p.is_alive()]
                    if dead:
                        failure = (f"data worker pid {dead[0].pid} died (exit code {dead[0].exitcode}) while fetching "
                                   f"{filename}; {len(self.in_flight)} file(s) in flight")
                        break
                    if filename in self.done:
                        self.in_flight.discard(filename)
                        data = self.done.pop(filename)
                        self._cond.notify_all()  # room under the bound again
                        break
                    if self._error is not None:
                        failure = f"{self._error} while fetching {filename}; {len(self.in_flight)} file(s) in flight"
                        break
                    left = deadline - time.monotonic()
                    if left <= 0:
                        failure = (f"no result for {filename} from the data workers within {self.timeout_s:.0f} s"
                                   + self._unfinished())
                        break
                    self._cond.wait(min(POLL_S, left))
            finally:
                self._waiting = False
        if failure is not None:
            self._fail(failure)  # outside _cond: close() joins the drain thread
        try:
            status, value = pickle.loads(data)
        except Exception as e:
            self._fail(f"unreadable result for {filename}: {type(e).__name__}: {e}")
        if status == "broken":
            self._fail(f"the data worker could not send its result for {filename}: {value}")
        return Fetched(status, value)

    def close(self):
        """Terminate and join every worker (idempotent). Called on every stop path of the trainer."""
        self.closed = True
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._drainer is not None:  # stopped before the readers close under it
            self._drainer.join(timeout=10)
            self._drainer = None
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        if self.procs:
            self.tasks.cancel_join_thread()
            self.tasks.close()
        for reader in self.readers:
            reader.close()
        self.procs, self.readers, self.buffers = [], [], []
        self.in_flight.clear()
        self.done.clear()
        self._error = None

    def pids(self) -> list:
        return [p.pid for p in self.procs]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
