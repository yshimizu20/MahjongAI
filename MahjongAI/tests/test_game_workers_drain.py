"""T-047 D-B: the parent drain thread reads every worker's result pipe continuously, so a worker never blocks on a
send until the trainer's next fetch; a live worker that leaves a frame unfinished is a bounded fatal error that names
it; every process and the drain thread are gone after it. (A worker's death mid-frame stays fatal:
test_game_workers.test_worker_death_mid_frame_is_fatal_and_leaves_no_process.)"""
import multiprocessing
import os
import pathlib
import struct
import threading
import time

import pytest

from MahjongAI.utils.game_workers import DataWorkerError, GameWorkers, _send_frame

FRAME_BYTES = 4 << 20  # a frame of >= 4 MB: 64 pipe-fulls of a macOS pipe


def _payload(filename: str) -> bytes:
    """A deterministic, file-specific 4 MB payload."""
    seed = filename.encode()
    return (seed * (FRAME_BYTES // len(seed) + 1))[:FRAME_BYTES]


def _parse_name(filename):
    return filename, None


def _build_big(halfturns, tokens):
    return _payload(halfturns)


def _send_and_mark(fd, body):
    """The worker's frame writer, then a marker file once the whole frame is written (the send returned)."""
    _send_frame(fd, body)
    marker_dir = pathlib.Path(os.environ["T047_MARKER_DIR"])
    (marker_dir / f"sent-{time.monotonic_ns()}").touch()


def _send_part_of_header(fd, body):
    os.write(fd, struct.pack("!Q", len(body))[:3])
    time.sleep(120)


def _send_part_of_body(fd, body):
    os.write(fd, struct.pack("!Q", len(body)) + body[:len(body) // 2])
    time.sleep(120)


def _gone(pids):
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        return False
    return True


def _drain_threads():
    return [t for t in threading.enumerate() if t.name == "game-workers-drain"]


def test_worker_finishes_two_large_frames_while_the_trainer_does_not_fetch(tmp_path, monkeypatch):
    """One worker, two 4 MB results, no fetch: both sends complete (the drain thread reads them), and the fetched
    payloads are byte-equal to what the worker built."""
    monkeypatch.setenv("T047_MARKER_DIR", str(tmp_path))
    files = ["game-a.log", "game-b.log"]
    with GameWorkers(1, _build_big, window=4, timeout_s=60, send=_send_and_mark, parse=_parse_name) as w:
        w.prefetch(files)
        deadline = time.monotonic() + 60
        while len(list(tmp_path.glob("sent-*"))) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        sent = len(list(tmp_path.glob("sent-*")))
        assert sent == 2, f"the worker finished {sent} of 2 large frames without a fetch: its send blocks"
        assert set(w.done) == set(files) and not w.in_flight
        for f in files:
            built, tokens = w.fetch(f).game()
            assert tokens is None and built == _payload(f)


@pytest.mark.parametrize("send", [_send_part_of_header, _send_part_of_body], ids=["header", "body"])
def test_live_worker_with_an_unfinished_frame_is_a_bounded_fatal_error(send):
    """A worker writes part of a frame (header or body) and stays alive: the fetch raises DataWorkerError within the
    deadline, naming the worker's pid and the unfinished frame; every worker and the drain thread are stopped."""
    w = GameWorkers(1, _build_big, window=4, timeout_s=4, send=send, parse=_parse_name)
    w.prefetch([])
    pids = w.pids()
    assert len(pids) == 1 and len(_drain_threads()) == 1
    t = time.monotonic()
    with pytest.raises(DataWorkerError) as error:
        w.fetch("game-a.log")
    seconds = time.monotonic() - t
    message = str(error.value)
    assert "no result for game-a.log from the data workers within 4 s" in message
    assert f"data worker pid {pids[0]} holds " in message and "byte(s) of an unfinished frame" in message, message
    assert seconds < 20 and w.pids() == [] and w.closed and w._drainer is None
    assert _gone(pids) and not multiprocessing.active_children() and not _drain_threads()


def test_drain_thread_start_failure_still_stops_every_worker(monkeypatch):
    """Thread.start() of the drain thread raises after the workers started: close() does not join the never-started
    thread, and every worker process is stopped."""
    start = threading.Thread.start

    def failing_start(self):
        if self.name == "game-workers-drain":
            raise RuntimeError("can't start new thread")
        start(self)

    monkeypatch.setattr(threading.Thread, "start", failing_start)
    w = GameWorkers(2, _build_big, window=4, timeout_s=10, parse=_parse_name)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        w.prefetch(["game-a.log"])
    procs = list(w.procs)
    assert len(procs) == 2 and all(p.is_alive() for p in procs)
    try:
        w.close()
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join(5)
    assert not any(p.is_alive() for p in procs) and w.procs == [] and w._drainer is None
    assert not multiprocessing.active_children() and not _drain_threads()


def test_interrupt_right_after_a_worker_start_still_stops_it(monkeypatch):
    """A KeyboardInterrupt arrives right after the second worker's start() returned (before the rest of _start ran):
    that worker is already registered, so close() stops it with the first."""
    ctx = multiprocessing.get_context("spawn")
    start, started = ctx.Process.start, []

    def interrupted_start(self):
        start(self)
        started.append(self)
        if len(started) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(ctx.Process, "start", interrupted_start)
    w = GameWorkers(3, _build_big, window=4, timeout_s=10, parse=_parse_name)
    with pytest.raises(KeyboardInterrupt):
        w.prefetch(["game-a.log"])
    assert len(started) == 2 and all(p.is_alive() for p in started)
    try:
        w.close()
    finally:
        left = [p for p in started if p.is_alive()]
        for p in left:
            p.kill()
            p.join(5)
    assert not left, f"{len(left)} started worker(s) left alive by close()"
    assert w.procs == [] and not multiprocessing.active_children() and not _drain_threads()
