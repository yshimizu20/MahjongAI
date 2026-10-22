"""Self-play games started from the viewer page (T-053 phase 2; the chess dashboard's job pattern).

A POST queues a game; one worker thread plays the queue in order (engine/game.py Game, the match.py players) and
saves every finished game as `<selfplay-dir>/<id>/` (game.log + decisions.json + meta.json), so the game picker
lists it and /api/game exports it after a server restart. While a game plays, its Game observer keeps a copy of the
round in play; a poll exports the rounds so far with the viewer's own `export_round` (export.export_live_round): the
finished rounds whole, the round in play up to its last settled decision point. Every step a live export
produced is fingerprinted; a later export that changes it, or a saved-log export (export.export_game) at the end
that differs from the live rounds, ends the job in an error: what the page followed is what it steps through after.
A failed job leaves `<id>/error.json` {status, error, cfg}: the listing skips it, a restarted server reports its error.

Seat specs: `model` (the server's --ckpt), `model:<checkpoint>` (under the --ckpt directory or a --ckpt-root, like
/api/game's roots [C8]), `greedy`, `uniform`.
"""
import glob
import hashlib
import json
import math
import os
import queue
import re
import threading
import time
import traceback
from typing import List, Optional

from MahjongAI.engine.game import Game
from MahjongAI.engine.players import make_player
from MahjongAI.viewer import export

SEAT_KINDS = ("model", "greedy", "uniform")
MAX_SEED = 2 ** 31 - 1
MAX_TEMPERATURE = 10.0
ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]{1,64}$")
ERROR_FILE = "error.json"


class BadRequest(ValueError):
    """A self-play request the server answers with a 400 and this message."""


def parse_request(body, default_ckpt: Optional[str], ckpt_roots: tuple = ()) -> dict:
    """{seats: four specs, seed: int (default 0), temperature: number (default 0)} -> the job config with the
    make_player specs; the first bad field raises BadRequest naming it. ckpt_roots: the real paths a `model:<path>`
    checkpoint must lie under."""
    if not isinstance(body, dict):
        raise BadRequest("the body must be a JSON object")
    seats = body.get("seats")
    if not isinstance(seats, list) or len(seats) != 4 or not all(isinstance(s, str) for s in seats):
        raise BadRequest("seats must be a list of 4 strings")
    specs = [_player_spec(s.strip(), i, default_ckpt, ckpt_roots) for i, s in enumerate(seats)]
    seed = body.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise BadRequest(f"seed must be an integer in [0, {MAX_SEED}], got {seed!r}")
    temperature = body.get("temperature", 0.0)
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature)
            or not 0.0 <= temperature <= MAX_TEMPERATURE):
        raise BadRequest(f"temperature must be a number in [0, {MAX_TEMPERATURE}], got {temperature!r}")
    return {"specs": specs, "seed": seed, "temperature": float(temperature)}


def _player_spec(seat: str, i: int, default_ckpt: Optional[str], ckpt_roots: tuple) -> str:
    kind, _, ckpt = seat.partition(":")
    if kind not in SEAT_KINDS or (kind != "model" and ckpt):
        raise BadRequest(f"seat {i}: unknown player {seat!r} (model, model:<checkpoint>, greedy, uniform)")
    if kind != "model":
        return kind
    ckpt = ckpt or default_ckpt
    if not ckpt:
        raise BadRequest(f"seat {i}: 'model' needs a checkpoint: model:<path>, or start the server with --ckpt")
    real = os.path.realpath(ckpt)
    if not any(real.startswith(r + os.sep) for r in ckpt_roots):  # before the existence check: no probing outside
        raise BadRequest(f"seat {i}: checkpoint {ckpt} is not under the checkpoint roots (the --ckpt directory, "
                         f"--ckpt-root)")
    if not os.path.isfile(real):
        raise BadRequest(f"seat {i}: checkpoint {ckpt} not found")
    return f"model:{real}"


def seat_label(spec: str) -> str:
    """A make_player spec as the page shows it before the player is built: a checkpoint by its file name."""
    kind, _, ckpt = spec.partition(":")
    return f"{kind}:{os.path.basename(ckpt)}" if ckpt else kind


def flat_steps(rounds: list) -> list:
    """[(round number, step)] over the rounds, in order: the global decision-point index k."""
    return [(ri, s) for ri, r in enumerate(rounds) for s in r["steps"]]


def first_difference(live: list, final: list) -> Optional[str]:
    """None when the live rounds equal the final export's rounds, else where they first differ."""
    if len(live) != len(final):
        return f"{len(live)} live rounds vs {len(final)} saved"
    for ri, (a, b) in enumerate(zip(live, final)):
        for key in ("index", "honba", "oya", "results"):
            if a[key] != b[key]:
                return f"round {ri} {key}: {a[key]!r} vs {b[key]!r}"
        if len(a["steps"]) != len(b["steps"]):
            return f"round {ri}: {len(a['steps'])} live steps vs {len(b['steps'])} saved"
        for si, (x, y) in enumerate(zip(a["steps"], b["steps"])):
            if x != y:
                keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
                return f"round {ri} step {si}: {keys} differ"
    return None


def fingerprint(step: dict) -> str:
    return hashlib.sha256(json.dumps(step, sort_keys=True).encode()).hexdigest()


class LiveRounds:
    """The rounds of one game as it plays. `observe` is the Game observer (the playing thread): a finished round is
    exported once; the round in play is kept as a copy of its events and records. `rounds` (any thread) returns the
    finished rounds plus the round in play exported up to its last settled decision point, cached per copy.
    `mismatch`: the first step a later export changed (None while every export extends the earlier ones)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._finished: List[dict] = []
        self._current = None  # (version, events, records) of the round in play
        self._version = 0
        self._cache = (-1, None)
        self._seen = {}  # (round number, step number) -> fingerprint of the first export of that step
        self._lengths = {}  # round number -> {version of the copy exported: its number of steps}
        self.mismatch: Optional[str] = None

    def _check_extends(self, ri: int, rnd: dict, version: int) -> None:
        """Every step equals the first export of it, and a later copy (version) never has fewer steps than an
        earlier one: exports of two copies may finish in either order (the playing thread, a poll)."""
        n = len(rnd["steps"])
        with self._lock:
            lengths = self._lengths.setdefault(ri, {})
            lengths[version] = n
            before = max((m for v, m in lengths.items() if v < version), default=0)
            after = min((m for v, m in lengths.items() if v > version), default=n)
            if (before > n or after < n) and self.mismatch is None:
                old, new = (before, n) if before > n else (n, after)
                self.mismatch = f"round {ri} shrank from {old} to {new} steps after it was shown"
        for si, step in enumerate(rnd["steps"]):
            fp = fingerprint(step)
            with self._lock:
                seen = self._seen.setdefault((ri, si), fp)
                if seen != fp and self.mismatch is None:
                    self.mismatch = f"round {ri} step {si} changed after it was shown"

    def observe(self, rnd, finished: bool) -> None:
        events = [{"event": tag, "attr": dict(attrs)} for tag, attrs in rnd.events]
        records = [dict(r) for r in rnd.decisions]
        if finished:
            exported = export.export_live_round(events, records, finished=True)
            self._check_extends(len(self._finished), exported, self._version + 1)  # only this thread moves _version
            with self._lock:
                self._finished.append(exported)
                self._current = None
                self._version += 1
        else:
            with self._lock:
                self._version += 1
                self._current = (self._version, events, records)

    def rounds(self) -> list:
        with self._lock:
            finished, current, cache = list(self._finished), self._current, self._cache
        if current is None:
            return finished
        version, events, records = current
        if cache[0] != version:
            cache = (version, export.export_live_round(events, records, finished=False))
            self._check_extends(len(finished), cache[1], version)
            with self._lock:
                if self._cache[0] < version:
                    self._cache = cache
        return finished + [cache[1]]


class SelfPlayJob:
    """One requested game: its config, its state (queued, running, done, error) and its rounds."""

    def __init__(self, job_id: str, cfg: dict, out_dir: str):
        self.id, self.cfg, self.out_dir = job_id, cfg, out_dir
        self.status, self.error = "queued", None
        self.players = [seat_label(s) for s in cfg["specs"]]  # the players' own labels once they are built
        self.live = LiveRounds()
        self.final: Optional[list] = None  # the saved game's rounds (export_game), once done
        self.created, self.finished_at = time.time(), None

    @property
    def path(self) -> str:
        return os.path.join(self.out_dir, "game.log")

    def snapshot(self, since: int) -> dict:
        rounds = self.final if self.final is not None else self.live.rounds()
        steps = flat_steps(rounds)
        return {"id": self.id, "status": self.status, "error": self.error, "players": self.players,
                "seed": self.cfg["seed"], "temperature": self.cfg["temperature"],
                "path": self.path if self.status == "done" else None, "source": self.id,
                "rounds": [{k: r[k] for k in ("index", "honba", "oya", "results")} for r in rounds],
                "total": len(steps),
                "steps": [{"round": ri, "step": s} for ri, s in steps[since:]]}


class SelfPlayRunner:
    """The jobs of this server process, a queue and the one worker thread that plays it (a failed game ends its
    job in 'error'; the worker and the server go on)."""

    def __init__(self, out_dir: str, default_ckpt: Optional[str] = None, ckpt_roots: tuple = ()):
        self.out_dir = os.path.abspath(out_dir)
        self.default_ckpt = default_ckpt
        self.ckpt_roots = tuple(os.path.realpath(r) for r in (*ckpt_roots, *(
            [os.path.dirname(os.path.realpath(default_ckpt))] if default_ckpt else [])))
        os.makedirs(self.out_dir, exist_ok=True)
        # every job of this process stays here until the server stops: its live rounds, then its saved rounds (a
        # hanchan's export is ~5 MB as JSON, more as Python objects); server.GameStore keeps each game's JSON bytes too
        self._jobs = {}
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        threading.Thread(target=self._worker, name="selfplay-worker", daemon=True).start()

    def submit(self, body) -> str:
        cfg = parse_request(body, self.default_ckpt, self.ckpt_roots)
        with self._lock:
            job_id = self._reserve(cfg["seed"])
            self._jobs[job_id] = SelfPlayJob(job_id, cfg, os.path.join(self.out_dir, job_id))
        self._queue.put(job_id)
        return job_id

    def _reserve(self, seed: int) -> str:
        """A new game folder `<YYYYmmdd-HHMMSS>_<seed>[-n]`, created now so no two jobs share it."""
        base = f"{time.strftime('%Y%m%d-%H%M%S')}_{seed:04d}"
        for n in range(1000):
            job_id = base if n == 0 else f"{base}-{n}"
            try:
                os.makedirs(os.path.join(self.out_dir, job_id))
                return job_id
            except FileExistsError:
                continue
        raise RuntimeError(f"no free game folder for {base}")

    def get(self, job_id: str, since: int) -> Optional[dict]:
        """A job's state from `since`; a job of an earlier server process is served from its folder: as error from its
        error.json, else as done from its saved game; None if unknown."""
        if not ID_PATTERN.match(job_id):
            raise BadRequest(f"bad game id {job_id!r}")
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            folder = os.path.join(self.out_dir, job_id)
            path = os.path.join(folder, "game.log")
            if os.path.isfile(os.path.join(folder, ERROR_FILE)):
                with open(os.path.join(folder, ERROR_FILE), encoding="utf-8") as f:
                    failed = json.load(f)
                job = SelfPlayJob(job_id, failed["cfg"], folder)
                job.status, job.error, job.final = "error", failed["error"], []
            elif os.path.isfile(path) and export.selfplay_meta(path) is not None:
                game = export.export_game(path, None)
                job = SelfPlayJob(job_id, {"specs": [], "seed": game["selfplay"]["seed"],
                                           "temperature": game["selfplay"]["temperature"]}, folder)
                job.players, job.status, job.final = game["players"], "done", game["rounds"]
            else:
                return None
        return job.snapshot(since)

    def saved(self) -> list:
        """The saved games, newest first: [{path, name}]; a failed job's folder (error.json) is left out."""
        out = []
        for path in sorted(glob.glob(os.path.join(self.out_dir, "*", "game.log")), reverse=True):
            meta = export.selfplay_meta(path)
            if meta is None or os.path.isfile(os.path.join(os.path.dirname(path), ERROR_FILE)):
                continue
            names = " / ".join(s["kind"] for s in meta["seats"])
            out.append({"path": path, "name": f"{export.selfplay_source(path)} · {names} · T={meta['temperature']}"})
        return out

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs[job_id]
            try:
                self._play(job)
            except Exception as e:  # one failed game must not stop the worker or the server
                print(f"self-play {job_id} failed\n{traceback.format_exc()}")
                job.error = f"{type(e).__name__}: {e}"
                try:
                    with open(os.path.join(job.out_dir, ERROR_FILE), "w", encoding="utf-8") as f:
                        json.dump({"status": "error", "error": job.error, "cfg": job.cfg}, f)
                except OSError:
                    print(f"self-play {job_id}: error.json not written\n{traceback.format_exc()}")
                job.status = "error"
            job.finished_at = time.time()

    def _play(self, job: SelfPlayJob) -> None:
        cfg = job.cfg
        job.status = "running"
        players = [make_player(spec, cfg["temperature"], cfg["seed"]) for spec in cfg["specs"]]
        job.players = [p.label for p in players]
        record = Game(players, job.players, cfg["seed"], cfg["temperature"], observer=job.live.observe).play()
        record.save(job.out_dir)
        final = export.export_game(job.path, None)["rounds"]
        diff = job.live.mismatch or first_difference(job.live.rounds(), final)
        if diff is not None:
            raise AssertionError(f"the live view differs from the saved game: {diff}")
        job.final = final
        job.status = "done"
