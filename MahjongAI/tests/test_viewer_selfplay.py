"""T-053 phase 2: self-play from the viewer. One uniform / greedy game is started through POST /api/selfplay and polled
with `since` until it ends; for its first two rounds the live round is exported at EVERY decision point (the server's
own check then proves each export extends the earlier ones and the saved game equals the live one). Then: the polled
steps equal the saved game's /api/game export, the listing, a restarted server serves the saved game, the 400 paths,
a game that fails does not stop the worker and leaves its error.json; the export shows every sidecar record (its
player's distribution and choice, not the viewer model's) with the meta.json labels and passes the T-052 display checks.
The page's pure logic (viewer.js under node): a round in play has no result view, follow shows a finished round's
result, a poll whose points do not add up is refused."""
import http.client
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

import pytest

from MahjongAI.viewer import export, selfplay, server, viewer_lib

SEATS = ["uniform", "greedy", "uniform", "greedy"]
SEED = 3
STRICT_ROUNDS = 2  # rounds exported live at every decision point


def start(selfplay_dir, ckpt=None, ckpt_roots=()):
    httpd = server.make_server(None, str(selfplay_dir), 0, selfplay_dir=str(selfplay_dir), ckpt=ckpt,
                               ckpt_roots=ckpt_roots)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def call(base, path, body=None):
    """(status, JSON body) of a GET, or of a POST when `body` is given (bytes are sent as they are)."""
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    req = urllib.request.Request(base + path, data=data, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        assert e.headers["Content-Type"].startswith("application/json")
        return e.code, json.load(e)


def wait_done(base, game_id, timeout=600):
    """Poll with `since` like the page: (final snapshot, [(round, step)] collected, number of polls)."""
    collected, polls, t0 = [], 0, time.time()
    while time.time() - t0 < timeout:
        code, snap = call(base, f"/api/selfplay/{game_id}?since={len(collected)}")
        assert code == 200, (code, snap.get("error"))
        collected += [(s["round"], s["step"]) for s in snap["steps"]]
        assert snap["total"] == len(collected)
        polls += 1
        if snap["status"] in ("done", "error"):
            return snap, collected, polls
        time.sleep(0.5)
    raise AssertionError(f"{game_id} did not end in {timeout} s")


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    folder = tmp_path_factory.mktemp("selfplay")
    ckpt_root = tmp_path_factory.mktemp("checkpoints")
    original = selfplay.LiveRounds.observe
    counts = {"live exports": 0}

    def strict(self, rnd, finished):  # export the round in play at every decision point of the first rounds
        original(self, rnd, finished)
        if not finished and len(self._finished) < STRICT_ROUNDS:
            self.rounds()
            counts["live exports"] += 1

    selfplay.LiveRounds.observe = strict
    httpd, base = start(folder, ckpt_roots=(str(ckpt_root),))
    try:
        code, body = call(base, "/api/selfplay", {"seats": SEATS, "seed": SEED, "temperature": 0})
        assert code == 201, body
        snap, collected, polls = wait_done(base, body["id"])
    finally:
        selfplay.LiveRounds.observe = original
    yield {"base": base, "httpd": httpd, "folder": folder, "ckpt_root": ckpt_root, "id": body["id"], "snap": snap,
           "collected": collected, "polls": polls, "live_exports": counts["live exports"]}
    httpd.shutdown()
    httpd.server_close()


def test_game_runs_to_the_end_and_live_equals_saved(played):
    snap = played["snap"]
    assert snap["status"] == "done", snap["error"]
    assert played["live_exports"] > 100 and played["polls"] >= 3, (played["live_exports"], played["polls"])
    assert snap["players"] == SEATS and snap["seed"] == SEED
    code, saved = call(played["base"], f"/api/game?path={urllib.request.quote(snap['path'])}")
    assert code == 200
    assert played["collected"] == selfplay.flat_steps(saved["rounds"])
    assert [r["results"] for r in saved["rounds"]] == [r["results"] for r in snap["rounds"]]
    code, tail = call(played["base"], f"/api/selfplay/{played['id']}?since={snap['total']}")
    assert code == 200 and tail["steps"] == [] and tail["total"] == snap["total"]


def test_listing_and_a_restarted_server(played):
    code, listing = call(played["base"], "/api/selfplay")
    assert code == 200 and [g["path"] for g in listing["games"]] == [played["snap"]["path"]]
    assert listing["games"][0]["name"].startswith(played["id"])
    httpd, base = start(played["folder"])  # a new process would see only the folder
    try:
        code, snap = call(base, f"/api/selfplay/{played['id']}?since=5")
        assert code == 200 and snap["status"] == "done" and snap["total"] == played["snap"]["total"]
        assert [(s["round"], s["step"]) for s in snap["steps"]] == played["collected"][5:]
        code, game = call(base, f"/api/game?path={urllib.request.quote(played['snap']['path'])}")
        assert code == 200 and game["source"] == played["id"] and game["players"] == SEATS
        code, body = call(base, "/api/selfplay/20200101-000000_0000")
        assert code == 404, body
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize("body, message", [
    ({"seats": ["greedy"] * 3}, "seats must be a list of 4 strings"),
    ({"seats": ["greedy", "random", "greedy", "greedy"]}, "seat 1: unknown player 'random'"),
    ({"seats": ["greedy:x", "greedy", "greedy", "greedy"]}, "seat 0: unknown player"),
    ({"seats": ["model", "greedy", "greedy", "greedy"]}, "seat 0: 'model' needs a checkpoint"),
    ({"seats": ["greedy", "greedy", "greedy", "model:/no/such.pt"]}, "seat 3: checkpoint /no/such.pt is not under"),
    ({"seats": ["greedy", "greedy", "greedy", "model:/etc/hosts"]}, "seat 3: checkpoint /etc/hosts is not under"),
    ({"seats": ["greedy"] * 4, "seed": -1}, "seed must be an integer"),
    ({"seats": ["greedy"] * 4, "seed": "7"}, "seed must be an integer"),
    ({"seats": ["greedy"] * 4, "seed": 1.5}, "seed must be an integer"),
    ({"seats": ["greedy"] * 4, "seed": True}, "seed must be an integer"),
    ({"seats": ["greedy"] * 4, "temperature": -0.1}, "temperature must be a number"),
    ({"seats": ["greedy"] * 4, "temperature": "hot"}, "temperature must be a number"),
    ([1, 2], "the body must be a JSON object"),
    (b"{not json", "the body is not JSON"),
])
def test_bad_requests_are_400s(played, body, message):
    code, answer = call(played["base"], "/api/selfplay", body)
    assert code == 400 and message in answer["error"], answer


def test_checkpoints_load_only_under_the_roots(played):
    root = played["ckpt_root"]
    (root / "escape.pt").symlink_to("/etc/hosts")  # inside by name, outside once resolved
    for seat, message in ((f"model:{root / 'escape.pt'}", "is not under the checkpoint roots"),
                          (f"model:{root / 'missing.pt'}", "not found"),
                          (f"model:{root}/../{root.name}-x/a.pt", "is not under the checkpoint roots")):
        code, answer = call(played["base"], "/api/selfplay", {"seats": [seat, "greedy", "greedy", "greedy"]})
        assert code == 400 and message in answer["error"], (seat, answer)


def raw_post(conn, length_header, body=b""):
    conn.putrequest("POST", "/api/selfplay")
    conn.putheader("Content-Length", length_header)
    conn.endheaders(body)
    r = conn.getresponse()
    return r.status, r.getheader("Connection"), json.loads(r.read())


def test_post_with_a_bad_or_too_long_body_is_a_400_and_the_connection_stays_sound(played):
    port = played["httpd"].server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        for bad in ("abc", "-5", "²"):  # "²".isdigit() is True, int("²") raises
            status, connection, answer = raw_post(conn, bad)
            assert status == 400 and "Content-Length must be" in answer["error"] and connection == "close", answer
            conn.close()
        status, connection, answer = raw_post(conn, str(server.MAX_BODY + 4000), b"x" * (server.MAX_BODY + 4000))
        assert status == 400 and answer["error"] == "request body too large" and connection is None, answer
        conn.request("GET", "/api/selfplay")  # the same connection: the body was drained, not parsed as a request
        r = conn.getresponse()
        assert r.status == 200 and "games" in json.loads(r.read())
    finally:
        conn.close()


def test_bad_since_and_id_are_400s(played):
    code, answer = call(played["base"], f"/api/selfplay/{played['id']}?since=-1")
    assert code == 400 and "since" in answer["error"], answer
    code, answer = call(played["base"], "/api/selfplay/..%2F..%2Fetc")
    assert code == 400 and "bad game id" in answer["error"], answer


def assert_failed_job_is_kept(played, game_id, error_part):
    """error.json in the job folder, the listing skips it, a restarted server reports the error."""
    with open(os.path.join(played["folder"], game_id, selfplay.ERROR_FILE)) as f:
        failed = json.load(f)
    assert failed["status"] == "error" and error_part in failed["error"] and len(failed["cfg"]["specs"]) == 4, failed
    code, listing = call(played["base"], "/api/selfplay")
    assert code == 200 and game_id not in {g["name"].split(" ")[0] for g in listing["games"]}
    httpd, base = start(played["folder"])
    try:
        code, snap = call(base, f"/api/selfplay/{game_id}")
        assert code == 200 and snap["status"] == "error" and snap["error"] == failed["error"], snap
        code, listing = call(base, "/api/selfplay")
        assert game_id not in {g["name"].split(" ")[0] for g in listing["games"]}
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_failing_game_ends_in_error_and_the_worker_goes_on(played):
    bad = played["ckpt_root"] / "broken.pt"
    bad.write_bytes(b"not a checkpoint")
    code, body = call(played["base"], "/api/selfplay", {"seats": [f"model:{bad}", "greedy", "greedy", "greedy"]})
    assert code == 201, body
    snap, _, _ = wait_done(played["base"], body["id"], timeout=60)
    assert snap["status"] == "error" and snap["error"], snap
    assert_failed_job_is_kept(played, body["id"], snap["error"])
    assert snap["players"][0] == "model:broken.pt"  # a queued / failed job shows the file name, not the host path


def test_a_saved_game_that_fails_the_live_check_is_not_served_as_done(played, monkeypatch):
    source = os.path.dirname(played["snap"]["path"])

    class SavedOnly:  # saves the played game's files but shows nothing live: the live-vs-saved check must fail
        def __init__(self, *args, **kwargs):
            pass

        def play(self):
            return self

        def save(self, out_dir):
            for name in ("game.log", "decisions.json", "meta.json"):
                shutil.copy(os.path.join(source, name), out_dir)

    monkeypatch.setattr(selfplay, "Game", SavedOnly)
    code, body = call(played["base"], "/api/selfplay", {"seats": ["greedy"] * 4, "seed": 11})
    assert code == 201, body
    snap, _, _ = wait_done(played["base"], body["id"], timeout=120)
    assert snap["status"] == "error" and "the live view differs from the saved game" in snap["error"], snap
    assert os.path.isfile(os.path.join(played["folder"], body["id"], "game.log"))
    assert_failed_job_is_kept(played, body["id"], "the live view differs")


def test_export_shows_each_seats_own_player(played):
    path = played["snap"]["path"]
    meta, records = export.selfplay_sidecar(path)
    game = viewer_lib.check_log(path)  # T-052 display checks, admission off for a generated game
    assert game["players"] == [s["label"] for s in meta["seats"]] == SEATS
    shown = {}
    for ri, rnd in enumerate(game["rounds"]):
        for si, step in enumerate(rnd["steps"]):
            for d in step["decisions"]:
                if "player" in d:
                    shown[(ri, si, d["seat"])] = d
                else:
                    assert len(d["legal"]) == 1, f"round {ri} step {si} seat {d['seat']}: no sidecar record"
    assert len(shown) == len(records) == meta["n_decisions"]
    for rec in records:
        d = shown[(rec["kyoku"], rec["halfturn"], rec["seat"])]
        assert d["probs"] == rec["candidates"] and d["logits"] == {}
        assert d["player"]["label"] == rec["player"] == SEATS[rec["seat"]]
        assert d["player"]["chosen"]["idx"] == rec["chosen"] and d["player"]["sampled"] == rec["sampled"]
        assert d["player"]["resolved"] == rec["resolved"]["outcome"]
        if rec["player"] == "greedy":  # its own one-hot, not the viewer model's distribution
            assert sorted(d["probs"].values())[-1] == 1.0 and d["model_top"]["idx"] == rec["chosen"]
        elif len(rec["offer"]) > 1:  # uniform: a flat distribution has no unique top
            assert len(set(d["probs"].values())) == 1 and d["model_top"] is None


def test_export_refuses_a_record_that_does_not_match_the_parse(played, tmp_path):
    folder = os.path.dirname(played["snap"]["path"])
    for name in ("game.log", "meta.json"):
        (tmp_path / name).write_bytes(open(os.path.join(folder, name), "rb").read())
    records = json.load(open(os.path.join(folder, "decisions.json")))
    rec = next(r for r in records if r["kind"] == "discard" and len(r["offer"]) > 1)
    rec["offer"] = rec["offer"][1:]
    (tmp_path / "decisions.json").write_text(json.dumps(records))
    with pytest.raises(AssertionError, match="sidecar record"):
        export.export_game(str(tmp_path / "game.log"), None)
    del records[records.index(rec)]
    (tmp_path / "decisions.json").write_text(json.dumps(records))
    with pytest.raises(export.Unsettled, match="sidecar record missing"):
        export.export_game(str(tmp_path / "game.log"), None)


def test_export_refuses_a_self_play_game_with_a_dropped_round(played, tmp_path):
    folder = os.path.dirname(played["snap"]["path"])
    for name in ("decisions.json", "meta.json"):
        shutil.copy(os.path.join(folder, name), tmp_path)
    log = open(os.path.join(folder, "game.log"), "rb").read()
    second = log.index(b"<INIT ", log.index(b"<INIT ") + 1)
    (tmp_path / "game.log").write_bytes(log[:second] + b'<BYE who="1" />' + log[second:])  # cuts the rounds after it
    with pytest.raises(AssertionError, match="rounds dropped"):
        export.export_game(str(tmp_path / "game.log"), None)


def test_live_checks_catch_a_changed_step_and_a_different_saved_game():
    rnd = {"index": 0, "honba": "0", "oya": 0, "results": [], "steps": [{"a": 1}, {"a": 2}]}
    live = selfplay.LiveRounds()
    live._check_extends(0, dict(rnd, steps=[{"a": 1}, {"a": 2}, {"a": 3}]), 2)
    live._check_extends(0, rnd, 1)  # an older copy whose export finished later (a poll): not a shrink
    assert live.mismatch is None
    live._check_extends(0, dict(rnd, steps=[{"a": 1}, {"a": 9}, {"a": 3}]), 3)
    assert live.mismatch == "round 0 step 1 changed after it was shown"
    for order in ((1, 2), (2, 1)):  # the newer copy has a step fewer, whichever export ends first
        shorter = selfplay.LiveRounds()
        for version in order:
            shorter._check_extends(1, dict(rnd, steps=rnd["steps"] + [{"a": 3}] * (version == 1)), version)
        assert shorter.mismatch == "round 1 shrank from 3 to 2 steps after it was shown", order
    assert selfplay.first_difference([rnd], [rnd]) is None
    assert selfplay.first_difference([rnd], [dict(rnd, steps=[{"a": 1}, {"a": 3}])]) == "round 0 step 1: ['a'] differ"
    assert selfplay.first_difference([rnd], [dict(rnd, oya=1)]) == "round 0 oya: 0 vs 1"
    assert selfplay.first_difference([rnd], []) == "1 live rounds vs 0 saved"


def test_a_queued_job_shows_checkpoint_file_names():
    job = selfplay.SelfPlayJob("x", {"specs": ["model:/runs/a/step_9.pt", "greedy", "uniform", "model:/b/c.pt"],
                                     "seed": 0, "temperature": 0.0}, "/nowhere/x")
    assert job.snapshot(0)["players"] == ["model:step_9.pt", "greedy", "uniform", "model:c.pt"]


VIEWER_JS = os.path.join(os.path.dirname(selfplay.__file__), "static", "viewer.js")
PAGE_LOGIC = r"""
const assert = require("assert");
const V = require(process.argv[1]);
const s = { kind: "discard" };
const done = { results: [{ type: "ryuukyoku" }], steps: [s, s] };
const inPlay = { results: [], steps: [s, s, s] };
const game = { rounds: [done, inPlay] };
const at = (round, step) => ({ game, round, step, pov: 0, open: true, playing: true });
// a round in play has no result view: End, next and the step clamp stop at its last point
assert.strictEqual(V.reduce(at(1, 0), { type: "step", step: Infinity }).step, 2);
const next = V.reduce(at(1, 2), { type: "next" });
assert.deepStrictEqual([next.round, next.step, next.playing], [1, 2, false]);
assert.strictEqual(V.reduce(at(1, 1), { type: "nextDiscard" }).step, 2);
assert.strictEqual(V.reduce(at(1, 0), { type: "goto", round: 1, step: 3 }).step, 2);
// a finished round keeps its result view
assert.strictEqual(V.reduce(at(0, 0), { type: "step", step: Infinity }).step, 2);
const prev = V.reduce(at(1, 0), { type: "prev" });
assert.deepStrictEqual([prev.round, prev.step], [0, 2]);
// follow: a round that finished since the last poll shows its result; else the last point of the last round
assert.deepStrictEqual(V.followTarget(game, 0), { round: 0, step: 2 });
assert.deepStrictEqual(V.followTarget(game, 1), { round: 1, step: 2 });
const over = { rounds: [done, { results: [{ type: "agari" }], steps: [s, s, s] }] };
assert.deepStrictEqual(V.followTarget(over, 1), { round: 1, step: 3 });
assert.deepStrictEqual(V.followTarget(over, 2), { round: 1, step: 3 });
// a poll whose points do not add up to the game's count is refused, whatever `since` it echoes
const page = { source: "g", players: [], rounds: [{ index: 0, honba: 0, oya: 0, results: [], steps: [s, s] }] };
const header = [{ index: 0, honba: 0, oya: 0, results: [] }];
const ok = V.mergeLive(page, { since: 2, total: 3, rounds: header, steps: [{ round: 0, step: s }], status: "running" });
assert.strictEqual(ok.rounds[0].steps.length, 3);
assert.throws(() => V.mergeLive(page, { since: 2, total: 4, rounds: header, steps: [{ round: 0, step: s }],
  status: "running" }), /the game 4/);
assert.throws(() => V.mergeLive(page, { since: 2, total: 2, rounds: header, steps: [{ round: 0, step: s }],
  status: "running" }), /the game 2/);
console.log("page logic ok");
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_logic_under_node():
    r = subprocess.run(["node", "-e", PAGE_LOGIC, VIEWER_JS], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "page logic ok" in r.stdout, r.stderr
