"""G3 end-to-end smoke (T-052): a real browser drives the served page.

Starts the server in a thread, opens the page in headless Chromium (Playwright), loads sample_haifu.xml and one
kan-dora fixture, steps forward through every decision point of every round (to each round's result view), steps
back, changes round, rotates the POV, toggles the hands, loads an exported JSON file, curls each API route, and
asserts 0 console / page errors and every tile image loaded. Also: a rapid A-then-B game switch keeps B (R3), two
nine-round games relabel the round picker (R4), an imported JSON with hostile strings renders them as text (R1), an
invalid JSON is refused with its field path, and a malformed log is a JSON 400 (R9). T-053: a self-play game started
from the page's form (model x2 + greedy + uniform; greedy / uniform x2 without --ckpt) is followed live (the step label
advances while it plays; a result view shows only for a finished round, and follow shows at least one finished
round's result during play and the last one at the end), stepped back and forth after it ends, equals the saved
game's /api/game export, and is listed under self-play; the game is saved under <shots>/selfplay. Screenshots go to
--shots.

    python -m MahjongAI.viewer.smoke_playwright [--ckpt pt] [--port 8766] [--shots dir] [--data-root dir]
"""
import argparse
import copy
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

from MahjongAI.viewer import export, infer, server

KAN_DORA_FIXTURE = "2021010101gm-00a9-0000-6422c07e.log"  # tests/test_kan_dora.py: open-kan indicator after the discard
NINE_ROUNDS_A = "2021010105gm-00a9-0000-0e0fb57b.log"  # 9 rounds, third = 東2局 0本場
NINE_ROUNDS_B = "2021010214gm-00a9-0000-7ed48ecb.log"  # 9 rounds, third = 東1局 2本場
SLOW_FIXTURE = "2021010823gm-00a9-0000-725fdda5.log"  # never requested before the race: exported on demand
HOSTILE = '<img src=x onerror="window.__pwned=1"><script>window.__pwned=2</script>&lt;b&gt;'


def fixture_path(name: str) -> str:
    import glob
    found = glob.glob(os.path.join(server.FIXTURE_DIR, "gate_games", "*", "*", "*", name))
    assert len(found) == 1, f"fixture {name}: {found}"
    return found[0]


def images_loaded(page) -> tuple:
    """(n images, n not loaded) on the page right now."""
    return tuple(page.evaluate(
        "() => { const a = Array.from(document.images); "
        "return [a.length, a.filter(i => !(i.complete && i.naturalWidth > 0)).length]; }"))


def step_label(page) -> str:
    return page.text_content("#step-label")


def status(page) -> str:
    return page.text_content("#status")


def round_labels(page) -> list:
    return page.evaluate("Array.from(document.querySelectorAll('#round-select option')).map(o => o.textContent)")


def load_fixture(page, path: str) -> None:
    page.select_option("#fixture-select", path)
    page.wait_for_function(f"document.querySelector('#status').textContent.startsWith({json.dumps(os.path.basename(path))})")


def walk_game(page, game: dict, shots: str, tag: str, log) -> None:
    """Forward through every decision point of every round, back a few, every round picker entry."""
    n_rounds = len(game["rounds"])
    total = 0
    for ri in range(n_rounds):
        page.select_option("#round-select", str(ri))
        n = len(game["rounds"][ri]["steps"])
        for _ in range(n):
            page.click("#next")
        total += n
        assert step_label(page).startswith("結果"), f"{tag} round {ri}: {step_label(page)!r} after {n} steps"
        assert page.query_selector("#panel .result, #panel .hint") is not None, f"{tag} round {ri}: no result view"
        if ri == 0:
            page.screenshot(path=os.path.join(shots, f"{tag}_round0_result.png"))
    log(f"  {tag}: {n_rounds} rounds, {total} decision points stepped, every round reached its result")
    for _ in range(5):
        page.keyboard.press("ArrowLeft")
    assert not step_label(page).startswith("結果"), step_label(page)
    page.keyboard.press("Home")
    assert step_label(page).startswith("1 /"), step_label(page)
    page.keyboard.press("ArrowDown")  # next discard
    page.keyboard.press("End")
    assert step_label(page).startswith("結果"), step_label(page)


def hostile_game(game: dict) -> dict:
    """The exported game with markup in every string field the panel shows: names, a draw reason, a yaku."""
    g = copy.deepcopy(game)
    g["players"][0] = HOSTILE + " p0"
    g["source"] = HOSTILE + ".log"
    r0 = g["rounds"][0]
    last = r0["steps"][-1]
    r0["results"] = [{"type": "ryuukyoku", "reason": HOSTILE, "tenpai": None, "revealed": [0], "deltas": [0] * 4,
                      "scores_after": list(last["scores"])}]
    agari = next(x for r in g["rounds"] for x in r["results"] if x["type"] == "agari")
    agari["yaku"] = [[HOSTILE + " yaku", 1]]
    agari["limit"] = HOSTILE
    return g


def api_error(base: str, path: str) -> tuple:
    """(status, error message) of GET /api/game?path=... that must fail."""
    try:
        urllib.request.urlopen(f"{base}/api/game?path={urllib.request.quote(path)}")
    except urllib.error.HTTPError as e:
        assert e.headers.get("Content-Type", "").startswith("application/json"), e.headers
        return e.code, json.load(e)["error"]
    raise AssertionError(f"{path} was served")


def selfplay_scenario(page, base: str, shots: str, has_ckpt: bool, log) -> dict:
    """Start a game from the form, follow it live, step it after the end; returns what was seen."""
    seats = ["model", "model", "greedy", "uniform"] if has_ckpt else ["greedy", "uniform", "greedy", "uniform"]
    for i, kind in enumerate(seats):
        page.select_option(f".sp-seat >> nth={i}", kind)
    page.fill("#sp-seed", "7")
    page.fill("#sp-temp", "0")
    page.check("#sp-follow")
    t0 = time.perf_counter()
    page.click("#sp-start")
    seen = []  # (round, step label) each time it changed while the game was running
    results_seen = set()  # rounds whose result view follow showed during play
    state = page.evaluate("window.viewerState()")
    while time.perf_counter() - t0 < 600:
        state = page.evaluate("window.viewerState()")
        if state["liveStatus"] in ("done", "error"):
            break
        if state["liveStatus"] == "running" and state["source"] == state["live"]:
            here = (state["round"], step_label(page))
            if here[1].startswith("結果"):  # a result view: only for a round that has its result
                n_results = page.evaluate(f"window.viewerGame().rounds[{here[0]}].results.length")
                assert n_results > 0, f"result view of round {here[0]} in play, without a result: {here}"
                results_seen.add(here[0])
            if not seen or seen[-1] != here:
                seen.append(here)
                if len(seen) == 2:
                    page.screenshot(path=os.path.join(shots, "selfplay_live.png"))
        page.wait_for_timeout(250)
    wall = time.perf_counter() - t0
    assert state["liveStatus"] == "done", (state, status(page))
    assert len(seen) >= 3, f"the step label advanced {len(seen) - 1} times during play: {seen}"
    assert results_seen, f"follow never showed a finished round's result during play: {seen}"
    log(f"  self-play {state['live']} ({', '.join(seats)}): done in {wall:.1f} s; followed live, the step label "
        f"advanced {len(seen) - 1} times during play: {seen[0]} -> {seen[-1]}; finished rounds whose result "
        f"follow showed during play: {sorted(results_seen)}")
    game_id = state["live"]
    end = (state["round"], step_label(page))
    assert end[1].startswith("結果"), f"follow did not end on the last round's result: {end}"
    for _ in range(12):
        page.click("#prev")
    back = (page.evaluate("window.viewerState()")["round"], step_label(page))
    assert back != end, (back, end)
    for _ in range(12):
        page.click("#next")
    assert (page.evaluate("window.viewerState()")["round"], step_label(page)) == end, step_label(page)
    page.select_option("#round-select", "0")
    page.keyboard.press("End")
    assert step_label(page).startswith("結果"), step_label(page)
    page.keyboard.press("Home")
    assert step_label(page).startswith("1 /"), step_label(page)
    page.screenshot(path=os.path.join(shots, "selfplay_after_end.png"))
    log(f"  self-play after the end: back 12 to {back}, forward 12 to {end}, round 0 result and first point")
    live_rounds = json.loads(page.evaluate("JSON.stringify(window.viewerGame().rounds)"))
    snap = json.load(urllib.request.urlopen(f"{base}/api/selfplay/{game_id}?since=0"))
    with urllib.request.urlopen(f"{base}/api/game?path={urllib.request.quote(snap['path'])}") as r:
        saved = json.load(r)
    assert live_rounds == saved["rounds"], "the live game differs from the saved game's export"
    n_steps = sum(len(r["steps"]) for r in saved["rounds"])
    log(f"  self-play live game == /api/game export of {snap['path']}: {len(saved['rounds'])} rounds, "
        f"{n_steps} decision points")
    page.wait_for_function("document.querySelectorAll('#selfplay-select option').length > 1")
    page.select_option("#selfplay-select", snap["path"])
    page.wait_for_function(f"document.querySelector('#status').textContent.startsWith({json.dumps(game_id)})")
    for _ in range(5):
        page.click("#next")
    assert step_label(page).startswith("6 /"), step_label(page)
    log(f"  self-play picker: {game_id} reloaded from disk and stepped")
    return {"selfplay_wall_s": wall, "selfplay_labels": len(seen), "selfplay_steps": n_steps}


def run(port: int, ckpt: str, shots: str, data_root: str = server.DEFAULT_DATA_ROOT, log=print) -> dict:
    from playwright.sync_api import sync_playwright

    os.makedirs(shots, exist_ok=True)
    model = infer.load_model(ckpt) if ckpt else None
    tmp = tempfile.mkdtemp()
    # the followed self-play game is kept next to the screenshots
    httpd = server.make_server(model, data_root, port, extra_roots=(tmp,), selfplay_dir=os.path.join(shots, "selfplay"),
                               ckpt=ckpt)
    store = httpd.RequestHandlerClass.keywords["store"]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    errors = []
    report = {"console_errors": errors}
    try:
        # the API routes
        fixtures = json.load(urllib.request.urlopen(f"{base}/api/fixtures"))["games"]
        assert any(g["name"] == "sample_haifu.xml" for g in fixtures), fixtures[:3]
        games = json.load(urllib.request.urlopen(f"{base}/api/games?date=20220101"))["games"]
        log(f"  /api/fixtures: {len(fixtures)} logs; /api/games?date=20220101: {len(games)} logs")
        code, msg = api_error(base, "/etc/passwd")
        assert code == 400, code
        log(f"  /api/game outside the roots -> {code} {msg}")
        sample = next(g["path"] for g in fixtures if g["name"] == "sample_haifu.xml")
        kan_dora = fixture_path(KAN_DORA_FIXTURE)
        malformed = os.path.join(tmp, "truncated.log")
        with open(sample, "rb") as f, open(malformed, "wb") as out:
            out.write(f.read()[:4000])  # cut inside a tag: not well-formed XML
        code, msg = api_error(base, malformed)
        assert code == 400 and msg.startswith("malformed log:"), (code, msg)
        log(f"  /api/game on a truncated log -> {code} {msg}")
        with urllib.request.urlopen(f"{base}/api/game?path={urllib.request.quote(sample)}") as r:
            sample_game = json.load(r)
        with urllib.request.urlopen(f"{base}/api/game?path={urllib.request.quote(kan_dora)}") as r:
            kan_game = json.load(r)
        exported = os.path.join(tmp, "exported.json")
        with open(exported, "w", encoding="utf-8") as f:
            json.dump(export.export_game(kan_dora, model), f, ensure_ascii=False)
        hostile = os.path.join(tmp, "hostile.json")
        hostile_g = hostile_game(kan_game)
        with open(hostile, "w", encoding="utf-8") as f:
            json.dump(hostile_g, f, ensure_ascii=False)
        invalid = os.path.join(tmp, "invalid.json")
        bad = copy.deepcopy(kan_game)
        bad["rounds"][0]["steps"][0]["hands"][0][0] = "x"
        with open(invalid, "w", encoding="utf-8") as f:
            json.dump(bad, f, ensure_ascii=False)

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 920})
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            # a superseded /api/game fetch is aborted by the page on purpose (R3): not a failure
            page.on("requestfailed", lambda r: errors.append(f"requestfailed: {r.url} {r.failure}")
                    if not ("/api/game?" in r.url and "ERR_ABORTED" in str(r.failure)) else None)
            page.goto(f"{base}/")
            page.wait_for_function("document.querySelectorAll('#fixture-select option').length > 1")

            load_fixture(page, sample)
            page.screenshot(path=os.path.join(shots, "sample_first_point.png"))
            for _ in range(40):
                page.click("#next")
            page.screenshot(path=os.path.join(shots, "sample_point_41.png"))
            walk_game(page, sample_game, shots, "sample_haifu", log)

            load_fixture(page, kan_dora)
            walk_game(page, kan_game, shots, "kan_dora", log)
            # the open-kan point of tests/test_kan_dora.py: round 9, discard 107 -> post 108 flips the indicator
            page.select_option("#round-select", "9")
            for _ in range(108):
                page.click("#next")
            n_doras = page.evaluate("document.querySelectorAll('#doras .tile').length")
            assert n_doras == 2, f"kan-dora round 9 post 108 shows {n_doras} indicators, expected 2"
            page.click("#prev")
            n_doras = page.evaluate("document.querySelectorAll('#doras .tile').length")
            assert n_doras == 1, f"kan-dora round 9 discard 107 shows {n_doras} indicators, expected 1"
            n_called = page.evaluate("document.querySelectorAll('.pond .tile.called.side img').length")
            assert n_called > 0, "no called pond tile with a face at this point"
            page.screenshot(path=os.path.join(shots, "kan_dora_round9_discard107.png"))
            log(f"  kan_dora round 9 discard 107: {n_called} called pond tiles drawn with their face, sideways")

            for pov in (1, 2, 3, 0):
                page.select_option("#pov", str(pov))
                assert page.evaluate("document.querySelector('.seat.pos-0 .hand .tile') !== null")
            page.uncheck("#open-hands")
            assert page.evaluate("document.querySelectorAll('.seat.pos-2 .hand .tile.back').length") > 0
            page.check("#open-hands")
            page.click("#play")
            page.wait_for_timeout(1200)
            page.click("#play")

            # R4: two nine-round games relabel the picker
            load_fixture(page, fixture_path(NINE_ROUNDS_A))
            labels_a = round_labels(page)
            load_fixture(page, fixture_path(NINE_ROUNDS_B))
            labels_b = round_labels(page)
            assert len(labels_a) == len(labels_b) == 9, (labels_a, labels_b)
            assert labels_a[2] == "3: 東2局 0本場" and labels_b[2] == "3: 東1局 2本場", (labels_a[2], labels_b[2])
            page.select_option("#round-select", "2")
            page.screenshot(path=os.path.join(shots, "picker_equal_rounds.png"))
            log(f"  round picker: {labels_a[2]!r} -> {labels_b[2]!r} on two 9-round games")

            # R3: a slow (uncached) A then a cached B: B stays after A's export completes
            slow = fixture_path(SLOW_FIXTURE)
            assert os.path.realpath(slow) not in store._cache
            page.select_option("#fixture-select", slow)
            page.select_option("#fixture-select", sample)
            page.wait_for_function("document.querySelector('#status').textContent.startsWith('sample_haifu.xml')")
            deadline = 120
            while os.path.realpath(slow) not in store._cache and deadline > 0:
                page.wait_for_timeout(250)
                deadline -= 1
            assert os.path.realpath(slow) in store._cache, "the slow export never finished"
            page.wait_for_timeout(500)
            state = page.evaluate("window.viewerState()")
            assert state["source"] == "sample_haifu.xml", (state, status(page))
            assert status(page).startswith("sample_haifu.xml"), status(page)
            log(f"  rapid A-then-B: B ({state['source']}) still shown after A's export finished (loadGen {state['loadGen']})")

            page.set_input_files("#json-file", exported)
            page.wait_for_function("document.querySelector('#status').textContent.includes('(file)')")
            assert step_label(page).startswith("1 /"), step_label(page)
            page.screenshot(path=os.path.join(shots, "json_file_loaded.png"))

            # an invalid file is refused with its field path; the previous game stays
            page.set_input_files("#json-file", invalid)
            page.wait_for_function("document.querySelector('#status').textContent.startsWith('error: invalid trajectory')")
            assert "rounds[0].steps[0].hands[0][0] is not a tile id" in status(page), status(page)
            assert page.evaluate("window.viewerState()")["source"] == KAN_DORA_FIXTURE
            log(f"  invalid JSON refused: {status(page)}")

            # R1: hostile strings render as text
            page.set_input_files("#json-file", hostile)
            page.wait_for_function("document.querySelector('#status').textContent.includes('(file)')")
            page.keyboard.press("End")
            title = page.text_content("#panel .result .title")
            assert HOSTILE in title, title
            assert page.evaluate("window.__pwned") is None
            assert page.evaluate("document.querySelectorAll('#panel script, #panel img:not([src^=\"tiles/\"])').length") == 0
            assert HOSTILE in page.text_content(".seat.pos-0 .hand-row .name")
            page.screenshot(path=os.path.join(shots, "hostile_json_result.png"))
            agari_round = next(i for i, r in enumerate(hostile_g["rounds"])
                               if r["results"] and r["results"][0]["type"] == "agari")
            page.select_option("#round-select", str(agari_round))
            page.keyboard.press("End")
            assert HOSTILE + " yaku" in page.text_content("#panel .result"), page.text_content("#panel .result")
            assert page.evaluate("window.__pwned") is None
            log("  hostile JSON: reason, name, yaku and limit rendered as text; window.__pwned undefined")

            report.update(selfplay_scenario(page, base, shots, bool(ckpt), log))

            n_img, n_bad = images_loaded(page)
            report.update({"images": n_img, "images_not_loaded": n_bad})
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--shots", default=os.path.join(tempfile.gettempdir(), "viewer_shots"))
    ap.add_argument("--data-root", default=server.DEFAULT_DATA_ROOT)
    args = ap.parse_args(argv)
    report = run(args.port, args.ckpt, args.shots, args.data_root)
    print(f"G3: console errors {len(report['console_errors'])}, images {report['images']}, "
          f"not loaded {report['images_not_loaded']}, screenshots in {args.shots}")
    for e in report["console_errors"]:
        print(f"  {e}")
    ok = not report["console_errors"] and report["images_not_loaded"] == 0
    print("G3 PASS" if ok else "G3 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
