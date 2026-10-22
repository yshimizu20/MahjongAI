"""T-053 engine: one uniform / greedy hanchan parses, passes the bookkeeping checks, replays byte for byte from its
meta.json seats, and its sidecar matches the completed parse (coverage, exact actions); the prefix-vs-completed parity
on one real log."""
import json
import os

import pytest

from MahjongAI.engine import engine_lib
from MahjongAI.engine.game import Game
from MahjongAI.engine.players import GreedyPlayer, UniformPlayer
from MahjongAI.process_xml import process
from MahjongAI.utils import gates


@pytest.fixture(scope="module")
def game():
    players = [UniformPlayer(), GreedyPlayer(), UniformPlayer(), GreedyPlayer()]
    labels = [p.label for p in players]
    for p in players:
        p.reseed(3)
    return players, labels, Game(players, labels, 3).play()


def test_game_checks_and_sidecar(game):
    players, labels, record = game
    engine_lib.check_rounds(record.xml, "seed 3")
    engine_lib.check_agari_tags(record.xml, "seed 3")
    engine_lib.check_kan_emission(record.xml, "seed 3")
    counts = engine_lib.check_sidecar(record.xml, record.decisions, where="seed 3")
    assert counts["offer parity: discard"] > 100 and counts["resolved: post passed"] > 0
    assert record.meta["rounds"] >= 4 and record.meta["final_scores"] is not None


def test_game_replays_identically_from_meta_seats(game):
    players, labels, record = game
    assert [d["kind"] for d in record.meta["seats"]] == ["uniform", "greedy", "uniform", "greedy"]
    assert record.meta["seats"][0] == {"kind": "uniform", "label": "uniform", "seed": 3}
    rebuilt = engine_lib.players_of(record.meta)
    engine_lib.check_replay(record.xml, record.decisions, 3, rebuilt, labels, where="seed 3")


def test_saved_game_parses_and_counts(game, tmp_path):
    _, _, record = game
    record.save(str(tmp_path))
    log = os.path.join(str(tmp_path), "game.log")
    H, T = process(log, admission=False)
    assert len(H) == record.meta["rounds"] == json.load(open(os.path.join(str(tmp_path), "meta.json")))["rounds"]
    counters = gates.run([log], admission=False)
    assert counters["games"] == 1 and counters["skipped_invalid"] == 0 and counters["parse_error"] == 0


def _copy(record):
    return json.loads(json.dumps(record.decisions))


def test_sidecar_catches_a_flipped_choice_as_the_wrong_executed_action(game):
    _, _, record = game
    bad = _copy(record)
    engine_lib._normalised(bad[0])["chosen"] = next(a for a in bad[0]["offer"] if a != bad[0]["chosen"])
    with pytest.raises(engine_lib.EngineCheckError, match="executed action"):
        engine_lib.check_sidecar(record.xml, bad)


def test_sidecar_requires_complete_coverage(game):
    _, _, record = game
    with pytest.raises(engine_lib.EngineCheckError, match="decision coverage"):
        engine_lib.check_sidecar(record.xml, [])
    bad = _copy(record)
    del bad[len(bad) // 3]
    for n, d in enumerate(bad):
        d["k"] = n
    with pytest.raises(engine_lib.EngineCheckError, match="decision coverage"):
        engine_lib.check_sidecar(record.xml, bad)


def test_sidecar_catches_an_executed_ron_relabelled_as_a_pass(game):
    _, _, record = game
    bad = _copy(record)
    ron = next(d for d in bad if d["kind"] == "post" and d["chosen"] == engine_lib.POST_RON
               and d["resolved"]["outcome"] == "executed")
    engine_lib._normalised(ron).update(chosen=engine_lib.POST_PASS, resolved={"outcome": "passed", "event": None})
    with pytest.raises(engine_lib.EngineCheckError, match="executed action"):
        engine_lib.check_sidecar(record.xml, bad)


def test_sidecar_catches_a_call_changed_to_another_offered_call(game):
    _, _, record = game
    bad = _copy(record)
    call = next(d for d in bad if d["kind"] == "post" and d["chosen"] >= 2 and d["resolved"]["outcome"] == "executed"
                and any(a >= 2 and a != d["chosen"] for a in d["offer"]))
    engine_lib._normalised(call)["chosen"] = next(a for a in call["offer"] if a >= 2 and a != call["chosen"])
    with pytest.raises(engine_lib.EngineCheckError, match="executed action"):
        engine_lib.check_sidecar(record.xml, bad)


def test_ryuukyoku_check_derives_nagashi_and_tenpai_from_the_events():
    played = engine_lib.play_scenario(engine_lib.sc_exhaustive())
    xml = played.xml
    with pytest.raises(engine_lib.EngineCheckError, match="nagashi"):
        engine_lib.check_agari_tags(xml.replace('<RYUUKYOKU ba="0,0" sc="250,15,250,-15,250,15,250,-15"',
                                                '<RYUUKYOKU type="nm" ba="0,0" sc="250,-40,250,80,250,-20,250,-20"'))
    with pytest.raises(engine_lib.EngineCheckError, match="tenpai seats"):
        engine_lib.check_agari_tags(engine_lib.re.sub(r'(<RYUUKYOKU [^>]*?) hai2="[^"]*"', r"\1", xml))


def test_prefix_parity_on_one_real_log():
    counts = engine_lib.check_prefix_parity(engine_lib.parity_logs(1)[0])
    assert counts["parity: during"] > 0 and counts["parity: discard pending"] > 0 and counts["parity: post"] > 0
