"""T-052 G2 on sample_haifu.xml and the kan-dora / kyuushu / chankan / ron3 / BYE fixture logs: every display-state
check of viewer/viewer_lib.py passes on every exported step, the self-test's corruptions raise in the check they name
(and a disabled check makes the self-test fail), the four tests/test_kan_dora.py games show the literal indicator
lists, a kakan exports its four tiles, a kyuushu hand is not tenpai, a BYE game is cut like the parser cuts it and a
rejected game raises."""
import glob
import os

import pytest

from MahjongAI.draw import Naki
from MahjongAI.process_xml import InvalidGameException
from MahjongAI.utils import gates
from MahjongAI.viewer import export, viewer_lib

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")
SAMPLE_DIR = os.path.join(os.path.dirname(GATE_GAMES), "..", "..", "data", "sample")
FIXTURES = (
    "2021010101gm-00a9-0000-6422c07e.log",  # kan dora: open-kan indicator after the discard, ron counts it
    "2021010105gm-00a9-0000-0e0fb57b.log",  # kan dora: pending flip before the second rinshan draw
    "2021010214gm-00a9-0000-7ed48ecb.log",  # kan dora: rinshan tsumo after two kakan
    "2021010823gm-00a9-0000-725fdda5.log",  # kan dora: ankan after daiminkan flips both
    "2021010100gm-00a9-0000-3ceec6a2.log",  # red kan indicator
    "2021010110gm-00a9-0000-85d17b74.log",  # kakan in ippatsu (chankan window)
    "2021010100gm-00a9-0000-c851c43f.log",  # kyuushu abort
    "2021010115gm-00a9-0000-52ecaf6b.log",  # kyuushu declined / after a call
    "2021010521gm-00a9-0000-e76b4550.log",  # triple ron
    "2021112020gm-00a9-0000-8c7c1c3f.log",  # BYE in a round
    "2021122100gm-00a9-0000-8ea24988.log",  # BYE between rounds
)


def fixture(name):
    found = glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", name))
    assert len(found) == 1, found
    return found[0]


@pytest.mark.parametrize("path", [gates.SAMPLE_LOG, os.path.join(SAMPLE_DIR, "chankan_pass_synthetic.xml")]
                         + [fixture(n) for n in FIXTURES], ids=os.path.basename)
def test_g2_checks_pass(path):
    game = viewer_lib.check_log(path, model=None)
    assert all(len(r["steps"]) > 0 for r in game["rounds"])


def test_self_test_every_corruption_raises_in_its_own_check(capsys):
    viewer_lib.self_test()
    out = capsys.readouterr().out
    assert "self-test ok" in out
    assert out.count(": caught by ") == len(viewer_lib._corruptions()) == 19
    assert {c for _, c in viewer_lib._corruptions().values()} == set(viewer_lib.CHECK_ORDER)


@pytest.mark.parametrize("disabled", viewer_lib.CHECK_ORDER)
def test_self_test_fails_when_a_check_is_disabled(monkeypatch, disabled):
    monkeypatch.setitem(viewer_lib.CHECKS, disabled, lambda ctx: None)
    with pytest.raises(AssertionError, match=f"passed check {disabled}"):
        viewer_lib.self_test()


def doras_at(game, ri, si):
    return game["rounds"][ri]["steps"][si]["doras"]


def test_open_kan_indicator_flips_after_the_discard():
    # test_kan_dora.py: round 9, discard 107 (W69 <DORA 16> G69), the ron at post 108 sees 16
    game = export.export_game(fixture(FIXTURES[0]), None)
    assert doras_at(game, 9, 106) == [2] and doras_at(game, 9, 107) == [2]
    assert doras_at(game, 9, 108) == [2, 16]
    step = game["rounds"][9]["steps"][108]
    assert step["kind"] == "post" and 69 in [e["tile"] for e in step["ponds"][3]]
    assert game["rounds"][9]["results"][0]["doras"] == [2, 16]


def test_pending_flip_before_the_second_rinshan_draw():
    game = export.export_game(fixture(FIXTURES[1]), None)
    assert doras_at(game, 1, 58) == [59, 133] and doras_at(game, 1, 59) == [59, 133]
    assert doras_at(game, 1, 60) == [59, 133, 114]


def test_rinshan_tsumo_after_two_kakan_sees_only_the_first_indicator():
    game = export.export_game(fixture(FIXTURES[2]), None)
    last = game["rounds"][7]["steps"][-1]
    assert last["kind"] == "during" and last["doras"] == [121, 31]
    assert last["decisions"][0]["actual"] == {"status": "executed", "action": {"idx": 1, "name": "tsumo"}}


def test_kakan_exports_all_four_tiles_with_called_and_added_identity():
    # 7ed48ecb round 7: <AGARI m="45587,42099"> by seat 1 after two kakan; the result and the table agree
    game = export.export_game(fixture(FIXTURES[2]), None)
    rnd = game["rounds"][7]
    kakans = [{"kind": "kakan", "tiles": [116, 117, 118, 119], "called": 119, "from": 0, "added": 116},
              {"kind": "kakan", "tiles": [108, 109, 110, 111], "called": 109, "from": 0, "added": 111}]
    assert rnd["results"][0]["melds"] == kakans
    assert sorted(rnd["steps"][-1]["melds"][1], key=lambda m: m["tiles"]) == sorted(kakans, key=lambda m: m["tiles"])
    assert export.kakan_parts(Naki(45587)) == ([117, 118], 119, 116)


def test_kyuushu_hand_is_revealed_not_tenpai():
    # c851c43f round 1 (index 0): <RYUUKYOKU type="yao9" hai0=...> is the kyuushu hand, three shanten
    game = export.export_game(fixture(FIXTURES[6]), None)
    draws = [r["results"][0] for r in game["rounds"] if r["results"] and r["results"][0]["type"] == "ryuukyoku"]
    assert (draws[0]["reason"], draws[0]["tenpai"], draws[0]["revealed"]) == ("yao9", None, [0])
    assert (draws[1]["reason"], draws[1]["tenpai"], draws[1]["revealed"]) == ("exhaustive", [0], [0])
    assert draws[3]["tenpai"] == [1, 3]


def test_ankan_after_daiminkan_flips_both_at_once():
    game = export.export_game(fixture(FIXTURES[3]), None)
    assert doras_at(game, 3, 98) == [68, 20, 123] and doras_at(game, 3, 99) == [68, 20, 123]


def test_bye_game_is_cut_like_the_parser():
    game = export.export_game(fixture(FIXTURES[9]), None)
    assert not game["complete"] and game["rounds_dropped"] > 0 and len(game["rounds"]) == 2
    with pytest.raises(InvalidGameException):
        export.export_game(fixture("2021010100gm-00a9-0000-1500bf1f.log"), None)  # BYE in round 1: 0 rounds kept


def test_action_names():
    assert [export.post_action_name(i) for i in (0, 1, 2, 4, 23, 65, 83, 110, 113, 119, 120, 153)] == [
        "pass", "ron", "chi 1-3m (1m)", "chi 1-3m (3m)", "chi 1-3p (1p)", "chi 3-5m red (3m)", "pon 1m",
        "pon 5m red", "pon 東", "pon 中", "kan 1m", "kan 中"]
    assert [export.during_action_name(i) for i in (0, 1, 2, 3, 37, 71)] == [
        "pass", "tsumo", "riichi", "ankan 1m", "kakan 1m", "kyuushu"]
    assert export.tile_name(16) == "5mr" and export.tile_name(17) == "5m" and export.tile_name(135) == "中"
