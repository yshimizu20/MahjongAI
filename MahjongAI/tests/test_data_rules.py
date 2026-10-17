"""Data rules (T-039, T-043, T-044) on the committed gate fixtures and on synthetic event lists.
T-039: rounds_remaining / parent_rounds_remaining are clamped at 0 in an extension round.
T-043: a game with a seat that throws away AUTO_PLAY_RUN draws in a row outside riichi (no own action in between) is
rejected "Auto-played seat"; gates.auto_play_runs restates the rule on the XML.
T-044: a game with a <BYE> keeps the rounds that end before the first BYE; gates.kept_xml restates the cut."""
import os
import re
import xml.etree.ElementTree as ET

import pytest

from MahjongAI import process_xml
from MahjongAI.gameinfo import GameInfo
from MahjongAI.process_xml import AUTO_PLAY_RUN, InvalidGameException, process
from MahjongAI.utils import gates

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")


def fixture_path(fixture):
    day, name, _ = fixture
    return os.path.join(GATE_GAMES, day[:4], day, f"{day}_00", name)


def raw_of(path):
    return open(path, "rb").read().decode("utf-8", "ignore")


TONPUU, WEST = fixture_path(gates.TONPUU_SOUTH_LOG), fixture_path(gates.HANCHAN_WEST_LOG)
AUTO_PLAYED = [fixture_path(g) for g in gates.AUTO_PLAYED_LOGS]
RUN_24 = fixture_path(gates.RUN_24_LOG)
BYE_MID, BYE_BETWEEN = fixture_path(gates.BYE_MID_LOG), fixture_path(gates.BYE_BETWEEN_LOG)
BYE_ZERO = fixture_path(gates.BYE_ZERO_LOG)


# ---- T-039: clamp at 0

@pytest.mark.parametrize("path", [TONPUU, WEST], ids=["tonpuu", "west"])
def test_extension_states_clamped(path):
    """Every state of an extension round has 0 rounds left and parents [0, 0, 0, 0]; every regular round equals the
    counting definition. tonpuu: South 1-2 (seeds 4, 5; the old formula gave 0 and -1). west: the boundary case, its
    only extension round is W1 (seed 8), where the old formula already gave 0."""
    H, _ = process(path)
    xr = gates.xml_rounds(raw_of(path))
    assert len(xr) == len(H)
    extension = 0
    for r, halfturns in zip(xr, H):
        seed0, n_rounds = r["seed"][0], r["n_rounds"]
        regular = range(seed0, n_rounds)
        want = (len(regular), [sum(1 for j in regular if j % 4 == i) for i in range(4)])
        extension += seed0 >= n_rounds
        for so in gates._states([halfturns]):
            got = (so.rounds_remaining, list(so.parent_rounds_remaining))
            assert got == want, (seed0, got, want)
            assert so.rounds_remaining >= 0 and min(so.parent_rounds_remaining) >= 0
            if seed0 >= n_rounds:
                assert got == (0, [0, 0, 0, 0])
    assert extension > 0, "the fixture lost its extension round"


def test_gate_flags_unclamped_parser(monkeypatch):
    monkeypatch.setattr(GameInfo, "rounds_remaining", lambda self, r: self.n_rounds() - r)
    C = gates._count(TONPUU)
    assert C["state_xml_rounds_remaining"] > 0 and C["state_xml_parent_rounds"] > 0
    assert "state_xml_mismatch" in gates.failed_gates(C)


# ---- T-043: auto-played seat

@pytest.mark.parametrize("path", AUTO_PLAYED, ids=["absent", "late_join", "idle_25"])
def test_auto_played_rejected(path):
    with pytest.raises(InvalidGameException) as e:
        process(path)
    assert e.value.msg == "Auto-played seat"


def test_run_24_accepted():
    H, _ = process(RUN_24)
    assert H
    assert max(gates.auto_play_runs(raw_of(RUN_24))) == AUTO_PLAY_RUN - 1


def _xml(body, go=169):
    return f'<mjloggm ver="2.3"><GO type="{go}" lobby="0"/>{body}</mjloggm>'


def _init(seed0=0):
    hands = " ".join(f'hai{q}="{",".join(str(100 + 13 * q + k) for k in range(13))}"' for q in range(4))
    return f'<INIT seed="{seed0},0,0,0,0,0" ten="250,250,250,250" oya="0" {hands}/>'


def _tsumogiri(seat, n, tile=0):
    return "".join(f'<{"TUVW"[seat]}{tile + k}/><{"DEFG"[seat]}{tile + k}/>' for k in range(n))


def _tedashi(seat, tile=60):
    return f'<{"TUVW"[seat]}{tile}/><{"DEFG"[seat]}{tile + 1}/>'


def _events(xml):
    """The flat event list of _get_rounds, from an XML string (pre-order, as its traverse)."""
    return [{"event": e.tag, "attr": e.attrib} for e in ET.fromstring(xml).iter()]


def _both_rules(xml):
    """(the parser's _auto_played_seat, the gate's longest runs) of one synthetic game."""
    return process_xml._auto_played_seat(process_xml._rounds_from_events(_events(xml))), gates.auto_play_runs(xml)


EXHAUSTIVE = '<RYUUKYOKU ba="0,0" sc="250,0,250,0,250,0,250,0" hai1="1,2,3"/>'  # seat 1's tenpai hand shown
OWN_ACTIONS = {  # an action of seat 1 that ends its run
    "call": '<N who="1" m="1"/>',
    "riichi": '<REACH who="1" step="1"/>',
    "win": '<AGARI who="1" fromWho="0" ba="0,0" sc="0,0,0,0,0,0,0,0"/>' + _init(1),
    "yao9": '<RYUUKYOKU type="yao9" ba="0,0" sc="0,0,0,0,0,0,0,0" hai1="1,2,3"/>' + _init(1),
    "ron3": '<RYUUKYOKU type="ron3" ba="0,0" sc="0,0,0,0,0,0,0,0" hai1="1,2" hai2="3,4" hai3="5,6"/>' + _init(1),
    "tedashi": _tedashi(1),
}
OTHER_ACTIONS = {  # actions of other seats, and shown hands at an exhaustive draw: the run goes on
    "other_call": '<N who="2" m="1"/>',
    "other_riichi": '<REACH who="2" step="1"/><REACH who="2" step="2" ten="250,250,240,250"/>',
    "other_win": '<AGARI who="2" fromWho="0" ba="0,0" sc="0,0,0,0,0,0,0,0"/>' + _init(1),
    "exhaustive_shown_hand": EXHAUSTIVE + _init(1),
    "dora": '<DORA hai="5"/>',
}


def test_auto_played_rule_synthetic():
    n = AUTO_PLAY_RUN
    assert _both_rules(_xml(_init() + _tsumogiri(1, n - 1))) == (None, [0, n - 1, 0, 0])
    assert _both_rules(_xml(_init() + _tsumogiri(1, n))) == (1, [0, n, 0, 0])
    # discards in riichi are not counted (the declaration discard included)
    riichi = _init() + _tsumogiri(1, 10) + '<REACH who="1" step="1"/>' + _tsumogiri(1, 40, tile=20)
    assert _both_rules(_xml(riichi)) == (None, [0, 10, 0, 0])
    # the riichi ends with the round: the run counts again from the next INIT
    assert _both_rules(_xml(riichi + EXHAUSTIVE + _init(1) + _tsumogiri(1, n))) == (1, [0, n, 0, 0])
    half = n // 2 + 1
    for name, action in OWN_ACTIONS.items():
        got = _both_rules(_xml(_init() + _tsumogiri(1, half) + action + _tsumogiri(1, half, tile=20)))
        assert got == (None, [0, half, 0, 0]), name
    for name, action in OTHER_ACTIONS.items():  # across a round boundary too (win / exhaustive draw: a new INIT)
        got = _both_rules(_xml(_init() + _tsumogiri(1, half) + action + _tsumogiri(1, half, tile=20)))
        assert got == (1, [0, 2 * half, 0, 0]), name
    # a DORA between the draw and the discard: the discard is still the drawn tile; DORA is no discard
    dora_mid = _init() + "".join(f'<U{k}/><DORA hai="{k}"/><E{k}/>' for k in range(n))
    assert _both_rules(_xml(dora_mid)) == (1, [0, n, 0, 0])
    # a call clears the drawn tile: the caller's next discard is not tsumogiri
    call_then = _init() + '<U5/><N who="1" m="1"/><E5/>' + _tsumogiri(1, n - 1, tile=20)
    assert _both_rules(_xml(call_then)) == (None, [0, n - 1, 0, 0])


def test_gates_count_auto_played(capsys):
    C = gates.run(AUTO_PLAYED)
    assert (C["games"], C["skipped_invalid"], C["skipped_invalid_auto_played"], C["parse_error"]) == (0, 3, 3, 0)
    out = capsys.readouterr().out
    for p in AUTO_PLAYED:
        assert f"Auto-played seat rejected: {p}" in out


# ---- T-044: keep the rounds that end before the first BYE

@pytest.mark.parametrize("path", [BYE_MID, BYE_BETWEEN], ids=["mid", "between"])
def test_bye_keeps_finished_rounds(path):
    H, _ = process(path)
    kept, truncated = gates.kept_xml(raw_of(path))
    assert truncated and len(H) == 2 == kept.count("<INIT")


def test_bye_in_first_round_rejected():
    with pytest.raises(InvalidGameException) as e:
        process(BYE_ZERO)
    assert e.value.msg == "BYE"
    C = gates.run([BYE_ZERO])
    assert (C["games"], C["skipped_invalid"], C["skipped_invalid_bye"], C["parse_error"]) == (0, 1, 1, 0)


AGARI = '<AGARI who="0" fromWho="2" ba="0,0" sc="0,0,0,0,0,0,0,0"/>'
AGARI_2 = '<AGARI who="1" fromWho="2" ba="0,0" sc="0,0,0,0,0,0,0,0"/>'
BYE, UN = '<BYE who="3"/>', '<UN n3="x"/>'
ROUND_1 = _init(0) + _tsumogiri(0, 3) + AGARI
BYE_CASES = {  # name: (XML body, rounds kept; 0: rejected "BYE")
    "bye_after_agari": (ROUND_1 + BYE + _init(1) + _tsumogiri(0, 3) + AGARI, 1),
    "bye_inside_round": (ROUND_1 + _init(1) + _tsumogiri(0, 2) + BYE + _tsumogiri(0, 3) + AGARI, 1),
    "split_double_ron": (ROUND_1 + _init(1) + _tsumogiri(0, 2) + AGARI + BYE + UN + AGARI_2, 1),
    "split_double_ron_spaced": (ROUND_1 + _init(1) + _tsumogiri(0, 2) + AGARI + "\n" + BYE + "\n" + UN + "\n" + AGARI_2,
                                1),
    "bye_after_exhaustive": (ROUND_1 + _init(1) + _tsumogiri(0, 2) + EXHAUSTIVE + BYE + _init(2), 2),
    "bye_before_any_result": (_init(0) + _tsumogiri(0, 2) + BYE + _tsumogiri(0, 2) + AGARI, 0),
    "no_bye": (ROUND_1 + _init(1) + _tsumogiri(0, 2) + AGARI, 2),
}


def test_bye_cut_synthetic():
    """The parser's cut (_rounds_from_events) and the gate's (kept_xml) on the same synthetic games."""
    for name, (body, kept) in BYE_CASES.items():
        xml = _xml(body)
        if kept:
            assert len(process_xml._rounds_from_events(_events(xml))) == kept, name
        else:
            with pytest.raises(InvalidGameException) as e:
                process_xml._rounds_from_events(_events(xml))
            assert e.value.msg == "BYE", name
        cut, truncated = gates.kept_xml(xml)
        assert cut.count("<INIT") == kept and truncated == ("<BYE" in body), name


def test_gate_flags_parser_ignoring_bye(monkeypatch):
    """A parser that ignores the BYE keeps round 3 of BYE_MID: the gate's cut disagrees and count_game asserts."""
    orig = process_xml._rounds_from_events
    monkeypatch.setattr(process_xml, "_rounds_from_events",
                        lambda events: orig([e for e in events if e["event"] != "BYE"]))
    assert re.search(r"<BYE\b", raw_of(BYE_MID))
    with pytest.raises(AssertionError, match="2 rounds in XML, 3 parsed"):
        gates.run([BYE_MID])
