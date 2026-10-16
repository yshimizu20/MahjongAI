"""Kan-dora timing on real games (T-035). Tenhou flips an ankan indicator at once and a daiminkan / kakan one only
after the kan player's discard, or right before his next rinshan draw (「明槓/加槓は後めくり」). The XML writes the
open-kan DORA early (`N T DORA D`), so the parser holds it back. Expected indicator lists are literals read off the
XML and the <AGARI doraHai> of each fixture in tests/data/gate_games/, not the gate oracle."""
import glob
import os

from MahjongAI.process_xml import process
from MahjongAI.turn import DiscardTurn, DuringTurn, PostTurn
from MahjongAI.utils.constants import DECISION_AGARI_IDX, EventStateTypes

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")


def parse(name):
    found = glob.glob(os.path.join(GATE_GAMES, "2021", "*", "*", name))
    assert len(found) == 1, f"fixture {name} missing under {GATE_GAMES}"
    return process(found[0])


def is_new_dora(tok):
    return (tok >> 13) & 7 == EventStateTypes.NEW_DORA


def kan_discard(halfturns, i):
    """(draw, discard, post) halfturns around the discard at index i."""
    draw, disc, post = halfturns[i - 1], halfturns[i], halfturns[i + 1]
    assert isinstance(draw, DuringTurn) and isinstance(disc, DiscardTurn) and isinstance(post, PostTurn)
    assert draw.player == disc.player == post.player
    return draw, disc, post


def test_open_kan_indicator_after_the_discard_and_ron_counts_it():
    # round 9: seat 3 kakans, `W69 <DORA hai="16" /> G69` (16 = the red 5m; tsumogiri), seat 0 rons the 69
    H, T = parse("2021010101gm-00a9-0000-6422c07e.log")
    halfturns, tokens = H[9], T[9]
    draw, disc, post = kan_discard(halfturns, 107)
    assert disc.discarded_tile == 69
    assert draw.stateObj.dora == [2] and disc.stateObj.dora == [2]  # the discarder never sees 16
    assert {p: rs.dora for p, rs in post.responder_states.items()} == {0: [2, 16], 1: [2, 16], 2: [2, 16]}
    # unseen counts: the discarder's red-5m slot is unchanged since the draw; every responder sees the red indicator
    assert disc.stateObj.remaining_tiles_pov[[4, 34]].tolist() == draw.stateObj.remaining_tiles_pov[[4, 34]].tolist()
    assert all(rs.remaining_tiles_pov[34] == 0 for rs in post.responder_states.values())
    # history: the discard sample stops before the indicator token; the responders see `NEW_DORA, discard`
    assert is_new_dora(tokens[disc.encoding_idx]) and tokens[disc.encoding_idx] & 0x7F == 4  # kind index of 5m
    assert not any(is_new_dora(t) for t in tokens[draw.encoding_idx: disc.encoding_idx])
    assert post.encoding_idx == disc.encoding_idx + 2
    # the ron: Tenhou's doraHai="2,16" == the winner's indicators
    assert halfturns[-1] is post and post.decisions[0][DECISION_AGARI_IDX][0].executed


def test_pending_flip_before_the_second_rinshan_draw():
    # round 1: seat 2 kakans, draws, kakans again (XML `N DORA hai=133 V17`), then `DORA hai=114` before `F73`
    H, _ = parse("2021010105gm-00a9-0000-0e0fb57b.log")
    draw, disc, post = kan_discard(H[1], 59)
    assert disc.discarded_tile == 73
    assert draw.stateObj.dora == [59, 133]  # the 1st kakan's indicator flipped before the 2nd rinshan draw
    assert disc.stateObj.dora == [59, 133]  # the 2nd kakan's is still face down
    assert all(rs.dora == [59, 133, 114] for rs in post.responder_states.values())


def test_rinshan_tsumo_after_two_kakan_sees_only_the_first_indicator():
    # round 7: seat 1 kakans twice, `N DORA hai=31 U82 AGARI doraHai="121,31"`: a rinshan tsumo
    H, _ = parse("2021010214gm-00a9-0000-7ed48ecb.log")
    last = H[7][-1]
    assert isinstance(last, DuringTurn) and last.decisions[DECISION_AGARI_IDX][0].executed
    assert last.stateObj.dora == [121, 31]


def test_ankan_after_daiminkan_flips_both_at_once():
    # round 3: seat 2 daiminkans, draws `V84`, ankans: `N DORA hai=20 DORA hai=123 V110 F110`
    H, T = parse("2021010823gm-00a9-0000-725fdda5.log")
    draw, disc, _ = kan_discard(H[3], 99)
    assert disc.discarded_tile == 110
    assert draw.stateObj.dora == [68, 20, 123] and disc.stateObj.dora == [68, 20, 123]
    assert not is_new_dora(T[3][disc.encoding_idx])  # nothing held back
