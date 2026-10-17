"""Unit tests for MahjongAI.utils.agari: the yaku check, the win-context flags, waits and furiten (T-033).
Hands are written in Tenhou notation (1z-7z = E S W N haku hatsu chun); counts are 34-kind vectors."""
import numpy as np
import pytest
from mahjong.constants import EAST, NORTH, SOUTH, WEST

from MahjongAI.decision import AgariDecision
from MahjongAI.draw import Naki
from MahjongAI.process_xml import _mark_ron3_executed
from MahjongAI.turn import PostTurn
from MahjongAI.utils.agari import (
    FURITEN_DISCARD,
    FURITEN_RIICHI,
    FURITEN_TEMP,
    NO_YAKU,
    FuritenTracker,
    WinContext,
    WinRuleError,
    evaluate_ron,
    evaluate_tsumo,
    is_complete,
    round_wind,
    seat_wind,
    waits,
    win_yaku,
)
from MahjongAI.utils.constants import DECISION_AGARI_IDX

SUITS = "mpsz"


def hand(text: str) -> np.ndarray:
    """'123m456p11z' -> (34,) counts."""
    counts = np.zeros(34, dtype=int)
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
            continue
        for d in digits:
            counts[9 * SUITS.index(ch) + int(d) - 1] += 1
        digits = ""
    return counts


def kind(text: str) -> int:
    return int(np.flatnonzero(hand(text))[0])


def chi(text: str) -> Naki:  # lowest tile of the run, e.g. chi("1m") = 123m
    k = kind(text)
    return Naki.from_chi_info(k // 9, k % 9, 0, False)


def pon(text: str) -> Naki:
    return Naki.from_pon_info(kind(text), 0, False, 1)


def ankan(text: str) -> Naki:
    return Naki.from_ankan_info(kind(text))


def ron_ctx(seat=SOUTH, rnd=EAST, **flags) -> WinContext:
    return WinContext(is_tsumo=False, seat_wind=seat, round_wind=rnd, **flags)


def tsumo_ctx(seat=SOUTH, rnd=EAST, **flags) -> WinContext:
    return WinContext(is_tsumo=True, seat_wind=seat, round_wind=rnd, **flags)


# ---- the yaku check: open hand chi 123m, pon 9p, 456s 78s 55z (haku pair) waiting on 3s / 6s / 9s

OPEN_NO_YAKU = (hand("456s78s55z"), [chi("1m"), pon("9p")])


def with_tile(concealed, text):
    out = concealed.copy()
    out[kind(text)] += 1
    return out


def test_open_hand_without_yaku():
    concealed, melds = OPEN_NO_YAKU
    assert waits(concealed) == {kind("3s"), kind("6s"), kind("9s")}
    assert win_yaku(with_tile(concealed, "6s"), melds, kind("6s"), ron_ctx()) is None
    assert win_yaku(with_tile(concealed, "9s"), melds, kind("9s"), tsumo_ctx()) is None


@pytest.mark.parametrize("ctx, expected", [
    (tsumo_ctx(is_rinshan=True), ("Rinshan",)),
    (tsumo_ctx(is_haitei=True), ("Haitei",)),
    (ron_ctx(is_houtei=True), ("Houtei",)),
    (ron_ctx(is_chankan=True), ("Chankan",)),
])
def test_each_situational_flag_alone_gives_a_yaku(ctx, expected):
    concealed, melds = OPEN_NO_YAKU
    assert win_yaku(with_tile(concealed, "6s"), melds, kind("6s"), ctx) == expected


def test_library_errors_raise_instead_of_hiding_an_offer():
    concealed, melds = OPEN_NO_YAKU
    with pytest.raises(WinRuleError, match="haitei_with_rinshan"):
        win_yaku(with_tile(concealed, "6s"), melds, kind("6s"), tsumo_ctx(is_rinshan=True, is_haitei=True))
    with pytest.raises(WinRuleError, match="hand_not_winning"):
        win_yaku(with_tile(concealed, "1s"), melds, kind("1s"), ron_ctx(is_houtei=True))


# ---- winds: chi 234m, pon of a wind, 456p 789s 9m waiting on 9m (tanki)

def wind_hand(wind_text):
    return hand("456p789s99m"), [chi("2m"), pon(wind_text)]


def test_seat_wind_only_for_its_seat():
    concealed, melds = wind_hand("2z")  # a South pon in the East round
    assert win_yaku(concealed, melds, kind("9m"), ron_ctx(seat=SOUTH)) == ("SeatWind:S",)
    assert win_yaku(concealed, melds, kind("9m"), ron_ctx(seat=WEST)) is None
    assert win_yaku(concealed, melds, kind("9m"), ron_ctx(seat=EAST)) is None


def test_double_east_for_the_dealer_in_the_east_round():
    concealed, melds = wind_hand("1z")
    assert win_yaku(concealed, melds, kind("9m"), ron_ctx(seat=EAST)) == ("RoundWind:E", "SeatWind:E")
    assert win_yaku(concealed, melds, kind("9m"), ron_ctx(seat=NORTH)) == ("RoundWind:E",)
    assert win_yaku(concealed, melds, kind("9m"), ron_ctx(seat=NORTH, rnd=SOUTH)) is None


def test_west_round_wind():  # 西入: the library's RoundWindWest class
    concealed, melds = wind_hand("3z")
    assert win_yaku(concealed, melds, kind("9m"), ron_ctx(seat=SOUTH, rnd=WEST)) == ("RoundWind:W",)
    assert win_yaku(concealed, melds, kind("9m"), ron_ctx(seat=WEST, rnd=WEST)) == ("RoundWind:W", "SeatWind:W")
    assert win_yaku(concealed, melds, kind("9m"), ron_ctx(seat=SOUTH, rnd=SOUTH)) is None


def test_north_seat_wind():  # the library's SeatWindNorth class
    concealed, melds = wind_hand("4z")
    assert win_yaku(concealed, melds, kind("9m"), ron_ctx(seat=NORTH, rnd=SOUTH)) == ("SeatWind:N",)
    assert win_yaku(concealed, melds, kind("9m"), ron_ctx(seat=WEST, rnd=SOUTH)) is None


def test_wind_helpers():
    assert [seat_wind(p, oya=2) for p in range(4)] == [WEST, NORTH, EAST, SOUTH]
    assert [round_wind(r) for r in (0, 3, 4, 7, 8, 11)] == [EAST, EAST, SOUTH, SOUTH, WEST, WEST]


# ---- closed hand: 234m 567p 789s 11z 34s, East pair (round wind: no pinfu), waiting on 2s / 5s

CLOSED = hand("234m567p789s11z34s")


def test_closed_ron_without_yaku_but_tsumo_has_menzen_tsumo():
    won = with_tile(CLOSED, "2s")
    assert win_yaku(won, [], kind("2s"), ron_ctx()) is None
    assert win_yaku(won, [], kind("2s"), tsumo_ctx()) == ("Tsumo",)
    assert win_yaku(won, [], kind("2s"), ron_ctx(is_riichi=True)) == ("Riichi",)


def test_tenhou_dealer_only_chiihou_non_dealer_only():
    won = with_tile(CLOSED, "2s")
    assert win_yaku(won, [], kind("2s"), tsumo_ctx(seat=EAST, is_tenhou=True)) == ("Tenhou",)
    assert win_yaku(won, [], kind("2s"), tsumo_ctx(seat=SOUTH, is_chiihou=True)) == ("Chiihou",)
    with pytest.raises(WinRuleError, match="tenhou_not_as_dealer"):
        win_yaku(won, [], kind("2s"), tsumo_ctx(seat=SOUTH, is_tenhou=True))
    with pytest.raises(WinRuleError, match="chiihou_as_dealer"):
        win_yaku(won, [], kind("2s"), tsumo_ctx(seat=EAST, is_chiihou=True))


def test_ankan_keeps_the_hand_closed():
    concealed = hand("234m567p11z34s")  # + ankan 9s
    won = with_tile(concealed, "2s")
    assert win_yaku(won, [ankan("9s")], kind("2s"), tsumo_ctx()) == ("Tsumo",)
    assert win_yaku(won, [ankan("9s")], kind("2s"), ron_ctx()) is None


# ---- shapes

def test_chiitoitsu_cannot_use_four_of_a_kind():
    assert not is_complete(hand("111m11p11s11z22z33z"), kind("1m"))
    assert is_complete(hand("11m11p11s11z22z33z5z"), kind("5z"))


def test_waits_never_count_a_fifth_copy():
    concealed = hand("1111345678999m")
    assert kind("1m") not in waits(concealed)
    assert waits(concealed) == {kind("2m")}


# ---- evaluate_tsumo / evaluate_ron

def test_evaluate_tsumo_reports_no_yaku():
    concealed, melds = OPEN_NO_YAKU
    decisions, blocked = evaluate_tsumo(1, with_tile(concealed, "6s"), melds, kind("6s"), tsumo_ctx())
    assert decisions == [] and blocked == NO_YAKU
    decisions, blocked = evaluate_tsumo(1, with_tile(concealed, "6s"), melds, kind("6s"), tsumo_ctx(is_rinshan=True))
    assert [d.yaku for d in decisions] == [("Rinshan",)] and blocked is None
    decisions, blocked = evaluate_tsumo(1, with_tile(concealed, "1s"), melds, kind("1s"), tsumo_ctx())
    assert decisions == [] and blocked is None  # not complete: nothing to block


def four_hands(p, concealed):
    hands = [hand("1112223334445z")] * 4  # the other seats wait on 5z only
    hands[p] = concealed
    return hands


def make_tracker(hands):
    tracker = FuritenTracker()
    for p, h in enumerate(hands):
        tracker.set_hand(p, h)
    return tracker


def test_evaluate_ron_offers_and_blocks():
    hands = four_hands(1, with_tile(CLOSED, "2s") - hand("2s"))
    ctxs = {p: ron_ctx(is_riichi=(p == 1)) for p in (1, 2, 3)}
    tracker = make_tracker(hands)
    rons, blocked = evaluate_ron(0, kind("5s"), hands, [[]] * 4, tracker, ctxs)
    assert isinstance(rons[1], AgariDecision) and rons[1].yaku == ("Riichi",) and blocked == {}
    rons, blocked = evaluate_ron(0, kind("5s"), hands, [[]] * 4, tracker, {p: ron_ctx() for p in (1, 2, 3)})
    assert rons[1] is None and blocked == {1: NO_YAKU}
    tracker.on_discard(1, kind("2s"))
    rons, blocked = evaluate_ron(0, kind("5s"), hands, [[]] * 4, tracker, ctxs)
    assert rons[1] is None and blocked == {1: FURITEN_DISCARD}


# ---- FuritenTracker: 23m 456p 789s 111z 55z waits on 1m / 4m

WAIT_14M = hand("23m456p789s111z55z")


def test_discard_furiten_on_the_other_wait():
    tracker = make_tracker(four_hands(1, WAIT_14M))
    assert tracker.waits(1) == {kind("1m"), kind("4m")}
    tracker.on_discard(1, kind("1m"))  # also when 1m was called away: the tracker never forgets a discard
    assert tracker.reason(1) == FURITEN_DISCARD


def test_temp_furiten_from_a_pass_cleared_only_by_own_discard():
    hands = four_hands(1, WAIT_14M)
    tracker = make_tracker(hands)
    tracker.on_offer_passed(kind("9p"), [0, 1, 2], [0] * 4)  # not a wait: nothing
    assert tracker.reason(1) is None
    tracker.on_offer_passed(kind("1m"), [0, 1, 2], [0] * 4)  # a passed wait (a no-yaku wait or a chankan alike)
    assert tracker.reason(1) == FURITEN_TEMP
    tracker.on_offer_passed(kind("9p"), [0, 1, 2], [0] * 4)  # play goes on, another player calls: still furiten
    assert tracker.reason(1) == FURITEN_TEMP
    tracker.on_discard(1, kind("5z"))  # own discard (also one right after an own chi / pon)
    tracker.set_hand(1, hands[1])
    assert tracker.reason(1) is None


def test_riichi_furiten_survives_own_discard():
    hands = four_hands(1, WAIT_14M)
    tracker = make_tracker(hands)
    reaches = [0, 1, 0, 0]
    tracker.on_offer_passed(kind("4m"), [0, 1, 2], reaches)
    assert tracker.reason(1) == FURITEN_RIICHI
    tracker.on_discard(1, kind("9s"))
    tracker.set_hand(1, hands[1])
    assert tracker.reason(1) == FURITEN_RIICHI


def test_temp_furiten_only_for_the_responders():
    tracker = make_tracker(four_hands(1, WAIT_14M))
    tracker.on_offer_passed(kind("1m"), [0, 2, 3], [0] * 4)  # player 1 discarded it: not a pass
    assert tracker.reason(1) is None


def test_furiten_never_blocks_a_tsumo():
    decisions, blocked = evaluate_tsumo(1, with_tile(CLOSED, "2s"), [], kind("2s"), tsumo_ctx(is_riichi=True))
    assert [d.yaku for d in decisions] == [("Riichi", "Tsumo")]  # evaluate_tsumo has no furiten input at all


# ---- triple ron: the final PostTurn must hold the three offers

def post_turn(player, offered):
    decisions = [[[] for _ in range(3)] for _ in range(4)]
    for p in offered:
        decisions[p][DECISION_AGARI_IDX].append(AgariDecision(p, False))
    return PostTurn(player, None, decisions, 0, {})


def test_ron3_takes_the_final_post_turn():
    good = [post_turn(1, []), post_turn(0, [1, 2, 3])]
    _mark_ron3_executed(good)
    assert all(good[-1].decisions[p][DECISION_AGARI_IDX][0].executed for p in (1, 2, 3))
    with pytest.raises(AssertionError):
        _mark_ron3_executed([post_turn(0, [1, 2, 3]), post_turn(1, [])])


# ---- the parser's call sites: a passed chankan (synthetic game, no clean real-log case exists)

def test_parser_passed_chankan_gives_same_turn_furiten():
    import os
    from MahjongAI.process_xml import process
    game = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "chankan_pass_synthetic.xml")
    H, _ = process(game)
    posts = [t for t in H[0] if isinstance(t, PostTurn)]
    chankan, rinshan_discard, own_discard, win = posts[-4:]  # seat 0: kakan 3m, rinshan discard 6m; seat 1: 3s; seat 2: 6m
    assert [d.yaku for d in chankan.decisions[1][DECISION_AGARI_IDX]] == [("Chankan", "Ittsu", "Pinfu")]
    assert rinshan_discard.decisions[1][DECISION_AGARI_IDX] == [] and rinshan_discard.ron_blocked == {1: FURITEN_TEMP}
    assert own_discard.player == 1  # seat 1's own discard clears the same-turn furiten: the next 6m is a ron
    assert [(d.yaku, d.executed) for d in win.decisions[1][DECISION_AGARI_IDX]] == [(("Ittsu", "Pinfu"), True)]
