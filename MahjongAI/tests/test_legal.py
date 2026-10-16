"""Unit tests for MahjongAI.utils.legal: riichi-declaration tenpai, kuikae, the 37-slot red-five mask, and the
option rules (riichi points / wall / suukaikan, kan wall / cap, riichi ankan, calls, kuikae dead end), and
kyuushu kyuuhai.
Hands are written in Tenhou notation: 0m / 0p / 0s is a red five (it counts in its kind AND its red slot)."""
import numpy as np
import pytest

from MahjongAI.draw import Naki
from MahjongAI.turn import DiscardTurn
from MahjongAI.utils.legal import (
    call_leaves_legal_discard,
    calls_allowed,
    kan_allowed,
    kans_on_table,
    kyuushu_legal,
    riichi_allowed,
    riichi_ankan_allowed,
    wait_kinds,
    kuikae_forbidden_kinds,
    legal_discard_kinds,
    legal_discard_mask,
    presence_mask,
    tenpai_keeping_kinds,
)

SUITS = "mpsz"
RED = {0: 34, 1: 35, 2: 36}


def hand(text: str) -> np.ndarray:
    """'123m456p0s11z' -> (37,) counts."""
    counts = np.zeros(37, dtype=np.float32)
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
            continue
        color = SUITS.index(ch)
        for d in digits:
            n = int(d)
            if n == 0:
                counts[9 * color + 4] += 1
                counts[RED[color]] += 1
            else:
                counts[9 * color + n - 1] += 1
        digits = ""
    return counts


def kinds(text: str) -> set:
    return set(np.flatnonzero(hand(text)[:34]).tolist())


# ---- riichi declaration: tenpai-keeping discards

def test_single_tenpai_discard():
    assert tenpai_keeping_kinds(hand("123m456p789s34s11z" + "9p")) == kinds("9p")


def test_chiitoitsu():
    assert tenpai_keeping_kinds(hand("11m99p11s99s1122z35z")) == kinds("35z")


def test_standard_and_chiitoitsu_shapes():
    assert tenpai_keeping_kinds(hand("11223344556678m")) == kinds("12345678m")


def test_junkara_wait_is_not_tenpai():
    # discarding 9m leaves a 1m-only wait on four concealed 1m: illegal; discarding 1m leaves a 9m wait
    assert tenpai_keeping_kinds(hand("11119m234567p789s")) == kinds("1m")


def test_riichi_with_ankan():
    # 11 concealed tiles (an ankan is down): 123m 456p 34s 99s + 7z -> only 7z keeps the 2s/5s wait
    assert tenpai_keeping_kinds(hand("123m456p3499s7z")) == kinds("7z")


def test_kokushi_13_sided_every_discard_keeps_tenpai():
    h = hand("119m19p19s1234567z")
    assert tenpai_keeping_kinds(h) == kinds("19m19p19s1234567z")


def test_kokushi_single_wait():
    assert tenpai_keeping_kinds(hand("1159m19p19s123456z")) == kinds("5m")


def test_riichi_declaration_restricts_kinds_only_when_declaring():
    h = hand("123m456p789s34s11z" + "9p")
    assert legal_discard_kinds(h, False, None) == kinds("123m4569p789s34s1z")
    assert legal_discard_kinds(h, True, None) == kinds("9p")


# ---- kuikae

@pytest.mark.parametrize("color", [0, 1, 2])
@pytest.mark.parametrize(
    "number, which, forbidden",
    [
        (2, 0, {2, 5}),  # hold 45, chi 3 (run 345, low end): 3 and 6
        (3, 2, {5, 2}),  # hold 45, chi 6 (run 456, high end): 6 and 3
        (2, 1, {3}),  # hold 35, chi 4 (kanchan): 4 only
        (0, 2, {2}),  # hold 12, chi 3: 3 only (no rank 0)
        (6, 0, {6}),  # hold 89, chi 7: 7 only (no rank 10)
        (0, 0, {0, 3}),  # hold 23, chi 1: 1 and 4
        (6, 2, {8, 5}),  # hold 78, chi 9: 9 and 6
        (0, 1, {1}),  # hold 13, chi 2
    ],
)
def test_chi_kuikae(color, number, which, forbidden):
    call = Naki.from_chi_info(color, number, which, False)
    assert kuikae_forbidden_kinds(call) == {9 * color + k for k in forbidden}


def test_pon_kuikae_plain_and_honor():
    assert kuikae_forbidden_kinds(Naki.from_pon_info(4, 0, True, 1)) == {4}
    assert kuikae_forbidden_kinds(Naki.from_pon_info(27, 0, False, 2)) == {27}


@pytest.mark.parametrize(
    "call",
    [Naki.from_minkan_info(4, 1, 0), Naki.from_kakan_info(13, 0), Naki.from_ankan_info(27)],
)
def test_kan_has_no_kuikae(call):
    assert kuikae_forbidden_kinds(call) == set()


# ---- 37-slot mask: red fives

def test_red_only_five_stays_legal_on_its_red_slot():
    m = legal_discard_mask(hand("0m123p456p789s11z"), False, None)
    assert (m[4], m[34]) == (0.0, 1.0)


def test_plain_and_red_five_both_legal():
    m = legal_discard_mask(hand("50m123p456p789s1z"), False, None)
    assert (m[4], m[34]) == (1.0, 1.0)


def test_forbidden_five_clears_both_slots():
    # chi 2m with 34m (run 234, low end): 2m and 5m forbidden; the hand after the call holds 5m and 0m
    m = legal_discard_mask(hand("50m123p456p78s"), False, Naki.from_chi_info(0, 1, 0, False))
    assert (m[4], m[34]) == (0.0, 0.0)
    assert m[9] == 1.0  # 1p: not touched


def test_pon_of_plain_five_holding_red():
    # pon 5p while holding a third copy, the red one: both slots off
    m = legal_discard_mask(hand("0p123m456s789s1z"), False, Naki.from_pon_info(13, 0, False, 2))
    assert (m[13], m[35]) == (0.0, 0.0)


def test_honor_pon_mask():
    m = legal_discard_mask(hand("1z123m456p789s"), False, Naki.from_pon_info(27, 0, False, 2))
    assert m[27] == 0.0 and m[0] == 1.0


def test_unrestricted_mask_equals_presence():
    h = hand("120m50p456s1234z")
    assert np.array_equal(legal_discard_mask(h, False, None), presence_mask(h))


def test_riichi_mask_clears_red_slot_of_illegal_five():
    # 123m 406p 789s 34s 11z + 9p: only 9p keeps tenpai; the red 5p (slot 35) goes off with its kind
    m = legal_discard_mask(hand("123m406p789s34s11z9p"), True, None)
    assert set(np.flatnonzero(m).tolist()) == {17}


# ---- DiscardTurn invariants

def test_discard_turn_rejects_kan_as_after_call():
    with pytest.raises(AssertionError):
        DiscardTurn(0, None, 0, 0, after_call=Naki.from_ankan_info(27))


def test_discard_turn_rejects_riichi_after_call():
    with pytest.raises(AssertionError):
        DiscardTurn(0, None, 0, 0, is_riichi_declaration=True, after_call=Naki.from_chi_info(0, 0, 0, False))


def test_discard_turn_accepts_chi_and_pon():
    for call in (Naki.from_chi_info(0, 0, 0, False), Naki.from_pon_info(27, 0, False, 2)):
        assert DiscardTurn(0, None, 0, 0, after_call=call).after_call is call


# ---- option rules (T-032)

def kind(text: str) -> int:
    (k,) = kinds(text)
    return k


@pytest.mark.parametrize("score, draws, n_kans, n_owners, ok", [
    (10, 4, 0, 0, True),  # exactly 1,000 points and exactly one own draw left: legal
    (9, 4, 0, 0, False),
    (10, 3, 0, 0, False),
    (10, 4, 4, 2, False),  # right after a 4th kan by two players: the round ends with the next discard
    (10, 4, 4, 1, True),  # four kans by one player: no abort, play goes on
    (10, 4, 3, 2, True),
])
def test_riichi_allowed(score, draws, n_kans, n_owners, ok):
    assert riichi_allowed(score, draws, n_kans, n_owners) is ok


@pytest.mark.parametrize("draws, n_kans, ok", [(1, 3, True), (0, 0, False), (1, 4, False)])
def test_kan_allowed(draws, n_kans, ok):
    assert kan_allowed(draws, n_kans) is ok


@pytest.mark.parametrize("draws, n_kans, n_owners, ok", [
    (1, 0, 0, True),
    (0, 0, 0, False),  # the last discard (houtei)
    (5, 4, 2, False),  # suukaikan pending
    (5, 4, 1, True),  # chi / pon stay legal; daiminkan is refused by kan_allowed
])
def test_calls_allowed(draws, n_kans, n_owners, ok):
    assert calls_allowed(draws, n_kans, n_owners) is ok


def test_kans_on_table():
    melds = [[Naki.from_ankan_info(0), Naki.from_pon_info(9, 0, False, 1)], [], [Naki.from_minkan_info(27, 1, 0)], []]
    assert kans_on_table(melds) == (2, 2)
    assert kans_on_table([[Naki.from_ankan_info(k) for k in range(4)], [], [], []]) == (4, 1)


def test_riichi_okurikan_refused():
    # 3333m held from before the riichi, 1z drawn: a kan of any kind but the drawn one is refused (okurikan)
    assert not riichi_ankan_allowed(hand("3333m123p456p789s1z"), kind("1z"), kind("3m"), [])


def test_riichi_ankan_that_changes_the_wait_is_refused():
    before = hand("3335m123p456p789s")  # 33 + 35 waits 4m, 333 + 5 waits 5m
    assert wait_kinds(before, []) == kinds("45m")
    assert not riichi_ankan_allowed(hand("33335m123p456p789s"), kind("3m"), kind("3m"), [])


def test_riichi_ankan_same_tanki_wait_is_allowed():
    assert riichi_ankan_allowed(hand("1111m234p567p789s9m"), kind("1m"), kind("1m"), [])


def test_riichi_ankan_that_changes_the_decomposition_but_not_the_wait_is_allowed():
    # 1112444m: 2m by 111 + 22 + 444, 3m by 111 + 234 + 44 (4m in a run); after the kan, 1112m waits 2m / 3m too
    assert wait_kinds(hand("1112444m123p456p"), []) == kinds("23m")
    assert riichi_ankan_allowed(hand("11124444m123p456p"), kind("4m"), kind("4m"), [])


def test_riichi_ankan_fifth_copy_wait_regression():
    # Codex review: 4455556678999m; an unfiltered shanten oracle would add 5m (all four 5m concealed) before the
    # kan only. With the junkara filter both sides wait 3m 4m 6m 7m 9m, so the kan of the drawn 9m is legal.
    assert wait_kinds(hand("4455556678999m"), []) == kinds("34679m")
    assert riichi_ankan_allowed(hand("4455556678999m9m"), kind("9m"), kind("9m"), [])


def test_riichi_ankan_with_an_earlier_ankan():
    kan = [Naki.from_ankan_info(kind("1z"))]
    assert riichi_ankan_allowed(hand("1111m234p567p9m"), kind("1m"), kind("1m"), kan)
    assert not riichi_ankan_allowed(hand("33335m123p456p"), kind("3m"), kind("3m"), kan)


def test_call_leaving_only_kuikae_kinds_is_refused():
    # three melds down, 3445m held: chi 4m with 35m leaves 44m, both forbidden (called kind 4m)
    chi = Naki.from_chi_info(0, 2, 1, False)  # run 345m, called 4m (the middle)
    assert sorted(kuikae_forbidden_kinds(chi)) == [kind("4m")]
    assert not call_leaves_legal_discard(hand("3445m"), chi)
    assert call_leaves_legal_discard(hand("3445m1z"), chi)


def test_pon_always_leaves_a_legal_discard():
    # a pon dead end would need five copies of one kind: with 111z2z, a pon of 1z leaves 1z (forbidden) and 2z
    pon = Naki.from_pon_info(kind("1z"), 0, False, 1)
    assert call_leaves_legal_discard(hand("111z2z"), pon)


def test_kyuushu_nine_kinds_on_the_first_draw():
    h = hand("19m19p1s1234z23456m")  # 1m 9m 1p 9p 1s E S W N: 9 kinds, 14 tiles
    assert kyuushu_legal(h[:34], True)
    assert not kyuushu_legal(h[:34], False)  # not the first draw, or a call came before it


def test_kyuushu_eight_kinds_is_not_enough():
    assert not kyuushu_legal(hand("19m19p1s123z234566m")[:34], True)


def test_kyuushu_counts_distinct_kinds_not_tiles():
    # 11m 99m 11p 99p 11s 11z 22z: 14 tiles, all terminal / honor, but 7 kinds
    assert not kyuushu_legal(hand("1199m1199p11s1122z")[:34], True)
    assert kyuushu_legal(hand("1199m1199p15s1234z")[:34], True)  # 9 kinds once the pairs open up


def test_kyuushu_ignores_red_fives_and_middle_tiles():
    assert not kyuushu_legal(hand("19m19p1s123z0m0p0556s")[:34], True)  # 8 kinds; a red five is a five
