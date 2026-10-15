"""Unit tests for MahjongAI.utils.legal: riichi-declaration tenpai, kuikae, and the 37-slot red-five mask.
Hands are written in Tenhou notation: 0m / 0p / 0s is a red five (it counts in its kind AND its red slot)."""
import sys

sys.path.append("../../")

import numpy as np
import pytest

from MahjongAI.draw import Naki
from MahjongAI.turn import DiscardTurn
from MahjongAI.utils.legal import (
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
