"""T-026: decision.decision_mask (the post-turn call offers) against legal_oracles.call_offers_by_tiles, which derives
the offers from every pair of PHYSICAL tiles, plus named hands. T-008: every offered code's called tile has the
discard's kind and red-ness (a red daiminkan discard used to give a code that calls a plain five)."""
import collections
import random

import numpy as np

from MahjongAI.decision import decision_mask
from MahjongAI.utils import legal_oracles

RED = legal_oracles.RED_IDS


def _tensors(seat_ids):
    """{seat: 136-ids} -> the parser's (4, 37) hand counts (34 kinds + one red flag per suit)."""
    H = np.zeros((4, 37), np.float32)
    for p, ids in seat_ids.items():
        for i in ids:
            H[p][i // 4] += 1
            if i in RED:
                H[p][34 + i // 36] = 1
    return H


def _offers(seat_ids, discard, discarder=0):
    out = decision_mask(discarder, _tensors(seat_ids), discard // 4, discard in RED)
    return {p: [d.naki for d in out[p]] for p in range(4)}


def _indices(nakis):
    return sorted(n.get_post_turn_filter_idx() for n in nakis)


def _called(nakis):
    return {legal_oracles.decode_meld(n.naki_code).called for n in nakis}


def test_mask_equals_offers_by_tiles():
    rng = random.Random(0)
    checks, seen = 0, collections.Counter()
    for _ in range(3000):
        k = rng.randrange(34)
        discard = 4 * k + rng.randrange(4)
        near = [i for i in range(136) if i != discard and abs(i // 4 - k) <= 2 and (i // 4 >= 27) == (k >= 27)
                and (k >= 27 or i // 36 == k // 9)]
        rest = [i for i in range(136) if i != discard and i not in near]
        m = rng.randint(2, min(8, len(near)))
        hand = rng.sample(near, m) + rng.sample(rest, 13 - m)
        for p in (1, 2, 3):
            offers = _offers({p: hand}, discard)
            assert all(not v for q, v in offers.items() if q != p)
            melds = [legal_oracles.decode_meld(n.naki_code) for n in offers[p]]
            got = [legal_oracles.post_index(md) for md in melds]
            assert len(got) == len(set(got)), (hand, discard, p, got)
            assert set(got) == legal_oracles.call_offers_by_tiles(hand, discard, p), (hand, discard, p, got)
            for md in melds:
                assert md.called // 4 == k and (md.called in RED) == (discard in RED), (hand, discard, p, md)
            checks += 1
            seen["own_red_chi"] += any(65 <= i <= 82 for i in got)
            seen["own_red_pon"] += any(110 <= i <= 112 for i in got)
            seen["daiminkan"] += any(i >= 120 for i in got)
            seen["red_daiminkan"] += any(i >= 120 for i in got) and discard in RED
            seen["2+_chi_runs"] += len({legal_oracles.meld_kind(md) for md in melds if md.kind == "chi"}) >= 2
    assert checks == 9000
    assert all(seen[k] > 0 for k in ("own_red_chi", "own_red_pon", "daiminkan", "red_daiminkan", "2+_chi_runs")), seen


def test_pon_from_three_plain_copies():
    assert _indices(_offers({2: [41, 42, 43]}, 40)[2]) == [93, 130]  # 3 x 2p held, 2p discarded: pon, daiminkan


def test_own_red_pon():
    # 0m55m held (red 16 + 17, 18), a plain 5m discarded: red pon, plain pon, daiminkan
    assert _indices(_offers({2: [16, 17, 18]}, 19)[2]) == [87, 110, 124]


def test_multi_shape_chi():
    hand = [8, 12, 20, 24]  # 3m 4m 6m 7m
    plain = _offers({1: hand}, 17)[1]  # plain 5m from the kamicha
    assert _indices(plain) == [10, 12, 14]  # 345m / 456m / 567m, each called 5m
    red = _offers({1: hand}, 16)[1]  # the red 5m: the called-red chis keep the plain indices
    assert _indices(red) == [10, 12, 14] and _called(red) == {16}


def test_only_the_shimocha_gets_chi():
    hand = [8, 12, 20, 24]
    offers = _offers({0: hand, 1: hand, 2: hand, 3: hand}, 17)
    assert _indices(offers[1]) == [10, 12, 14]
    assert offers[0] == offers[2] == offers[3] == []  # the discarder gets nothing, the others have no pair


def test_red_daiminkan_calls_the_red_five():
    offers = _offers({3: [17, 18, 19]}, 16)[3]  # 3 plain 5m held, the red 5m discarded
    assert _indices(offers) == [87, 124]
    daiminkan = [n for n in offers if n.is_minkan()]
    assert len(daiminkan) == 1 and _called(daiminkan) == {16}
    assert _called(offers) == {16}
