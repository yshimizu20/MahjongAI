"""draw.Naki: the synthetic constructors (offered codes), kakan_of, the offered-code guard, and the action-index
layout over the whole 4p code space (T-008 / T-026)."""
import collections

import pytest

from MahjongAI.draw import Naki
from MahjongAI.utils import legal_oracles


def test_naki_from_chi_info1():
    pin123chi1 = Naki.from_chi_info(1, 0, 0, False)
    assert pin123chi1.naki_code == 21679
    assert pin123chi1.is_chi()
    color, number, which, has_red, exposed, acquired = pin123chi1.pattern_chi()
    assert color == 1
    assert number == 0
    assert which == 0
    assert has_red == False
    assert exposed == [41, 45]
    assert acquired == 37


def test_naki_from_chi_info2():
    sou456chi5red = Naki.from_chi_info(2, 3, 1, True)
    assert sou456chi5red.naki_code == 53391
    assert sou456chi5red.is_chi()
    color, number, which, has_red, exposed, acquired = sou456chi5red.pattern_chi()
    assert color == 2
    assert number == 3
    assert which == 1
    assert has_red == True
    assert exposed == [85, 93]
    assert acquired == 88


def test_naki_from_chi_info3():
    man567chi7red = Naki.from_chi_info(0, 4, 2, True)
    assert man567chi7red.naki_code == 14503
    assert man567chi7red.is_chi()
    color, number, which, has_red, exposed, acquired = man567chi7red.pattern_chi()
    assert color == 0
    assert number == 4
    assert which == 2
    assert has_red == True
    assert exposed == [16, 21]
    assert acquired == 25


def test_naki_from_pon_info1():
    man222pon = Naki.from_pon_info(1, 0, False, 2)
    assert man222pon.naki_code == 1546
    assert man222pon.is_pon()
    color, number, which, has_red, exposed, acquired = man222pon.pattern_pon()
    assert color == 0
    assert number == 1
    assert which == 0
    assert has_red == False
    assert exposed == [6, 7]
    assert acquired == 5


def test_naki_from_pon_info2():
    pin555pongotred = Naki.from_pon_info(13, 0, True, 3)
    assert pin555pongotred.naki_code == 20075
    assert pin555pongotred.is_pon()
    color, number, which, has_red, exposed, acquired = pin555pongotred.pattern_pon()
    assert color == 1
    assert number == 4
    assert which == 0
    assert has_red == True
    assert exposed == [53, 54]
    assert acquired == 52


def test_naki_from_pon_info3():
    sou555ponhadred = Naki.from_pon_info(22, 1, True, 1)
    assert sou555ponhadred.naki_code == 34409
    assert sou555ponhadred.is_pon()
    color, number, which, has_red, exposed, acquired = sou555ponhadred.pattern_pon()
    assert color == 2
    assert number == 4
    assert which == 1
    assert has_red == True
    assert exposed == [88, 90]
    assert acquired == 89


def test_naki_from_pon_info4():
    hatsupon = Naki.from_pon_info(32, 0, False, 1)
    assert hatsupon.naki_code == 49161
    assert hatsupon.is_pon()
    color, number, which, has_red, exposed, acquired = hatsupon.pattern_pon()
    assert color == 3
    assert number == 5
    assert which == 0
    assert has_red == False
    assert exposed == [130, 131]
    assert acquired == 129


def test_kakan_of1():
    man444kakan = Naki.kakan_of(Naki.from_pon_info(3, 0, False, 1))
    assert man444kakan.naki_code == 4625
    assert man444kakan.is_kakan() and man444kakan.offered and man444kakan.from_who() == 1
    color, number, which, has_red, exposed, acquired = man444kakan.pattern_kakan()
    assert color == 0
    assert number == 3
    assert which == 0
    assert has_red == False
    assert exposed == [12]
    assert acquired is None


def test_kakan_of2():
    pin555kakangetred = Naki.kakan_of(Naki.from_pon_info(13, 0, False, 2))  # the pon lacks the red: the kakan adds it
    assert pin555kakangetred.naki_code == 19986
    assert pin555kakangetred.is_kakan() and pin555kakangetred.from_who() == 2
    color, number, which, has_red, exposed, acquired = pin555kakangetred.pattern_kakan()
    assert color == 1
    assert number == 4
    assert which == 0
    assert has_red == True
    assert exposed == [52]
    assert acquired is None


def test_kakan_of3():
    sou555kakanhadred = Naki.kakan_of(Naki.from_pon_info(22, 1, True, 3))  # the pon holds the red: a plain 5s is added
    assert sou555kakanhadred.naki_code == 34419
    assert sou555kakanhadred.is_kakan() and sou555kakanhadred.from_who() == 3
    color, number, which, has_red, exposed, acquired = sou555kakanhadred.pattern_kakan()
    assert color == 2
    assert number == 4
    assert which == 1
    assert has_red == True
    assert exposed == [91]
    assert acquired == None


def test_kakan_of4():
    peikakan = Naki.kakan_of(Naki.from_pon_info(30, 0, False, 1))
    assert peikakan.naki_code == 46097
    assert peikakan.is_kakan() and peikakan.from_who() == 1
    color, number, which, has_red, exposed, acquired = peikakan.pattern_kakan()
    assert color == 3
    assert number == 3
    assert which == 0
    assert has_red == False
    assert exposed == [120]
    assert acquired == None


def test_naki_from_minkan_info1():
    man777minkan = Naki.from_minkan_info(6, 3, 1)
    assert man777minkan.naki_code == 6403
    assert man777minkan.is_minkan()
    color, number, which, has_red, exposed, acquired = man777minkan.pattern_minkan()
    assert color == 0
    assert number == 6
    assert which == 1
    assert has_red == False
    assert exposed == [24, 26, 27]
    assert acquired == 25


def test_naki_from_minkan_info2():
    pin555minkangetred = Naki.from_minkan_info(13, 3, 0)
    assert pin555minkangetred.naki_code == 13315
    assert pin555minkangetred.is_minkan()
    (
        color,
        number,
        which,
        has_red,
        exposed,
        acquired,
    ) = pin555minkangetred.pattern_minkan()
    assert color == 1
    assert number == 4
    assert which == 0
    assert has_red == True
    assert exposed == [53, 54, 55]
    assert acquired == 52


def test_naki_from_minkan_info3():
    sou555minkanhadred = Naki.from_minkan_info(22, 1, 1)
    assert sou555minkanhadred.naki_code == 22785
    assert sou555minkanhadred.is_minkan()
    (
        color,
        number,
        which,
        has_red,
        exposed,
        acquired,
    ) = sou555minkanhadred.pattern_minkan()
    assert color == 2
    assert number == 4
    assert which == 1
    assert has_red == True
    assert exposed == [88, 90, 91]
    assert acquired == 89


def test_naki_from_minkan_info4():
    nanminkan = Naki.from_minkan_info(28, 3, 1)
    assert nanminkan.naki_code == 28931
    assert nanminkan.is_minkan()
    color, number, which, has_red, exposed, acquired = nanminkan.pattern_minkan()
    assert color == 3
    assert number == 1
    assert which == 1
    assert has_red == False
    assert exposed == [112, 114, 115]
    assert acquired == 113


def test_naki_from_ankan_info1():
    man888ankan = Naki.from_ankan_info(8)
    assert man888ankan.naki_code == 8192
    assert man888ankan.is_ankan()
    color, number, which, has_red, exposed, acquired = man888ankan.pattern_ankan()
    assert color == 0
    assert number == 8
    assert which == 0
    assert has_red == False
    assert exposed == [32, 33, 34, 35]
    assert acquired == None


def test_naki_from_ankan_info2():
    pin555ankangetred = Naki.from_ankan_info(13)
    assert pin555ankangetred.naki_code == 13312
    assert pin555ankangetred.is_ankan()
    color, number, which, has_red, exposed, acquired = pin555ankangetred.pattern_ankan()
    assert color == 1
    assert number == 4
    assert which == 0
    assert has_red == True
    assert exposed == [52, 53, 54, 55]
    assert acquired == None


# ---- T-026: the action-index layout over the whole 4p code space

def _every_4p_code():
    """Every 4p Tenhou meld code (own generator): chi 63 runs x called position x 4^3 copy offsets, pon / kakan
    34 kinds x 3 called positions x 4 unused copies x 3 from_rel, every hai0 x from_rel 0 (ankan) / 1-3 (daiminkan)."""
    for t in range(63):
        for c in range(64):
            yield t << 10 | (c >> 4) << 7 | ((c >> 2) & 3) << 5 | (c & 3) << 3 | 0x4 | 3
    for bit in (0x8, 0x10):
        for t in range(102):
            for unused in range(4):
                for rel in (1, 2, 3):
                    yield t << 9 | unused << 5 | bit | rel
    for hai0 in range(136):
        for rel in range(4):
            yield hai0 << 8 | rel


def _semantic_class(meld):
    """What the model's action means: chi = (run base, called kind, own red five), pon = (kind, own red five),
    a kan = its kind. Two codes of one class must share an index; two classes must never share one."""
    own_red = any(i in legal_oracles.RED_IDS for i in legal_oracles.taken_ids(meld))
    kind = legal_oracles.meld_kind(meld)
    if meld.kind == "chi":
        return ("chi", kind, meld.called // 4, own_red)
    if meld.kind == "pon":
        return ("pon", kind, own_red)
    return (meld.kind, kind)


def test_index_bijection_whole_code_space():
    codes = list(_every_4p_code())
    assert len(codes) == len(set(codes)) == 7024 == 4032 + 1224 + 1224 + 408 + 136
    heads = {"post": collections.defaultdict(set), "during": collections.defaultdict(set)}
    for m in codes:
        meld, nk = legal_oracles.decode_meld(m), Naki(m)
        if meld.kind in ("ankan", "kakan"):
            heads["during"][_semantic_class(meld)].add(nk.get_during_turn_filter_idx())
        else:
            heads["post"][_semantic_class(meld)].add(nk.get_post_turn_filter_idx())
    for head, n, lo, hi in (("post", 152, 2, 153), ("during", 68, 3, 70)):
        classes = heads[head]
        assert all(len(ix) == 1 for ix in classes.values()), [(c, ix) for c, ix in classes.items() if len(ix) > 1]
        index = {c: next(iter(ix)) for c, ix in classes.items()}
        assert len(index) == len(set(index.values())) == n, (head, len(index), len(set(index.values())))
        assert (min(index.values()), max(index.values())) == (lo, hi), head


def test_index_layout_anchors():
    post = lambda n: n.get_post_turn_filter_idx()  # noqa: E731
    during = lambda n: n.get_during_turn_filter_idx()  # noqa: E731
    assert post(Naki.from_chi_info(0, 0, 0, False)) == 2  # 123m, called 1m
    assert post(Naki.from_chi_info(2, 6, 2, False)) == 64  # 789s, called 9s
    assert post(Naki.from_chi_info(0, 2, 0, True)) == 65  # 345m called 3m, own red 5m: first own-red chi slot
    assert post(Naki.from_chi_info(2, 4, 2, True)) == 82  # 567s called 7s, own red 5s: last own-red chi slot
    assert post(Naki.from_pon_info(0, 0, False, 1)) == 83  # pon 1m
    assert post(Naki.from_pon_info(26, 0, False, 1)) == 109  # pon 9s
    assert [post(Naki.from_pon_info(t, 1, True, 1)) for t in (4, 13, 22)] == [110, 111, 112]  # own red 5m / 5p / 5s
    assert post(Naki.from_pon_info(27, 0, False, 2)) == 113  # pon East
    assert post(Naki.from_pon_info(33, 0, False, 3)) == 119  # pon chun
    assert post(Naki.from_minkan_info(0, 1, 1)) == 120  # daiminkan 1m
    assert post(Naki.from_minkan_info(33, 2, 0)) == 153  # daiminkan chun
    assert during(Naki.from_ankan_info(0)) == 3 and during(Naki.from_ankan_info(33)) == 36
    assert during(Naki.kakan_of(Naki.from_pon_info(0, 0, False, 1))) == 37
    assert during(Naki.kakan_of(Naki.from_pon_info(33, 0, False, 1))) == 70


def test_kakan_of_every_pon():
    pons = [m for m in _every_4p_code() if m & 0x8 and not m & 0x4]
    assert len(pons) == 1224
    for m in pons:
        pon = legal_oracles.decode_meld(m)
        kakan = Naki.kakan_of(Naki(m))
        meld = legal_oracles.decode_meld(kakan.naki_code)
        k = legal_oracles.meld_kind(pon)
        assert meld.kind == "kakan" and meld.ids == tuple(range(4 * k, 4 * k + 4)), (m, meld)
        assert {meld.added} == set(meld.ids) - set(pon.ids), (m, meld, pon)  # adds the copy the pon left out
        assert (meld.called, meld.from_rel) == (pon.called, pon.from_rel), (m, meld, pon)
        assert kakan.get_during_turn_filter_idx() == 37 + k, m
    with pytest.raises(AssertionError):
        Naki.kakan_of(Naki.from_chi_info(0, 0, 0, False))


def test_offered_code_refuses_get_exposed():
    offered = [Naki.from_chi_info(1, 3, 1, True), Naki.from_pon_info(22, 1, True, 1), Naki.from_minkan_info(13, 3, 0),
               Naki.from_ankan_info(8), Naki.kakan_of(Naki.from_pon_info(13, 0, False, 2))]
    for nk in offered:
        assert nk.offered
        with pytest.raises(ValueError, match="offered call"):
            nk.get_exposed()
        log = Naki(nk.naki_code)  # the same code read from a log names real tiles
        assert not log.offered
        exposed, _ = log.get_exposed()
        assert sorted(exposed) == sorted(legal_oracles.taken_ids(legal_oracles.decode_meld(nk.naki_code)))
