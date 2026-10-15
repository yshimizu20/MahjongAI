"""Tenhou discard legality, for any consumer: the tensor builder, the gates, a play-time engine.

Pure NumPy: no torch, no model imports. Legality is decided per tile KIND (0-33); the 37-slot mask only
maps it onto the action layout (a red five has its own slot 34/35/36).

Rules (Tenhou, kuikae nashi):
  riichi declaration  the declaration discard must leave the hand in tenpai. A wait on a kind whose four
                      copies are all CONCEALED does not count (junkara); mahjong.shanten already rejects it.
                      Copies in the player's own kan do not count toward that: tenhou.net/man, "手牌＝純手牌＋
                      副露手牌として純手牌で４枚使用していなければ成立" (an ankan is a declared meld, not 純手牌).
  kuikae              right after a chi / pon, the discard may not be the called kind, nor, for a chi called
                      on an END of its run, the kind on the other side of the two held tiles (suji kuikae).
                      A kan is followed by a rinshan draw, so it restricts nothing.

Option legality (tenhou.net/man), for the during head (riichi, ankan, kakan) and the post head (chi, pon,
daiminkan). draws_left = live draws after the current one (StateObject.remaining_tsumo: 70 minus every draw so
far, rinshan included); scores in 100-point units:
  riichi              「リーチ宣言は得点が1000点以上でツモ巡がある場合に可能」: score >= 1,000 and an own draw
                      left (draws_left >= 4); none right after a 4th kan by 2+ players (the round ends with the
                      next discard, so no draw is left)
  kan                 no kan without a live draw (the haitei draw); at most 4 kans per round
  riichi ankan        「リーチ後の暗槓は待ちが変わらない場合のみ。送り槓不可、牌姿や役の増減は不問」: only the
                      kind just drawn, and only if the wait set does not change
  calls               the last discard (houtei) cannot be called; after a 4th kan by 2+ players the next discard
                      can only be ronned (suukaikan); no daiminkan when 4 kans exist; no chi after which
                      every concealed tile is kuikae-forbidden (no legal discard would remain; a pon cannot reach
                      this: it would need five copies of one kind)
"""
from typing import List, Optional, Set, Tuple

import numpy as np
from mahjong.agari import Agari
from mahjong.shanten import Shanten

from MahjongAI.draw import Naki
from MahjongAI.utils.constants import RED_FIVE_SLOTS

_shanten = Shanten()
_agari = Agari()

RIICHI_MIN_SCORE = 10  # 1,000 points (scores are in 100-point units)
RIICHI_MIN_DRAWS_LEFT = 4  # an own draw must remain after this one
MAX_KANS = 4


def held_kinds(hand: np.ndarray) -> Set[int]:
    """Kinds (0-33) with at least one copy in the hand (34- or 37-slot counts; red copies count in their kind)."""
    return set(np.flatnonzero(np.asarray(hand)[:34] > 0).tolist())


def tenpai_keeping_kinds(hand34: np.ndarray) -> Set[int]:
    """Held kinds whose discard leaves the concealed hand in tenpai.
    hand34: concealed counts after the draw (14 - 3 * n_melds tiles; at a riichi only ankan can be down)."""
    counts = np.asarray(hand34)[:34].astype(int)
    keep = set()
    for k in held_kinds(counts):
        counts[k] -= 1
        if _shanten.calculate_shanten(counts.tolist()) == 0:
            keep.add(k)
        counts[k] += 1
    return keep


def kuikae_forbidden_kinds(call: Naki) -> Set[int]:
    """Kinds that may not be discarded right after this call (empty for a kan)."""
    if call.is_chi():
        color, number, which, *_ = call.pattern_chi()  # run = base, base + 1, base + 2; called = base + which
        base = 9 * color + number
        forbidden = {base + which}
        if which == 0 and number + 3 <= 8:  # called the low end: the kind above the run
            forbidden.add(base + 3)
        if which == 2 and number >= 1:  # called the high end: the kind below the run
            forbidden.add(base - 1)
        return forbidden
    if call.is_pon():
        color, number, *_ = call.pattern_pon()
        return {9 * color + number}
    return set()


def legal_discard_kinds(hand34: np.ndarray, riichi_declaration: bool, after_call: Optional[Naki]) -> Set[int]:
    """The rule, per kind: held kinds, minus those that break the riichi-declaration tenpai, minus kuikae."""
    kinds = held_kinds(hand34)
    if riichi_declaration:
        kinds &= tenpai_keeping_kinds(hand34)
    if after_call is not None:
        kinds -= kuikae_forbidden_kinds(after_call)
    return kinds


def presence_mask(hand37: np.ndarray) -> np.ndarray:
    """(37,) float32: 1 for every held tile slot. A red five is only ever slot 34/35/36; the plain slot of a
    five needs a non-red copy."""
    hand = np.asarray(hand37, dtype=np.float32)
    mask = (hand > 0).astype(np.float32)
    for plain, red in RED_FIVE_SLOTS:
        mask[plain] = float(hand[plain] - hand[red] > 0)
    return mask


def legal_discard_mask(hand37: np.ndarray, riichi_declaration: bool, after_call: Optional[Naki]) -> np.ndarray:
    """(37,) float32 legal discard mask: presence_mask with every illegal kind cleared (both slots of a five).
    A legal kind keeps its presence slots as they are: a red-only five stays legal on its red slot.
    May be all zero; the caller decides what an empty mask means."""
    mask = presence_mask(hand37)
    red_of = dict(RED_FIVE_SLOTS)
    for k in held_kinds(hand37) - legal_discard_kinds(hand37, riichi_declaration, after_call):
        mask[k] = 0.0
        if k in red_of:
            mask[red_of[k]] = 0.0
    return mask


def is_kan(meld: Naki) -> bool:
    return meld.is_ankan() or meld.is_minkan() or meld.is_kakan()


def kans_on_table(melds) -> Tuple[int, int]:
    """(kan count, distinct kan owners) over the melds of all four seats (a kakan replaces its pon)."""
    owners = [q for q, seat in enumerate(melds) for m in seat if is_kan(m)]
    return len(owners), len(set(owners))


def suukaikan_pending(n_kans: int, n_owners: int) -> bool:
    """A 4th kan by 2+ players: the next discard ends the round (only a ron can stop it)."""
    return n_kans >= MAX_KANS and n_owners >= 2


def riichi_allowed(score: int, draws_left: int, n_kans: int, n_owners: int) -> bool:
    """The point, wall and suukaikan conditions of a riichi (menzen / not in riichi / tenpai are the caller's)."""
    return score >= RIICHI_MIN_SCORE and draws_left >= RIICHI_MIN_DRAWS_LEFT and not suukaikan_pending(n_kans, n_owners)


def kan_allowed(draws_left: int, n_kans: int) -> bool:
    """Any kan (ankan, kakan, daiminkan) needs a live draw for its rinshan tile and a free kan slot."""
    return draws_left > 0 and n_kans < MAX_KANS


def calls_allowed(draws_left: int, n_kans: int, n_owners: int) -> bool:
    """Chi / pon / daiminkan on this discard at all: not the last discard, no suukaikan pending."""
    return draws_left > 0 and not suukaikan_pending(n_kans, n_owners)


def ankan_kinds(melds) -> List[int]:
    """Kinds of a riichi player's melds; a riichi hand can only hold ankans."""
    kinds = []
    for m in melds:
        assert m.is_ankan(), f"a riichi hand holds {m!r}"
        color, number, *_ = m.pattern_ankan()
        kinds.append(9 * color + number)
    return kinds


def wait_kinds(concealed34: np.ndarray, kan_kinds: List[int]) -> Set[int]:
    """Shape waits (no yaku) of a tenpai hand: kinds w that complete concealed + the ankans (Agari.is_agari with
    the kans as sets), except a kind whose four copies are all concealed (junkara, tenhou.net/man)."""
    concealed = np.asarray(concealed34)[:34].astype(int)
    full = concealed.copy()
    for k in kan_kinds:
        full[k] += 4
    sets = [[k] * 4 for k in kan_kinds] or None
    waits = set()
    for w in range(34):
        if concealed[w] >= 4:
            continue
        full[w] += 1
        if _agari.is_agari(full.tolist(), sets):
            waits.add(w)
        full[w] -= 1
    return waits


def riichi_ankan_allowed(concealed34: np.ndarray, drawn_kind: int, kan_kind: int, own_melds) -> bool:
    """An ankan in riichi: of the kind just drawn (no okurikan), and with the wait set unchanged.
    concealed34: the hand after the draw (holds four kan_kind)."""
    if kan_kind != drawn_kind:
        return False
    concealed = np.asarray(concealed34)[:34].astype(int)
    kans = ankan_kinds(own_melds)
    before = concealed.copy()
    before[drawn_kind] -= 1
    after = concealed.copy()
    after[kan_kind] -= 4
    return wait_kinds(before, kans) == wait_kinds(after, kans + [kan_kind])


def call_used_kinds(call: Naki) -> List[int]:
    """The two kinds a chi / pon takes from the caller's hand."""
    if call.is_chi():
        color, number, which, *_ = call.pattern_chi()
        return [9 * color + number + i for i in range(3) if i != which]
    color, number, *_ = call.pattern_pon()
    return [9 * color + number] * 2


def call_leaves_legal_discard(concealed34: np.ndarray, call: Naki) -> bool:
    """After this chi / pon, some concealed kind is not kuikae-forbidden (Tenhou refuses the call otherwise)."""
    after = np.asarray(concealed34)[:34].astype(int).copy()
    for k in call_used_kinds(call):
        after[k] -= 1
    assert (after >= 0).all(), f"{call!r} uses tiles the hand does not hold"
    return bool(held_kinds(after) - kuikae_forbidden_kinds(call))
