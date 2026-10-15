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
"""
from typing import Optional, Set

import numpy as np
from mahjong.shanten import Shanten

from MahjongAI.draw import Naki
from MahjongAI.utils.constants import RED_FIVE_SLOTS

_shanten = Shanten()


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
