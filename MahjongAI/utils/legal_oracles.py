"""Independent re-derivation of discard legality, for the gates only (gates.py).

Never imports MahjongAI.utils.legal: a gate must not check the code with itself. Each oracle reaches the
same rule by another route:
  tenpai_kinds_by_agari      wait enumeration with Agari.is_agari on the FULL hand (concealed + meld tiles),
                             not shanten; the junkara rule (no wait on a kind with four CONCEALED copies) is
                             explicit here, because is_agari accepts a fifth copy
  kuikae_forbidden_by_ranks  rank arithmetic on three physical tiles (the called kind and the two kinds that
                             left the hand), not the Naki decoder
  option rules (T-032)       riichi / kan / call offers from values replayed from the XML (gates.xml_snapshots),
                             the riichi-ankan wait set by Shanten == -1 on the concealed hand (not Agari with kan
                             sets), the kuikae dead end from the post-head action index (index arithmetic, not
                             the Naki decoder)
Pure functions over plain values; gates.py extracts those values from the tokens and the halfturns.
"""
from typing import List, Optional, Set

import numpy as np
from mahjong.agari import Agari
from mahjong.shanten import Shanten

_agari = Agari()
_shanten = Shanten()

# Tenhou option rules, restated (never imported from legal.py): scores in 100-point units, draws_left = live
# draws after the current one
MIN_RIICHI_SCORE = 10
MIN_RIICHI_DRAWS = 4
KAN_CAP = 4
ANKAN_IDX, KAKAN_IDX = 3, 37  # during head: 0 pass, 1 tsumo, 2 riichi, 3-36 ankan, 37-70 kakan
RIICHI_IDX = 2
MINKAN_IDX = 120  # post head: 0 pass, 1 ron, 2-82 chi, 83-119 pon, 120-153 daiminkan
RED_CHI_SLOT = {0: (2, 0), 1: (2, 1), 2: (3, 0), 3: (3, 2), 4: (4, 1), 5: (4, 2)}  # slot -> (lowest rank, called pos)


def meld_tiles(melds) -> tuple:
    """(counts34, open_sets) of the melds: a chi its run, a pon three copies, any kan four (a kakan meld lists only
    the added tile, so a kan is restated from its kind)."""
    counts = np.zeros(34, dtype=int)
    open_sets = []
    for meld in melds:
        exposed, acquired = meld.get_exposed()
        if meld.is_chi() or meld.is_pon():
            kinds = sorted(t // 4 for t in exposed + [acquired])
        else:
            kinds = [exposed[0] // 4] * 4
        for k in kinds:
            counts[k] += 1
        open_sets.append(kinds)
    return counts, open_sets


def tenpai_kinds_by_agari(concealed34, melds) -> Set[int]:
    """Held kinds k such that, after discarding k, some wait w completes the FULL hand (Agari.is_agari with the
    melds as open sets) and fewer than four copies of w are concealed."""
    concealed = np.asarray(concealed34)[:34].astype(int).copy()
    meld_counts, open_sets = meld_tiles(melds)
    full = concealed + meld_counts
    keep = set()
    for k in np.flatnonzero(concealed > 0).tolist():
        concealed[k] -= 1
        full[k] -= 1
        for w in range(34):
            if concealed[w] >= 4:
                continue
            full[w] += 1
            ok = _agari.is_agari(full.tolist(), open_sets or None)
            full[w] -= 1
            if ok:
                keep.add(k)
                break
        concealed[k] += 1
        full[k] += 1
    return keep


def kuikae_forbidden_by_ranks(called: int, used: List[int]) -> Optional[Set[int]]:
    """Kinds forbidden right after a call, from the called kind and the two kinds that left the hand.
    A pon (all three the same kind) forbids that kind. A chi (three consecutive kinds of one suit) forbids the
    called kind and, when the called tile is an END of the run, the next kind outward on the far side.
    None when the three kinds form neither shape."""
    used = sorted(used)
    if len(used) != 2:
        return None
    if used == [called, called]:
        return {called}
    run = sorted([called] + used)
    if not (run[2] < 27 and run[0] // 9 == run[2] // 9 and run[1] == run[0] + 1 and run[2] == run[0] + 2):
        return None
    forbidden = {called}
    if called == run[0] and run[2] % 9 != 8:
        forbidden.add(run[2] + 1)
    if called == run[2] and run[0] % 9 != 0:
        forbidden.add(run[0] - 1)
    return forbidden


def suukaikan(n_kans: int, n_owners: int) -> bool:
    return n_kans >= KAN_CAP and n_owners >= 2


def riichi_by_rules(concealed34, menzen: bool, in_riichi: bool, score: int, draws_left: int, n_kans: int,
                    n_owners: int) -> bool:
    """Riichi is offered: menzen, not yet in riichi, shanten <= 0, >= 1,000 points, an own draw left, no
    suukaikan pending."""
    return (menzen and not in_riichi and score >= MIN_RIICHI_SCORE and draws_left >= MIN_RIICHI_DRAWS
            and not suukaikan(n_kans, n_owners)
            and _shanten.calculate_shanten(np.asarray(concealed34)[:34].astype(int).tolist()) <= 0)


def waits_by_shanten(concealed34) -> Set[int]:
    """Kinds k that complete the concealed hand (Shanten == -1 with k added), except a kind with four concealed
    copies (junkara). Melds are left out: Shanten counts the missing sets from the tile count."""
    c = np.asarray(concealed34)[:34].astype(int).copy()
    waits = set()
    for k in range(34):
        if c[k] >= 4:
            continue
        c[k] += 1
        if _shanten.calculate_shanten(c.tolist()) == -1:
            waits.add(k)
        c[k] -= 1
    return waits


def riichi_ankan_keeps_waits(concealed34, kind: int) -> bool:
    """concealed34 holds the drawn fourth copy of kind: the waits without that copy == the waits after the kan."""
    c = np.asarray(concealed34)[:34].astype(int)
    before, after = c.copy(), c.copy()
    before[kind] -= 1
    after[kind] -= 4
    return waits_by_shanten(before) == waits_by_shanten(after)


def kan_by_rules(concealed34, pon_kinds: List[int], in_riichi: bool, drawn_kind: Optional[int], draws_left: int,
                 n_kans: int) -> Set[int]:
    """During-head kan indices offered: none without a live draw or with 4 kans down; in riichi only the ankan of
    the drawn kind that keeps the waits; otherwise every kind with four concealed copies (ankan) and every own pon
    whose fourth copy is held (kakan)."""
    c = np.asarray(concealed34)[:34].astype(int)
    if draws_left == 0 or n_kans >= KAN_CAP:
        return set()
    if in_riichi:
        if drawn_kind is not None and c[drawn_kind] == 4 and riichi_ankan_keeps_waits(c, drawn_kind):
            return {ANKAN_IDX + drawn_kind}
        return set()
    return {ANKAN_IDX + k for k in np.flatnonzero(c == 4).tolist()} | {KAKAN_IDX + k for k in pon_kinds if c[k] >= 1}


def call_used_kinds(idx: int) -> List[int]:
    """The two kinds a chi / pon at post-head index idx takes from the hand (index layout, not the Naki decoder)."""
    if 2 <= idx <= 64:
        color, rest = divmod(idx - 2, 21)
        number, which = divmod(rest, 3)
    elif 65 <= idx <= 82:
        color, slot = divmod(idx - 65, 6)
        number, which = RED_CHI_SLOT[slot]
    elif 83 <= idx <= 109:
        return [idx - 83] * 2
    elif 110 <= idx <= 112:
        return [9 * (idx - 110) + 4] * 2
    elif 113 <= idx <= 119:
        return [27 + idx - 113] * 2
    else:
        raise ValueError(f"post index {idx} is not a chi / pon")
    base = 9 * color + number
    return [base + i for i in range(3) if i != which]


def call_dead_end(concealed34, idx: int, called: int) -> bool:
    """After the chi / pon at index idx on kind called, every concealed kind is kuikae-forbidden."""
    used = call_used_kinds(idx)
    after = np.asarray(concealed34)[:34].astype(int).copy()
    for k in used:
        after[k] -= 1
    forbidden = kuikae_forbidden_by_ranks(called, used)
    assert forbidden is not None and (after >= 0).all(), (idx, called, used)
    return not (set(np.flatnonzero(after > 0).tolist()) - forbidden)


def calls_by_rules(unrestricted: Set[int], concealed34, called: int, in_riichi: bool, chankan: bool,
                   draws_left: int, n_kans: int, n_owners: int) -> Set[int]:
    """Post-head call indices offered to one responder, from the tile-only offers (decision_mask): none in a
    chankan window, for a responder in riichi, on the last discard or with suukaikan pending; no daiminkan with 4
    kans down; no chi / pon that leaves no legal discard."""
    if chankan or in_riichi or draws_left == 0 or suukaikan(n_kans, n_owners):
        return set()
    return {i for i in unrestricted
            if (i >= MINKAN_IDX and n_kans < KAN_CAP) or (i < MINKAN_IDX and not call_dead_end(concealed34, i, called))}
