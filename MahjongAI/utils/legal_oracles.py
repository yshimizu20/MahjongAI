"""Independent re-derivation of discard legality, for the gates only (gates.py).

Never imports MahjongAI.utils.legal: a gate must not check the code with itself. Each oracle reaches the
same rule by another route:
  tenpai_kinds_by_agari      wait enumeration with Agari.is_agari on the FULL hand (concealed + meld tiles),
                             not shanten; the junkara rule (no wait on a kind with four CONCEALED copies) is
                             explicit here, because is_agari accepts a fifth copy
  kuikae_forbidden_by_ranks  rank arithmetic on three physical tiles (the called kind and the two kinds that
                             left the hand), not the Naki decoder
Pure functions over plain values; gates.py extracts those values from the tokens and the halfturns.
"""
from typing import List, Optional, Set

import numpy as np
from mahjong.agari import Agari

_agari = Agari()


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
