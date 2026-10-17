"""Independent re-derivation of discard, call and option legality, for the gates only (gates.py).

Never imports MahjongAI.process_xml, MahjongAI.decision, MahjongAI.draw, MahjongAI.gameinfo, MahjongAI.utils.legal
or MahjongAI.utils.agari: a gate must not check the code with itself (tests/test_gates.py checks the imports).
Each oracle reaches the same rule by another route:
  decode_meld                Tenhou's `m` bit layout decoded from first principles (not draw.Naki); post_index /
                             during_index restate the action layout from the decoded tiles
  call_offers_by_tiles       chi / pon / daiminkan offers from every pair (triple) of PHYSICAL tiles in the
                             responder's concealed hand (136-ids), not decision.decision_mask
  tenpai_kinds_by_agari      wait enumeration with Agari.is_agari on the FULL hand (concealed + meld tiles),
                             not shanten; the junkara rule (no wait on a kind with four CONCEALED copies) is
                             explicit here, because is_agari accepts a fifth copy
  kuikae_forbidden_by_ranks  rank arithmetic on three physical tiles (the called kind and the two kinds that
                             left the hand), not the Naki decoder
  option rules (T-032)       riichi / kan / call offers from values replayed from the XML (gates.xml_rounds),
                             the riichi-ankan wait set by Shanten == -1 on the concealed hand (not Agari with kan
                             sets), the kuikae dead end from the post-head action index (index arithmetic, not
                             the Naki decoder)
  kyuushu_by_rules (T-034)   the seat's draw count and "any call yet" from the replay (not the parser's
                             first-go-around flag), terminals / honors by rank arithmetic (not a kind list)
Library routes, crossed against the parser (T-040: where both sides need the `mahjong` library, they take the
OTHER call, so one library fault cannot pass on both sides):
  rule                    parser (process_xml / utils)            oracle (here / win_oracles)
  riichi tenpai           Shanten.calculate_shanten <= 0          Agari.is_agari wait enumeration (riichi_tenpai)
  riichi discard mask     Shanten == 0 after the discard (legal)  Agari wait enumeration (tenpai_kinds_by_agari)
  riichi-ankan waits      Agari.is_agari (legal.py)               Shanten == -1 (waits_by_shanten)
  win shape               Agari.is_agari (agari.is_complete)      Shanten == -1 (win_oracles.complete_on)
  yaku                    HandCalculator via agari.win_yaku       hand-written win_oracles.has_yaku AND
                                                                  HandCalculator on the replay's real 136-ids
Pure functions over plain values; gates.py extracts those values from the XML replay, the tokens and the halfturns.
"""
import collections
import itertools
from typing import Iterable, List, Optional, Set

import numpy as np
from mahjong.agari import Agari
from mahjong.shanten import Shanten

_agari = Agari()
_shanten = Shanten()

RED_IDS = frozenset({16, 52, 88})  # the red fives (5m, 5p, 5s), copy 0 of kinds 4, 13, 22

# One Tenhou meld, decoded: kind 'chi' | 'pon' | 'kakan' | 'minkan' | 'ankan'; ids = every 136-id of the meld
# (a kakan: its pon's three plus the added one); called = the id taken from a discard (chi / pon / minkan; a
# kakan keeps its pon's), None for an ankan; added = a kakan's added id, else None; from_rel = the m code's low 2
# bits (discarder = (who + from_rel) % 4; 0 for an ankan); code = the raw m value.
Meld = collections.namedtuple("Meld", ["kind", "ids", "called", "added", "from_rel", "code"])


def decode_meld(m: int) -> Meld:
    """Tenhou's <N m="..."> from its bit layout: chi bit 2 (t = m >> 10: run base 9 (t//3//7) + (t//3)%7, called
    position t % 3, copy offsets at bits 3-4 / 5-6 / 7-8); pon bit 3 / kakan bit 4 (t = m >> 9: kind t // 3, called
    position t % 3, unused copy at bits 5-6); else a kan (hai0 = m >> 8; from_rel 0 = ankan). A nuki (bit 5 in the
    kan branch, three-player only) or pon and kakan bits together raise."""
    rel = m & 3
    if m & 0x4:
        t = m >> 10
        which, t = t % 3, t // 3
        assert t < 21, f"meld code {m}: chi base {t}"
        base = 9 * (t // 7) + t % 7
        ids = tuple((base + i) * 4 + ((m >> (3 + 2 * i)) & 3) for i in range(3))
        return Meld("chi", ids, ids[which], None, rel, m)
    assert not (m & 0x8 and m & 0x10), f"meld code {m}: pon and kakan bits both set"
    if m & 0x18:
        t = m >> 9
        which, k = t % 3, t // 3
        assert k < 34, f"meld code {m}: kind {k}"
        unused = (m >> 5) & 3
        ids = tuple(k * 4 + i for i in range(4) if i != unused)
        if m & 0x8:
            return Meld("pon", ids, ids[which], None, rel, m)
        return Meld("kakan", tuple(range(4 * k, 4 * k + 4)), ids[which], k * 4 + unused, rel, m)
    assert not m & 0x20, f"meld code {m}: nuki bit (three-player) set"
    hai0 = m >> 8
    k = hai0 // 4
    assert k < 34, f"meld code {m}: kind {k}"
    ids = tuple(range(4 * k, 4 * k + 4))
    if rel == 0:
        return Meld("ankan", ids, None, None, 0, m)
    return Meld("minkan", ids, hai0, None, rel, m)


def meld_kind(meld: Meld) -> int:
    """The tile kind of a pon / kan, the lowest kind of a chi's run."""
    return meld.ids[0] // 4


def taken_ids(meld: Meld) -> tuple:
    """The 136-ids a meld takes from its owner's concealed hand: chi / pon 2, daiminkan 3, ankan 4, kakan 1."""
    if meld.kind == "ankan":
        return meld.ids
    if meld.kind == "kakan":
        return (meld.added,)
    return tuple(i for i in meld.ids if i != meld.called)


def post_index(meld: Meld) -> int:
    """The post-head action index of a chi / pon / daiminkan, restated: chi 2-64 = 2 + 21 c + 3 n + which (c suit,
    n lowest rank, which = position of the called tile); a chi with a red five FROM THE HAND 65-82 = 65 + 6 c + slot
    (RED_CHI_SLOT); pon 83-109 = 83 + kind, a pon with a red five from the hand 110-112 = 110 + c, an honor pon
    113-119; daiminkan 120-153 = 120 + kind. A CALLED red five keeps the plain index."""
    k = meld_kind(meld)
    own_red = any(i in RED_IDS for i in taken_ids(meld))
    if meld.kind == "chi":
        c, n = divmod(k, 9)
        which = meld.called // 4 - k
        if own_red:
            return 65 + 6 * c + RED_CHI_INDEX[(n, which)]
        return 2 + 21 * c + 3 * n + which
    if meld.kind == "pon":
        if k >= 27:
            return 113 + k - 27
        return 110 + k // 9 if own_red else 83 + k
    if meld.kind == "minkan":
        return MINKAN_IDX + k
    raise ValueError(f"{meld.kind} has no post-head index")


def during_index(meld: Meld) -> int:
    """The during-head action index of an ankan (3 + kind) or a kakan (37 + kind)."""
    if meld.kind == "ankan":
        return ANKAN_IDX + meld_kind(meld)
    if meld.kind == "kakan":
        return KAKAN_IDX + meld_kind(meld)
    raise ValueError(f"{meld.kind} has no during-head index")


def call_offers_by_tiles(hand_ids: Iterable[int], discard_id: int, rel: int) -> Set[int]:
    """Post-head indices of every chi / pon / daiminkan the PHYSICAL tiles allow: every pair of concealed 136-ids
    that forms a pon with the discard, or a chi (rel == 1: the responder is the discarder's shimocha; one suit,
    consecutive kinds), and a daiminkan when three copies of the discard's kind are held. rel = (responder -
    discarder) % 4. No rule (riichi, last discard, kuikae, kan cap) is applied here: calls_by_rules does that."""
    hand_ids = sorted(hand_ids)
    k = discard_id // 4
    from_rel = -rel % 4
    offers = set()
    for a, b in itertools.combinations(hand_ids, 2):
        ka, kb = a // 4, b // 4
        if ka == kb == k:
            offers.add(post_index(Meld("pon", tuple(sorted((a, b, discard_id))), discard_id, None, from_rel, None)))
        run = sorted([k, ka, kb])
        if (rel == 1 and run[2] < 27 and run[0] // 9 == run[2] // 9 and run[1] == run[0] + 1
                and run[2] == run[0] + 2):
            ids = tuple(sorted((a, b, discard_id), key=lambda i: i // 4))
            offers.add(post_index(Meld("chi", ids, discard_id, None, from_rel, None)))
    if sum(1 for i in hand_ids if i // 4 == k) == 3:
        offers.add(MINKAN_IDX + k)
    return offers

# Tenhou option rules, restated (never imported from legal.py): scores in 100-point units, draws_left = live
# draws after the current one
MIN_RIICHI_SCORE = 10
MIN_RIICHI_DRAWS = 4
KAN_CAP = 4
ANKAN_IDX, KAKAN_IDX = 3, 37  # during head: 0 pass, 1 tsumo, 2 riichi, 3-36 ankan, 37-70 kakan
RIICHI_IDX = 2
KYUUSHU_IDX = 71  # during head: kyuushu kyuuhai abort
KYUUSHU_KINDS = 9
MINKAN_IDX = 120  # post head: 0 pass, 1 ron, 2-82 chi, 83-119 pon, 120-153 daiminkan
RED_CHI_SLOT = {0: (2, 0), 1: (2, 1), 2: (3, 0), 3: (3, 2), 4: (4, 1), 5: (4, 2)}  # slot -> (lowest rank, called pos)
RED_CHI_INDEX = {v: slot for slot, v in RED_CHI_SLOT.items()}


def meld_tiles(melds) -> tuple:
    """(counts34, open_sets) of DECODED melds (decode_meld): a chi its run, a pon three copies, any kan four."""
    counts = np.zeros(34, dtype=int)
    open_sets = []
    for meld in melds:
        kinds = sorted(i // 4 for i in meld.ids)
        for k in kinds:
            counts[k] += 1
        open_sets.append(kinds)
    return counts, open_sets


def tenpai_kinds_by_agari(concealed34, melds) -> Set[int]:
    """Held kinds k such that, after discarding k, some wait w completes the FULL hand (Agari.is_agari with the
    DECODED melds as open sets) and fewer than four copies of w are concealed."""
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


def riichi_tenpai(concealed34, melds) -> bool:
    """The 14-tile hand (concealed + DECODED menzen melds) has a discard that leaves tenpai, by Agari wait
    enumeration (tenpai_kinds_by_agari); the parser asks Shanten <= 0."""
    return bool(tenpai_kinds_by_agari(concealed34, melds))


def riichi_by_rules(tenpai: bool, menzen: bool, in_riichi: bool, score: int, draws_left: int, n_kans: int,
                    n_owners: int) -> bool:
    """Riichi is offered: menzen, not yet in riichi, tenpai (riichi_tenpai), >= 1,000 points, an own draw left, no
    suukaikan pending."""
    return (menzen and not in_riichi and tenpai and score >= MIN_RIICHI_SCORE and draws_left >= MIN_RIICHI_DRAWS
            and not suukaikan(n_kans, n_owners))


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
    """Post-head call indices offered to one responder, from the physical offers (call_offers_by_tiles): none in a
    chankan window, for a responder in riichi, on the last discard or with suukaikan pending; no daiminkan with 4
    kans down; no chi / pon that leaves no legal discard."""
    if chankan or in_riichi or draws_left == 0 or suukaikan(n_kans, n_owners):
        return set()
    return {i for i in unrestricted
            if (i >= MINKAN_IDX and n_kans < KAN_CAP) or (i < MINKAN_IDX and not call_dead_end(concealed34, i, called))}


def kyuushu_by_rules(concealed34, own_draws: int, called: bool) -> bool:
    """Kyuushu kyuuhai is offered: the seat's first draw of the round, no N (chi / pon / kan, ankan included) by
    anyone before it, and >= 9 distinct kinds of rank 1 or 9 or honor (kind >= 27) in the 14 tiles."""
    c = np.asarray(concealed34)[:34]
    terminal_kinds = sum(1 for k in range(34) if c[k] > 0 and (k >= 27 or k % 9 in (0, 8)))
    return own_draws == 1 and not called and terminal_kinds >= KYUUSHU_KINDS
