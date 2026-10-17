"""Independent re-derivation of win legality, for the gates only (gates.py).

Never imports MahjongAI.process_xml, MahjongAI.decision, MahjongAI.draw, MahjongAI.gameinfo, MahjongAI.utils.legal
or MahjongAI.utils.agari: a gate must not check the code with itself. Each oracle reaches the same rule by another
route (library routes crossed against the parser: see the table in legal_oracles.py):
  complete_on / complete_hand  Shanten.calculate_shanten == -1 on the concealed counts (the parser asks
                               Agari.is_agari), a fifth copy never completes; memoised per round by the caller
  furiten_reason               furiten from facts the gates read off the token history and the earlier responder
                               states, not FuritenTracker
  has_yaku                     hand-written existence predicate, no library call: situational yaku from the
                               replay's flags, then the structural yaku over the full hand and every standard
                               decomposition (own recursion) with every placement of the win tile
  yaku_ids_by_library          HandCalculator called directly with the replay's REAL 136-ids, decoded melds and a
                               HandConfig rebuilt from the replay (never agari.WinContext / _build_hand)
  TENHOU_YAKU_ID / xml_yaku_ids  yaku names mapped to Tenhou's ids, against the <AGARI> attributes
Pure functions over plain values; gates.py extracts those values from the XML replay, the tokens and the halfturns.
"""
import collections
import re
from typing import FrozenSet, Iterable, Optional

import numpy as np

from MahjongAI.utils.mahjong_version import require_mahjong_1

require_mahjong_1()

from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.hand_calculating.yaku_config import YakuConfig
from mahjong.meld import Meld as LibMeld
from mahjong.shanten import Shanten

_shanten = Shanten()
_calculator = HandCalculator()

WINDS = "ESWN"
XML_BONUS_IDS = frozenset({52, 53, 54})  # dora, ura dora, red dora: never a yaku (ura 53 is listed even with 0 han)


def _yaku_table() -> dict:
    """Canonical yaku name (utils/agari.py win_yaku) -> Tenhou id. The library's tenhou_id is right except: every seat
    wind is 10 and the round wind 11 (Tenhou: 10 + seat wind, 14 + round wind), and Suuankou / SuuankouTanki are
    swapped (library 41 / 40; Tenhou 40 = suuankou, 41 = suuankou tanki)."""
    table = {type(y).__name__: y.tenhou_id for y in vars(YakuConfig()).values() if getattr(y, "tenhou_id", None) is not None}
    # wind yaku reach the parser only as SeatWind / RoundWind (YakuhaiEast..North would be a renaming bug), and
    # Paarenchan reuses Tenhou's tenhou id (37): none of these may map to an id
    for name in ("YakuhaiOfPlace", "YakuhaiOfRound", "YakuhaiEast", "YakuhaiSouth", "YakuhaiWest", "YakuhaiNorth",
                 "Paarenchan", "Dora", "AkaDora"):
        table.pop(name, None)
    table.update({f"SeatWind:{w}": 10 + i for i, w in enumerate(WINDS)})
    table.update({f"RoundWind:{w}": 14 + i for i, w in enumerate(WINDS[:3])})
    table.update({"Suuankou": 40, "SuuankouTanki": 41})
    return table


TENHOU_YAKU_ID = _yaku_table()


def parser_yaku_ids(names: Iterable[str]) -> Optional[FrozenSet[int]]:
    """Tenhou ids of the parser's canonical yaku names; None if a name has no id."""
    ids = [TENHOU_YAKU_ID.get(n) for n in names]
    return None if None in ids else frozenset(ids)


def xml_yaku_ids(attrs: str) -> FrozenSet[int]:
    """The yaku ids of one <AGARI>: `yaku="id,han,..."` minus the bonus ids, or `yakuman="id,..."`."""
    m = re.search(r'\byaku="([^"]*)"', attrs)
    if m:
        v = list(map(int, m.group(1).split(",")))
        return frozenset(v[i] for i in range(0, len(v), 2) if v[i] not in XML_BONUS_IDS)
    return frozenset(map(int, re.search(r'\byakuman="([^"]*)"', attrs).group(1).split(",")))


def _shape_complete(counts: tuple, memo: Optional[dict]) -> bool:
    if memo is not None and counts in memo:
        return memo[counts]
    ok = _shanten.calculate_shanten(list(counts)) == -1
    if memo is not None:
        memo[counts] = ok
    return ok


def complete_hand(concealed34, memo: Optional[dict] = None) -> bool:
    """The concealed 3n+2 tiles have a complete shape (Shanten == -1). memo: a dict the caller keeps per round."""
    return _shape_complete(tuple(int(x) for x in np.asarray(concealed34)[:34]), memo)


def complete_on(concealed34, kind: int, memo: Optional[dict] = None) -> bool:
    """The concealed 3n+1 tiles plus one `kind` have a complete shape; a fifth copy never completes."""
    counts = [int(x) for x in np.asarray(concealed34)[:34]]
    if counts[kind] >= 4:
        return False
    counts[kind] += 1
    return _shape_complete(tuple(counts), memo)


def furiten_reason(concealed34, own_discard_kinds, passed_since_own_discard: bool, passed_since_reach: bool,
                   memo: Optional[dict] = None) -> Optional[str]:
    """Why a player whose hand the answered tile completes cannot ron it (Tenhou), or None:
      'discard'  a kind in the player's own discards (called away or not) completes the hand: any wait, yaku or not
      'riichi'   the player passed a complete tile after their reach
      'temp'     the player passed a complete tile after their own latest discard (another player's call between
                 does not matter)
    Precedence as the parser reports it: discard, riichi, temp."""
    if any(complete_on(concealed34, k, memo) for k in own_discard_kinds):
        return "discard"
    if passed_since_reach:
        return "riichi"
    if passed_since_own_discard:
        return "temp"
    return None


# The situation of one win, rebuilt by the gates from the XML replay and the token history (never the parser's
# WinContext). seat_wind / round_wind: LIBRARY wind kinds 27-30 (EAST..NORTH), as HandConfig needs them.
WinFlags = collections.namedtuple("WinFlags", ["tsumo", "riichi", "double_riichi", "ippatsu", "rinshan", "chankan",
                                               "haitei", "houtei", "tenhou", "chiihou", "seat_wind", "round_wind"])
DRAGONS = (31, 32, 33)
ORPHANS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)
TENHOU_RULES = OptionalRules(has_open_tanyao=True, has_aka_dora=True, has_double_yakuman=False)
LIBRARY_BONUS = frozenset({"Dora", "AkaDora", "Uradora"})  # bonus han, never a yaku


class YakuOracleError(Exception):
    """The library rejected a hand of the replay for a reason other than "no yaku" (a flag or replay bug)."""


def _terminal_or_honor(k: int) -> bool:
    return k >= 27 or k % 9 in (0, 8)


def _sets(c: list, i: int = 0):
    """Every split of the counts c into triplets and runs (lists of ('trip' | 'run', kind)); c is restored."""
    while i < 34 and c[i] == 0:
        i += 1
    if i == 34:
        yield []
        return
    if c[i] >= 3:
        c[i] -= 3
        for rest in _sets(c, i):
            yield [("trip", i)] + rest
        c[i] += 3
    if i < 27 and i % 9 <= 6 and c[i + 1] and c[i + 2]:
        for k in (i, i + 1, i + 2):
            c[k] -= 1
        for rest in _sets(c, i):
            yield [("run", i)] + rest
        for k in (i, i + 1, i + 2):
            c[k] += 1


def _decompositions(concealed: list):
    """Every standard decomposition of the concealed counts: (pair kind, [(shape, kind), ...])."""
    for p in range(34):
        if concealed[p] >= 2:
            concealed[p] -= 2
            for sets in _sets(concealed):
                yield p, sets
            concealed[p] += 2


def _structural_yaku(pair, closed, open_blocks, placed, win_kind, menzen, no_melds, ron, yakuhai) -> bool:
    """A yaku of one decomposition with the win tile (win_kind) in block `placed` (an index into closed, or 'pair').
    closed: the concealed sets; open_blocks: the meld blocks (shape, kind, concealed?) with shape run | trip | kan."""
    blocks = [(s, k, True) for s, k in closed] + open_blocks
    runs = [k for s, k, _ in blocks if s == "run"]
    trips = [k for s, k, _ in blocks if s in ("trip", "kan")]
    if no_melds and len(runs) == 4 and pair not in yakuhai and placed != "pair":
        s, k = closed[placed]
        if s == "run" and ((win_kind == k and k % 9 != 6) or (win_kind == k + 2 and k % 9 != 0)):
            return True  # pinfu: four runs, a non-yakuhai pair, a two-sided wait
    if menzen and len(set(k for s, k in closed if s == "run")) < len([k for s, k in closed if s == "run"]):
        return True  # iipeikou (ryanpeikou contains it)
    for n in range(9):
        if all(9 * c + n in runs for c in range(3)):
            return True  # sanshoku doujun
        if all(9 * c + n in trips for c in range(3)):
            return True  # sanshoku doukou
    for c in range(3):
        if all(9 * c + n in runs for n in (0, 3, 6)):
            return True  # ittsu
    if _terminal_or_honor(pair) and all(_terminal_or_honor(k) or (s == "run" and _terminal_or_honor(k + 2))
                                        for s, k, _ in blocks):
        return True  # chanta (junchan and honroutou inside it): every block, the pair included, has one
    if len(trips) == 4:
        return True  # toitoi
    concealed_trips = sum(1 for i, (s, k) in enumerate(closed) if s == "trip" and not (ron and placed == i))
    concealed_trips += sum(1 for s, k, closed_kan in open_blocks if s == "kan" and closed_kan)
    if concealed_trips >= 3:
        return True  # sanankou: on a ron the triplet the win tile completes is open
    if sum(1 for s, _, _ in open_blocks if s == "kan") >= 3:
        return True  # sankantsu
    return False


def has_yaku(concealed34, melds, win_kind: int, flags: WinFlags) -> bool:
    """True iff the complete hand has at least one yaku (dora never count), with no library call.
    concealed34: the concealed counts WITH the win tile (3n+2); melds: DECODED melds (legal_oracles.decode_meld).
    Situational yaku (riichi, double riichi, ippatsu, menzen tsumo, haitei, houtei, rinshan, chankan, tenhou,
    chiihou) -> True; else the structural ones over the full hand (tanyao, open allowed; yakuhai = >= 3 of a dragon /
    seat wind / round wind kind; honitsu / chinitsu; kokushi; chiitoitsu: 7 distinct pairs, no meld), then over every
    standard decomposition and every placement of the win tile (pinfu, iipeikou, sanshoku doujun / doukou, ittsu,
    chanta, toitoi, sanankou, sankantsu). Every yakuman lies inside one of these, so no yakuman list is needed."""
    concealed = [int(x) for x in np.asarray(concealed34)[:34]]
    menzen = all(m.kind == "ankan" for m in melds)
    if (flags.riichi or flags.double_riichi or flags.ippatsu or flags.rinshan or flags.chankan or flags.haitei
            or flags.houtei or flags.tenhou or flags.chiihou or (flags.tsumo and menzen)):
        return True
    full = list(concealed)
    open_blocks = []
    for m in melds:
        k = m.ids[0] // 4
        for i in m.ids:
            full[i // 4] += 1
        shape = "run" if m.kind == "chi" else "trip" if m.kind == "pon" else "kan"
        open_blocks.append((shape, k, m.kind == "ankan"))
    kinds = [k for k in range(34) if full[k]]
    yakuhai = set(DRAGONS) | {flags.seat_wind, flags.round_wind}
    if all(not _terminal_or_honor(k) for k in kinds):
        return True  # tanyao
    if any(full[k] >= 3 for k in yakuhai):
        return True  # yakuhai: an honor kind can only form a triplet
    if len({k // 9 for k in kinds if k < 27}) <= 1:
        return True  # honitsu / chinitsu (an all-honor hand: tsuuiisou)
    if not melds:
        if all(concealed[k] for k in ORPHANS) and sum(concealed) == 14:
            return True  # kokushi
        if sum(1 for v in concealed if v == 2) == 7:
            return True  # chiitoitsu
    for pair, closed in _decompositions(concealed):
        places = (["pair"] if pair == win_kind else []) + [
            i for i, (s, k) in enumerate(closed) if (k == win_kind if s == "trip" else k <= win_kind <= k + 2)]
        for placed in places:
            if _structural_yaku(pair, closed, open_blocks, placed, win_kind, menzen, not melds, not flags.tsumo, yakuhai):
                return True
    return False


def yaku_ids_by_library(concealed_ids, melds, win_id: int, flags: WinFlags) -> FrozenSet[int]:
    """Tenhou yaku ids of the win by mahjong's HandCalculator, called directly on the replay's REAL 136-ids (the
    concealed ones WITH the win tile, plus every meld id) and a HandConfig built from flags, Tenhou's options
    restated (open tanyao, aka dora, no double yakuman, no renhou). Bonus han dropped; YakuhaiOfPlace -> 10 + seat
    wind, YakuhaiOfRound -> 14 + round wind; other names through TENHOU_YAKU_ID. Empty when the hand has no yaku;
    any other library error, or a name without an id, raises YakuOracleError."""
    lib_melds, tiles = [], list(concealed_ids)
    for m in melds:
        meld_type = {"chi": LibMeld.CHI, "pon": LibMeld.PON, "minkan": LibMeld.KAN, "ankan": LibMeld.KAN,
                     "kakan": LibMeld.SHOUMINKAN}[m.kind]
        opened = m.kind != "ankan"
        lib_melds.append(LibMeld(meld_type, list(m.ids), opened, m.called if opened else None))
        tiles += list(m.ids)
    config = HandConfig(is_tsumo=flags.tsumo, is_riichi=flags.riichi, is_daburu_riichi=flags.double_riichi,
                        is_ippatsu=flags.ippatsu, is_rinshan=flags.rinshan, is_chankan=flags.chankan,
                        is_haitei=flags.haitei, is_houtei=flags.houtei, is_tenhou=flags.tenhou,
                        is_chiihou=flags.chiihou, is_renhou=False, player_wind=flags.seat_wind,
                        round_wind=flags.round_wind, options=TENHOU_RULES)
    result = _calculator.estimate_hand_value(sorted(tiles), win_id, lib_melds, [], config)
    if result.error == HandCalculator.ERR_NO_YAKU:
        return frozenset()
    if result.error:
        raise YakuOracleError(f"{result.error}: ids={sorted(tiles)} win={win_id} melds={list(melds)} {flags}")
    out = set()
    for y in result.yaku:
        name = type(y).__name__
        if name in LIBRARY_BONUS:
            continue
        if name == "YakuhaiOfPlace":
            out.add(10 + flags.seat_wind - 27)
        elif name == "YakuhaiOfRound":
            out.add(14 + flags.round_wind - 27)
        elif name in TENHOU_YAKU_ID:
            out.add(TENHOU_YAKU_ID[name])
        else:
            raise YakuOracleError(f"library yaku {name} has no Tenhou id")
    return frozenset(out)
