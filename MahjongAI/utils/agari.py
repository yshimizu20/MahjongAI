"""Win legality under Tenhou rules: the yaku check, furiten, and the win options offered to the model.

Model-agnostic: the parser (process_xml.py) uses it to build the win options, and a play-time engine can reuse it.
Every input is a 34-kind count vector (red fives counted in their kind) plus the player's melds as Naki objects.

Rules (tenhou.net/man and real logs, see task/plans/T-033-win-legality.md):
  - a win needs a complete shape AND at least one yaku (dora never counts); no renhou; open tanyao is allowed
  - furiten blocks rons only, never a tsumo:
      discard: some wait of the hand is among the player's own discards (called-away discards included)
      temp:    the player passed a complete tile (also a no-yaku tile, also a chankan) since their own last discard;
               another player's call does not clear it
      riichi:  the player passed a complete tile while in riichi; it lasts for the rest of the round
"""
import collections
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from MahjongAI.utils.mahjong_version import require_mahjong_2

require_mahjong_2()

from mahjong.agari import Agari
from mahjong.constants import EAST
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld

from MahjongAI.decision import AgariDecision
from MahjongAI.draw import Naki


_agari = Agari()
_calculator = HandCalculator()

TENHOU_RULES = OptionalRules(has_open_tanyao=True, has_aka_dora=True, has_double_yakuman=False)
BONUS_YAKU = frozenset({"Dora", "AkaDora", "UraDora"})  # dora never make a hand legal; the synthetic ids create AkaDora
# the library's wind yaku classes (one per direction) -> canonical names; Tenhou plays no North round
WIND_NAMES = {
    "SeatWindEast": "SeatWind:E", "SeatWindSouth": "SeatWind:S", "SeatWindWest": "SeatWind:W", "SeatWindNorth": "SeatWind:N",
    "RoundWindEast": "RoundWind:E", "RoundWindSouth": "RoundWind:S", "RoundWindWest": "RoundWind:W",
    "RoundWindNorth": "RoundWind:N",
}

FURITEN_DISCARD = "discard"
FURITEN_TEMP = "temp"
FURITEN_RIICHI = "riichi"
NO_YAKU = "no_yaku"


@dataclass(frozen=True)
class WinContext:
    """Everything about the situation that a yaku can depend on. seat_wind / round_wind: library constants
    EAST..NORTH (27..30)."""
    is_tsumo: bool
    seat_wind: int
    round_wind: int
    is_riichi: bool = False
    is_double_riichi: bool = False
    is_ippatsu: bool = False
    is_rinshan: bool = False
    is_chankan: bool = False
    is_haitei: bool = False
    is_houtei: bool = False
    is_tenhou: bool = False
    is_chiihou: bool = False

    def hand_config(self) -> HandConfig:
        return HandConfig(
            is_tsumo=self.is_tsumo,
            is_riichi=self.is_riichi,
            is_daburu_riichi=self.is_double_riichi,
            is_ippatsu=self.is_ippatsu,
            is_rinshan=self.is_rinshan,
            is_chankan=self.is_chankan,
            is_haitei=self.is_haitei,
            is_houtei=self.is_houtei,
            is_tenhou=self.is_tenhou,
            is_chiihou=self.is_chiihou,
            is_renhou=False,
            player_wind=self.seat_wind,
            round_wind=self.round_wind,
            options=TENHOU_RULES,
        )


class WinRuleError(Exception):
    """The yaku library rejected a hand for a reason other than "no yaku": a flag or builder bug, never a
    legal "no win"."""


def seat_wind(player: int, oya: int) -> int:
    return EAST + (player - oya) % 4


def round_wind(curr_round: int) -> int:
    """curr_round 0-3 East, 4-7 South, 8-11 West (Tenhou's 西入)."""
    return EAST + curr_round // 4


def meld_kinds(naki: Naki) -> Tuple[str, List[int]]:
    """(library meld type, the meld's kinds). A kan lists its kind four times."""
    if naki.is_chi():
        color, number, *_ = naki.pattern_chi()
        base = color * 9 + number
        return Meld.CHI, [base, base + 1, base + 2]
    if naki.is_pon():
        color, number, *_ = naki.pattern_pon()
        return Meld.PON, [color * 9 + number] * 3
    if naki.is_kakan():
        color, number, *_ = naki.pattern_kakan()
        return Meld.SHOUMINKAN, [color * 9 + number] * 4
    if naki.is_minkan():
        color, number, *_ = naki.pattern_minkan()
        return Meld.KAN, [color * 9 + number] * 4
    if naki.is_ankan():
        color, number, *_ = naki.pattern_ankan()
        return Meld.KAN, [color * 9 + number] * 4
    raise ValueError(f"Unknown naki type: {naki!r}")


def is_complete(concealed34, kind: int) -> bool:
    """True iff the concealed hand (3n+1 tiles) plus one tile of `kind` has a complete shape. A fifth copy is never
    a wait (Agari.is_agari accepts some fifth-copy shapes, e.g. 1111345678999m + 1m)."""
    counts = np.asarray(concealed34)[:34].astype(int)
    if counts[kind] >= 4:
        return False
    counts = counts.copy()
    counts[kind] += 1
    return bool(_agari.is_agari(counts.tolist()))  # no open sets: is_agari would SUBTRACT them from the counts


def waits(concealed34) -> frozenset:
    """Every kind that completes the concealed hand (3n+1 tiles). Shape only: a wait without yaku is a wait."""
    return frozenset(k for k in range(34) if is_complete(concealed34, k))


def _build_hand(concealed34, melds: Sequence[Naki], win_kind: int):
    """136-ids from 34-kind counts: melds first, then the concealed tiles; the win tile is a concealed id.
    Red fives and dora do not decide whether a yaku exists, so 34-kind input is exact for legality."""
    used = collections.Counter()

    def take(kind: int) -> int:
        tile = kind * 4 + used[kind]
        used[kind] += 1
        assert used[kind] <= 4, f"a fifth copy of kind {kind}"
        return tile

    lib_melds = []
    for naki in melds:
        meld_type, kinds = meld_kinds(naki)
        ids = [take(k) for k in kinds]
        opened = not naki.is_ankan()
        lib_melds.append(Meld(meld_type, ids, opened, ids[0] if opened else None))
    tiles = [t for m in lib_melds for t in m.tiles]
    win_tile = None
    counts = np.asarray(concealed34)[:34].astype(int)
    for k in np.flatnonzero(counts).tolist():
        for _ in range(counts[k]):
            tile = take(k)
            tiles.append(tile)
            if k == win_kind:
                win_tile = tile
    assert win_tile is not None, f"win kind {win_kind} is not in the concealed hand"
    return tiles, lib_melds, win_tile


def _canonical_name(yaku) -> Optional[str]:
    """The canonical name of one library yaku: None for bonus han, WIND_NAMES for a wind, else the class name."""
    name = type(yaku).__name__
    if name in BONUS_YAKU:
        return None
    return WIND_NAMES.get(name, name)


def win_yaku(concealed34, melds: Sequence[Naki], win_kind: int, ctx: WinContext) -> Optional[Tuple[str, ...]]:
    """The yaku of a complete hand, or None when it has none.

    concealed34: the concealed hand WITH the win tile (3n+2 tiles). Names: the library's yaku class names, bonus
    yaku dropped, SeatWind<East..North> -> "SeatWind:<E|S|W|N>", RoundWind<East..West> -> "RoundWind:<E|S|W>".
    Any library error other than "no_yaku" (hand_not_winning included) raises WinRuleError."""
    tiles, lib_melds, win_tile = _build_hand(concealed34, melds, win_kind)
    result = _calculator.estimate_hand_value(tiles, win_tile, lib_melds, [], ctx.hand_config())
    if result.error == HandCalculator.ERR_NO_YAKU:
        return None
    if result.error:
        raise WinRuleError(f"{result.error}: concealed={np.asarray(concealed34)[:34].astype(int).tolist()} "
                           f"melds={list(melds)!r} win_kind={win_kind} ctx={ctx}")
    names = tuple(sorted(n for n in (_canonical_name(y) for y in result.yaku) if n is not None))
    if not names:  # the library found a "yaku" that is only dora: still no yaku
        return None
    return names


class FuritenTracker:
    """The furiten state of one round. Owns each player's cached waits, own discard kinds, and the temp / riichi
    flags. Waits change only through the player's own discard (a call or a kan is always followed by that
    player's discard before they can answer another tile), so set_hand after each discard keeps them exact."""

    def __init__(self):
        self._waits = [frozenset() for _ in range(4)]
        self._discards = [set() for _ in range(4)]
        self._temp = [False] * 4
        self._riichi = [False] * 4

    def set_hand(self, p: int, concealed34) -> None:
        """At INIT and after each of p's discards: recompute p's waits; p's own discard ends same-turn furiten."""
        self._waits[p] = waits(concealed34)
        self._temp[p] = False

    def on_discard(self, p: int, kind: int) -> None:
        """p discarded `kind`: it stays in p's discards for furiten even if another player calls it."""
        self._discards[p].add(kind)

    def on_offer_passed(self, kind: int, responders: Sequence[int], reaches: Sequence[int]) -> None:
        """A discard or kakan of `kind` was answered and the round goes on: every responder passed it. A responder
        whose hand it completes is now in same-turn furiten, and for the rest of the round if in riichi."""
        for q in responders:
            if kind in self._waits[q]:
                self._temp[q] = True
                if reaches[q]:
                    self._riichi[q] = True

    def waits(self, p: int) -> frozenset:
        return self._waits[p]

    def reason(self, p: int) -> Optional[str]:
        """Why p cannot ron now, or None. Precedence: discard, riichi, temp."""
        if self._waits[p] & self._discards[p]:
            return FURITEN_DISCARD
        if self._riichi[p]:
            return FURITEN_RIICHI
        if self._temp[p]:
            return FURITEN_TEMP
        return None


def evaluate_ron(
    discarder: int,
    kind: int,
    concealed_hands: Sequence[np.ndarray],
    melds: Sequence[Sequence[Naki]],
    tracker: FuritenTracker,
    contexts: Dict[int, WinContext],
) -> Tuple[List[Optional[AgariDecision]], Dict[int, str]]:
    """The ron options on a discard (or a kakan, for chankan) of `kind`.

    Returns (rons, blocked): rons[p] is an AgariDecision or None; blocked[p] says why a responder whose hand
    the tile completes gets no option (a furiten reason or NO_YAKU)."""
    rons: List[Optional[AgariDecision]] = [None] * 4
    blocked: Dict[int, str] = {}
    for p in range(4):
        if p == discarder or not is_complete(concealed_hands[p], kind):
            continue
        reason = tracker.reason(p)
        yaku = None
        if reason is None:
            hand = np.asarray(concealed_hands[p])[:34].astype(int).copy()
            hand[kind] += 1
            yaku = win_yaku(hand, melds[p], kind, contexts[p])
            if yaku is None:
                reason = NO_YAKU
        if reason is None:
            rons[p] = AgariDecision(p, False, yaku=yaku)
        else:
            blocked[p] = reason
    return rons, blocked


def evaluate_tsumo(
    player: int,
    concealed34,
    melds: Sequence[Naki],
    drawn_kind: int,
    ctx: WinContext,
) -> Tuple[List[AgariDecision], Optional[str]]:
    """The tsumo option after a draw. concealed34: the hand WITH the drawn tile. Returns (decisions, blocked):
    blocked is NO_YAKU when the shape is complete but the hand has no yaku. Furiten never blocks a tsumo."""
    hand = np.asarray(concealed34)[:34].astype(int).copy()
    hand[drawn_kind] -= 1
    if not is_complete(hand, drawn_kind):
        return [], None
    yaku = win_yaku(concealed34, melds, drawn_kind, ctx)
    if yaku is None:
        return [], NO_YAKU
    return [AgariDecision(player, False, yaku=yaku)], None
