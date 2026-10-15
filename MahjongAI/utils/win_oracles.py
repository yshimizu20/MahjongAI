"""Independent re-derivation of win legality, for the gates only (gates.py).

Never imports MahjongAI.utils.agari: a gate must not check the code with itself. Each oracle reaches the same rule
by another route:
  complete_on / complete_hand  a direct Agari.is_agari call on the concealed counts (no open sets: is_agari
                               SUBTRACTS them), not the parser's cached waits
  furiten_reason               furiten from facts the gates read off the token history and the earlier responder
                               states, not FuritenTracker
  TENHOU_YAKU_ID / xml_yaku_ids  the parser's yaku names mapped to Tenhou's ids, against the <AGARI> attributes
Pure functions over plain values; gates.py extracts those values from the tokens and the halfturns.
"""
import re
from typing import FrozenSet, Iterable, Optional

import numpy as np

from MahjongAI.utils.mahjong_version import require_mahjong_1

require_mahjong_1()

from mahjong.agari import Agari
from mahjong.hand_calculating.yaku_config import YakuConfig

_agari = Agari()

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


def complete_hand(concealed34) -> bool:
    """The concealed 3n+2 tiles have a complete shape."""
    counts = np.asarray(concealed34)[:34].astype(int)
    return bool(_agari.is_agari(counts.tolist()))


def complete_on(concealed34, kind: int) -> bool:
    """The concealed 3n+1 tiles plus one `kind` have a complete shape; a fifth copy never completes."""
    counts = np.asarray(concealed34)[:34].astype(int).copy()
    if counts[kind] >= 4:
        return False
    counts[kind] += 1
    return bool(_agari.is_agari(counts.tolist()))


def furiten_reason(concealed34, own_discard_kinds, passed_since_own_discard: bool, passed_since_reach: bool) -> Optional[str]:
    """Why a player whose hand the answered tile completes cannot ron it (Tenhou), or None:
      'discard'  a kind in the player's own discards (called away or not) completes the hand: any wait, yaku or not
      'riichi'   the player passed a complete tile after their reach
      'temp'     the player passed a complete tile after their own latest discard (another player's call between
                 does not matter)
    Precedence as the parser reports it: discard, riichi, temp."""
    if any(complete_on(concealed34, k) for k in own_discard_kinds):
        return "discard"
    if passed_since_reach:
        return "riichi"
    if passed_since_own_discard:
        return "temp"
    return None
