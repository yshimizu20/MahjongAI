"""Scoring and payments of the engine (Tenhou 4-player rules, tenhou.net/man), pure functions over plain values.

Points come from the `mahjong` library's HandCalculator on the real 136-ids (dora, ura and red fives count), with
the same option set the parser's legality uses (open tanyao, aka dora, no double yakuman, kazoe = yakuman).
The engine owns the arithmetic the library does not write the way Tenhou's XML does: honba (300 per honba on a
ron, 100 per payer on a tsumo, to the head-first winner only on a double ron), riichi sticks (to the head-first
winner), pao (daisangen / daisuushii: the liable seat pays all of that yakuman on a tsumo, half on a ron, and the
honba: 「積み棒は包」), noten bappu, nagashi mangan paid as a tsumo, and the final `owari` points. Every score and
delta is in hundreds of points, as the XML writes them.
Pao covers the WHOLE hand, not only the liable yakuman: a daisangen + tsuuiisou tsumo with 3 honba is paid
64,000 + 900 by the liable seat alone (real log 2019103023gm-00a9-0000-f90e0fe9: sc 204,649,314,0,98,-649,384,0).

Settlement table [C7]: see engine_lib.py --test, one scripted fixture per row.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from mahjong.constants import EAST
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.hand_calculating.yaku_config import YAKU_ID_TO_TENHOU_ID
from mahjong.meld import Meld as LibMeld

from MahjongAI.utils.agari import TENHOU_RULES
from MahjongAI.utils.legal_oracles import Meld, decode_meld

_calculator = HandCalculator()

DORA_ID, URA_ID, AKA_ID = 52, 53, 54
PAO_YAKUMAN = (39, 49)  # daisangen, daisuushii: the only responsibility payments (tenhou.net/man)
YAKUMAN_POINTS = 32000
YAKUMAN_POINTS_DEALER = 48000
NOTEN_BAPPU = 30  # 3,000 points shared by the tenpai seats
MANGAN_TSUMO = (40, 20)  # (the dealer pays, a non-dealer pays) to a non-dealer; a dealer takes 40 from each
RIICHI_STICK = 10
HONBA_RON = 3
HONBA_TSUMO = 1
RETURN_POINTS = 300  # 30,000 返し
UMA = (20, 10, -10, -20)
# Tenhou's display order of the yaku ids in an <AGARI yaku="..."> (derived from the 2021 4p logs; the yakuhai
# 10-20 are ordered per hand by Tenhou, this list fixes one order)
YAKU_ORDER = [1, 21, 2, 4, 0, 3, 5, 6, 7, 9, 22, 29, 8, 24, 25, 28, 26, 30, 20, 19, 18, 14, 15, 10, 11, 16, 12, 13,
              23, 27, 31, 32, 33, 34, 35, 52, 54, 53]
LIMIT = {"": 0, "mangan": 1, "haneman": 2, "baiman": 3, "sanbaiman": 4}
# situational yaku, by the parser's canonical name (utils/agari.py win_yaku) -> HandConfig flag
FLAG_OF_YAKU = {"Riichi": "riichi", "DaburuRiichi": "double_riichi", "Ippatsu": "ippatsu", "Rinshan": "rinshan",
                "Chankan": "chankan", "Haitei": "haitei", "Houtei": "houtei", "Tenhou": "tenhou", "Chiihou": "chiihou"}


@dataclass(frozen=True)
class WinFlags:
    tsumo: bool
    seat_wind: int
    round_wind: int
    riichi: bool = False
    double_riichi: bool = False
    ippatsu: bool = False
    rinshan: bool = False
    chankan: bool = False
    haitei: bool = False
    houtei: bool = False
    tenhou: bool = False
    chiihou: bool = False

    def hand_config(self) -> HandConfig:
        return HandConfig(is_tsumo=self.tsumo, is_riichi=self.riichi or self.double_riichi,
                          is_daburu_riichi=self.double_riichi, is_ippatsu=self.ippatsu, is_rinshan=self.rinshan,
                          is_chankan=self.chankan, is_haitei=self.haitei, is_houtei=self.houtei, is_tenhou=self.tenhou,
                          is_chiihou=self.chiihou, is_renhou=False, player_wind=self.seat_wind,
                          round_wind=self.round_wind, options=TENHOU_RULES)


def flags_from_yaku(names: Iterable[str], tsumo: bool, seat_wind: int, round_wind: int) -> WinFlags:
    """The win flags from the parser's canonical yaku names of the offered win (AgariDecision.yaku): the situation
    the parser found legal is the one the engine scores."""
    flags = {FLAG_OF_YAKU[n]: True for n in names if n in FLAG_OF_YAKU}
    return WinFlags(tsumo=tsumo, seat_wind=seat_wind, round_wind=round_wind, **flags)


@dataclass(frozen=True)
class HandScore:
    han: int
    fu: int
    points: int  # the ron payment, or the tsumo total, without honba and sticks
    limit: int  # 0 none, 1 mangan, 2 haneman, 3 baiman, 4 sanbaiman, 5 yakuman (kazoe included)
    yaku: Tuple[Tuple[int, int], ...]  # (Tenhou id, han) in display order; empty for a yakuman hand
    yakuman: Tuple[int, ...]  # Tenhou ids of the yakuman; empty otherwise (a kazoe yakuman lists its yaku)
    pay_dealer: int  # tsumo: what the dealer pays a non-dealer winner (0 on a ron)
    pay_other: int  # tsumo: what each non-dealer pays (a dealer winner: each seat)

    def ten_attr(self) -> str:
        return f"{self.fu},{self.points},{self.limit}"

    def yaku_attr(self) -> Tuple[str, str]:
        """("yakuman", "39,42") for a yakuman hand, else ("yaku", "id,han,...")."""
        if self.yakuman:
            return "yakuman", ",".join(map(str, self.yakuman))
        return "yaku", ",".join(f"{i},{h}" for i, h in self.yaku)


def lib_melds(meld_codes: Sequence[int]) -> Tuple[List[LibMeld], List[int], bool]:
    """(library melds, their tile ids, hand is open) from Tenhou meld codes."""
    melds, ids, is_open = [], [], False
    for code in meld_codes:
        m: Meld = decode_meld(code)
        kind = {"chi": LibMeld.CHI, "pon": LibMeld.PON, "minkan": LibMeld.KAN, "ankan": LibMeld.KAN,
                "kakan": LibMeld.SHOUMINKAN}[m.kind]
        opened = m.kind != "ankan"
        is_open |= opened
        melds.append(LibMeld(kind, list(m.ids), opened, m.called if opened else None))
        ids += list(m.ids)
    return melds, ids, is_open


def score_hand(concealed_ids: Sequence[int], meld_codes: Sequence[int], win_id: int, flags: WinFlags,
               dora_indicators: Sequence[int], ura_indicators: Sequence[int]) -> HandScore:
    """concealed_ids: the concealed tiles WITH the win tile. ura_indicators: only for a riichi winner (the engine
    passes [] otherwise); a riichi winner always lists ura (id 53), with 0 han when none, as Tenhou does."""
    melds, meld_ids, is_open = lib_melds(meld_codes)
    tiles = sorted(list(concealed_ids) + meld_ids)
    n_kans = sum(1 for c in meld_codes if decode_meld(c).kind not in ("chi", "pon"))
    assert len(tiles) == 14 + n_kans, tiles
    result = _calculator.estimate_hand_value(tiles, win_id, melds, list(dora_indicators), flags.hand_config(),
                                             ura_dora_indicators=list(ura_indicators))
    assert not result.error, f"{result.error}: ids={tiles} win={win_id} melds={list(meld_codes)} {flags}"
    level = result.cost["yaku_level"]
    is_yakuman = "yakuman" in level
    yaku: Dict[int, int] = {}
    yakuman: List[int] = []
    for y in result.yaku:
        tenhou_id = YAKU_ID_TO_TENHOU_ID[type(y).yaku_id]
        if y.is_yakuman:
            yakuman.append(tenhou_id)
        else:
            yaku[tenhou_id] = y.han_open if is_open else y.han_closed
    if yakuman:
        assert is_yakuman and "kazoe" not in level, level
        yakuman.sort()
        yaku_pairs: Tuple[Tuple[int, int], ...] = ()
    else:
        if flags.riichi or flags.double_riichi:
            yaku.setdefault(URA_ID, 0)
        yaku_pairs = tuple((i, yaku[i]) for i in YAKU_ORDER if i in yaku)
        assert len(yaku_pairs) == len(yaku), (yaku, yaku_pairs)
    cost = result.cost
    is_dealer = flags.seat_wind == EAST
    if flags.tsumo:
        pay_dealer = 0 if is_dealer else cost["main"] // 100
        pay_other = cost["main"] // 100 if is_dealer else cost["additional"] // 100
    else:
        pay_dealer = pay_other = 0
    return HandScore(han=result.han, fu=result.fu, points=cost["total"], limit=5 if is_yakuman else LIMIT[level],
                     yaku=yaku_pairs, yakuman=tuple(yakuman), pay_dealer=pay_dealer, pay_other=pay_other)


def agari_deltas(score: HandScore, winner: int, from_who: int, oya: int, honba: int, kyotaku: int,
                 pao: Optional[Dict[int, int]] = None) -> List[int]:
    """The four score deltas (hundreds) of one win. from_who == winner: a tsumo. honba / kyotaku: what this win
    collects (0 for the second winner of a double ron). pao: {yakuman id: liable seat} for the winner's melds.
    With a pao the liable seat owes the WHOLE hand (every yakuman of it) and the honba (「積み棒は包」): all of it
    on a tsumo or on a ron by the liable seat; on a ron by a third seat the liable seat and the discarder pay half
    each and the liable seat adds the honba (real logs: 2020040823gm-00a9-0000-44be7808 ba 2,0 48000 -> liable
    -246, discarder -240; 2019103023gm-00a9-0000-f90e0fe9 compound tsumo ba 3,0 64000 -> liable -649)."""
    deltas = [0] * 4
    pao_ids = [i for i in score.yakuman if i in pao] if pao else []
    total = score.points // 100
    if winner == from_who:
        if pao_ids:
            deltas[pao[pao_ids[0]]] -= total + HONBA_RON * honba
        else:
            for q in range(4):
                if q != winner:
                    deltas[q] -= (score.pay_dealer if q == oya else score.pay_other) + HONBA_TSUMO * honba
    else:
        liable = pao[pao_ids[0]] if pao_ids else from_who
        if liable == from_who:
            deltas[from_who] -= total + HONBA_RON * honba
        else:
            deltas[liable] -= total // 2 + HONBA_RON * honba
            deltas[from_who] -= total - total // 2
    deltas[winner] = -sum(deltas) + RIICHI_STICK * kyotaku
    return deltas


def ryuukyoku_deltas(tenpai: Sequence[bool]) -> List[int]:
    """Noten bappu: 3,000 points from the noten seats to the tenpai seats; nothing with 0 or 4 tenpai."""
    n = sum(tenpai)
    if n in (0, 4):
        return [0] * 4
    return [NOTEN_BAPPU // n if t else -NOTEN_BAPPU // (4 - n) for t in tenpai]


def nagashi_deltas(qualifiers: Sequence[int], oya: int) -> List[int]:
    """Nagashi mangan as a draw settlement: each qualifier is paid a mangan tsumo (no honba, no sticks)."""
    deltas = [0] * 4
    for q in qualifiers:
        for p in range(4):
            if p == q:
                continue
            pay = 40 if q == oya else (MANGAN_TSUMO[0] if p == oya else MANGAN_TSUMO[1])
            deltas[p] -= pay
            deltas[q] += pay
    return deltas


def rank_order(scores: Sequence[int]) -> List[int]:
    """Seats from first to fourth: by score, ties by the initial seat order (seat 0 first)."""
    return sorted(range(4), key=lambda q: (-scores[q], q))


def round_half_away(x: float) -> int:
    """四捨五入 on the absolute value: 4.5 -> 5, -40.5 -> -41."""
    sign = -1 if x < 0 else 1
    return sign * int(abs(x) + 0.5)


OWARI_EXACT_FROM = "20220122"  # log date (yyyymmdd) from which Tenhou stopped rounding the raw scores (corpus scan)


def owari_points(final_scores: Sequence[int], exact: bool = False) -> List[float]:
    """Tenhou's end-of-game points: (raw / 1000 - 30) + the uma 20 / 10 / -10 / -20 for the places 2-4; the first
    place gets minus the sum of the other three (so the points sum to 0 and the oka lands there). Logs dated before
    OWARI_EXACT_FROM round every RAW score to thousands first (half away from zero): [265,448,-8,295] ->
    [-13,54,-51,10] (2021081109gm-00a9-0000-fc96a2e9), every owari of 5,491 real 2021 logs (engine_lib
    check_corpus). From 2022-01-22 (`exact=True`) the raw scores stay at 0.1 precision: [-5,423,269,313] ->
    [-50.5,52.3,-13.1,11.3] (2022051222gm-00a9-0000-b02c0edb). The engine emits the rounded rule of its training
    corpus; placements never depend on it."""
    order = rank_order(final_scores)
    pts = [0.0] * 4
    for rank, q in enumerate(order):
        if rank:
            thousands = final_scores[q] / 10 if exact else round_half_away(final_scores[q] / 10)
            pts[q] = round(thousands - RETURN_POINTS // 10 + UMA[rank], 1)
    pts[order[0]] = round(-sum(pts), 1)
    return pts


def owari_attr(final_scores: Sequence[int], exact: bool = False) -> str:
    pts = owari_points(final_scores, exact)
    return ",".join(f"{s},{p:.1f}" for s, p in zip(final_scores, pts))


def sc_attr(before: Sequence[int], deltas: Sequence[int]) -> str:
    return ",".join(f"{s},{d}" for s, d in zip(before, deltas))
