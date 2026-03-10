import numpy as np
import sys

sys.path.append("../../")

from mahjong.agari import Agari
from typing import List

from MahjongAI.draw import Naki
from MahjongAI.decision import AgariDecision
from MahjongAI.utils.constants import TILE2IDX


_agari = Agari()


def _open_sets_34(nakis):
    """Convert Naki list to open_sets_34 format for Agari.is_agari."""
    sets = []
    for naki in nakis:
        if naki.is_chi():
            color, number, *_ = naki.pattern_chi()
            base = color * 9 + number
            sets.append([base, base + 1, base + 2])
        elif naki.is_pon():
            color, number, *_ = naki.pattern_pon()
            t = color * 9 + number
            sets.append([t, t, t])
        elif naki.is_kakan():
            color, number, *_ = naki.pattern_kakan()
            t = color * 9 + number
            sets.append([t, t, t, t])
        elif naki.is_minkan():
            color, number, *_ = naki.pattern_minkan()
            t = color * 9 + number
            sets.append([t, t, t, t])
        elif naki.is_ankan():
            color, number, *_ = naki.pattern_ankan()
            t = color * 9 + number
            sets.append([t, t, t, t])
        else:
            raise ValueError("Unknown naki type")
    return sets or None


def evaluate_ron(
    player,
    hand_tensors_full: List[np.ndarray],
    naki_list: List[List[Naki]],
    sutehai_tensor: np.ndarray,
    discarded_tile: int,
    doras: List[int],
    reaches: List[int],
    ippatsu: List[int],
    is_chankan: bool,
    is_haitei: bool,
    double_reaches: List[int],
    is_renhou: List[bool],
    player_wind: int,
    round_wind: int,
    kyotaku: int,
    honba: int,
    verbose: bool = False,
) -> List[AgariDecision]:
    rons = [None for _ in range(4)]
    discarded_tile_idx = TILE2IDX[discarded_tile][0]

    for p, (hand_tensor, nakis) in enumerate(zip(hand_tensors_full, naki_list)):
        if p == player:
            continue

        # check for furiten
        if sutehai_tensor[p, discarded_tile_idx]:
            if verbose:
                print(f"Player {p} is furiten")
            continue

        hand_tensor[discarded_tile_idx] += 1
        tiles_34 = hand_tensor[:34].astype(int)
        hand_tensor[discarded_tile_idx] -= 1

        if _agari.is_agari(tiles_34, _open_sets_34(nakis)):
            rons[p] = AgariDecision(p, False)

    return rons


def evaluate_tsumo(
    player,
    hand_tensors_full: List[np.ndarray],
    naki_list: List[List[Naki]],
    tsumo_tile: int,
    doras: List[int],
    reaches: List[int],
    ippatsu: List[int],
    is_haitei: bool,
    is_rinshan: bool,
    double_reaches: List[int],
    is_tenhou: bool,
    is_chiihou: bool,
    player_wind: int,
    round_wind: int,
    kyotaku: int,
    honba: int,
    verbose: bool = False,
) -> List[AgariDecision]:
    hand_tensor = hand_tensors_full[player]
    tiles_34 = hand_tensor[:34].astype(int)

    if _agari.is_agari(tiles_34, _open_sets_34(naki_list[player])):
        return [AgariDecision(player, False)]
    return []
