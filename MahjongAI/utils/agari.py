import numpy as np
import sys

sys.path.append("../../")

from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_response import HandResponse
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld
from typing import List

from MahjongAI.draw import Naki
from MahjongAI.decision import AgariDecision
from MahjongAI.utils.constants import TILE2IDX


COLORINT2STR = {0: "m", 1: "p", 2: "s", 3: "z"}

OPTIONS = OptionalRules(
    has_open_tanyao=True,
    has_aka_dora=True,
    has_double_yakuman=True,
    kazoe_limit=13,
    kiriage=False,
    fu_for_open_pinfu=True,
    fu_for_pinfu_tsumo=False,
    renhou_as_yakuman=False,
    has_daisharin=False,
    has_daisharin_other_suits=False,
    has_sashikomi_yakuman=False,
    limit_to_sextuple_yakuman=True,
    paarenchan_needs_yaku=False,
    has_daichisei=False,
)

hand_calculator = HandCalculator()


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
):
    decisions = []
    discarded_tile_idx = TILE2IDX[discarded_tile]

    for p, (hand_tensor, nakis) in enumerate(zip(hand_tensors_full, naki_list)):
        if p == player:
            continue

        # check for furiten
        if (
            sutehai_tensor[p, discarded_tile_idx]
            or (
                discarded_tile_idx < 27
                and discarded_tile_idx % 9 == 4
                and sutehai_tensor[p, 34 + discarded_tile_idx // 9]
            )
            or (
                discarded_tile_idx >= 34
                and sutehai_tensor[p, (discarded_tile_idx - 34) * 9 + 4]
            )
        ):
            if verbose:
                print(f"Player {p} is furiten")
            continue

        hand_tensor[discarded_tile_idx] += 1
        man, pin, sou, honors = hand_tensor2strs(hand_tensor)
        hand_tensor[discarded_tile_idx] -= 1

        tiles = TilesConverter.string_to_136_array(
            man=man, pin=pin, sou=sou, honors=honors, has_aka_dora=True
        )

        melds = []
        for naki in nakis:
            if naki.is_chi():
                color, number, _, has_red, *_ = naki.pattern_chi()
                if has_red:
                    tile_lst = [(color * 9 + number + i) * 4 for i in range(3)]
                else:
                    tile_lst = [(color * 9 + number + i) * 4 + 1 for i in range(3)]
                meld = Meld(Meld.CHI, tiles=tile_lst)
                melds.append(meld)

            elif naki.is_pon():
                color, number, _, has_red, *_ = naki.pattern_pon()
                if has_red:
                    tile_lst = [(color * 9 + number) * 4 + i for i in range(3)]
                else:
                    tile_lst = [(color * 9 + number) * 4 + i + 1 for i in range(3)]
                meld = Meld(Meld.PON, tiles=tile_lst)
                melds.append(meld)

            elif naki.is_kakan():
                color, number, _, has_red, *_ = naki.pattern_kakan()
                tile_lst = [(color * 9 + number) * 4 + i for i in range(4)]
                meld = Meld(Meld.SHOUMINKAN, tiles=tile_lst)
                melds.append(meld)

            elif naki.is_minkan():
                color, number, _, has_red, *_ = naki.pattern_minkan()
                tile_lst = [(color * 9 + number) * 4 + i for i in range(4)]
                meld = Meld(Meld.KAN, tiles=tile_lst)
                melds.append(meld)

            elif naki.is_ankan():
                color, number, _, has_red, *_ = naki.pattern_ankan()
                tile_lst = [(color * 9 + number) * 4 + i for i in range(4)]
                meld = Meld(Meld.KAN, tiles=tile_lst, opened=False)
                melds.append(meld)

            else:
                raise ValueError("Unknown naki type")

        config = HandConfig(
            is_tsumo=False,
            is_riichi=reaches[p],
            is_ippatsu=ippatsu[p],
            is_rinshan=False,
            is_chankan=is_chankan,
            is_haitei=is_haitei,
            is_houtei=False,
            is_daburu_riichi=double_reaches[p],
            is_renhou=is_renhou[p],
            player_wind=player_wind,
            round_wind=round_wind,
            kyoutaku_number=kyotaku,
            tsumi_number=honba,
            options=OPTIONS,
        )

        # TODO: use Agari.is_agari instead for better performance
        result: HandResponse = hand_calculator.estimate_hand_value(
            tiles=tiles,
            win_tile=discarded_tile,
            melds=melds,
            dora_indicators=doras,
            config=config,
        )

        if verbose:
            print(result)

        if result.error is None:
            decisions.append(AgariDecision(p, False))

    return decisions


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
):
    decisions = []
    hand_tensor = hand_tensors_full[player]

    man, pin, sou, honors = hand_tensor2strs(hand_tensor)

    tiles = TilesConverter.string_to_136_array(
        man=man, pin=pin, sou=sou, honors=honors, has_aka_dora=True
    )
    nakis = naki_list[player]

    melds = []
    for naki in nakis:
        if naki.is_chi():
            color, number, _, has_red, *_ = naki.pattern_chi()
            if has_red:
                tile_lst = [(color * 9 + number + i) * 4 for i in range(3)]
            else:
                tile_lst = [(color * 9 + number + i) * 4 + 1 for i in range(3)]
            meld = Meld(Meld.CHI, tiles=tile_lst)
            melds.append(meld)

        elif naki.is_pon():
            color, number, _, has_red, *_ = naki.pattern_pon()
            if has_red:
                tile_lst = [(color * 9 + number) * 4 + i for i in range(3)]
            else:
                tile_lst = [(color * 9 + number) * 4 + i + 1 for i in range(3)]
            meld = Meld(Meld.PON, tiles=tile_lst)
            melds.append(meld)

        elif naki.is_kakan():
            color, number, _, has_red, *_ = naki.pattern_kakan()
            tile_lst = [(color * 9 + number) * 4 + i for i in range(4)]
            meld = Meld(Meld.SHOUMINKAN, tiles=tile_lst)
            melds.append(meld)

        elif naki.is_minkan():
            color, number, _, has_red, *_ = naki.pattern_minkan()
            tile_lst = [(color * 9 + number) * 4 + i for i in range(4)]
            meld = Meld(Meld.KAN, tiles=tile_lst)
            melds.append(meld)

        elif naki.is_ankan():
            color, number, _, has_red, *_ = naki.pattern_ankan()
            tile_lst = [(color * 9 + number) * 4 + i for i in range(4)]
            meld = Meld(Meld.KAN, tiles=tile_lst, opened=False)
            melds.append(meld)

        else:
            raise ValueError("Unknown naki type")

    config = HandConfig(
        is_tsumo=True,
        is_riichi=reaches[player],
        is_ippatsu=ippatsu[player],
        is_rinshan=is_rinshan,
        is_chankan=False,
        is_haitei=is_haitei,
        is_houtei=False,
        is_daburu_riichi=double_reaches[player],
        is_tenhou=is_tenhou,
        is_chiihou=is_chiihou,
        is_renhou=False,
        player_wind=player_wind,
        round_wind=round_wind,
        kyoutaku_number=kyotaku,
        tsumi_number=honba,
        options=OPTIONS,
    )

    result: HandResponse = hand_calculator.estimate_hand_value(
        tiles=tiles,
        win_tile=tsumo_tile,
        melds=melds,
        dora_indicators=doras,
        config=config,
    )

    if verbose:
        print(result)

    if result.error is None:
        decisions.append(AgariDecision(player, False))

    return decisions


def hand_tensor2strs(hand_tensor):
    tile_strs = []

    for i in range(0, 27, 9):
        lst = []
        for j in range(9):
            lst += [j + 1] * int(hand_tensor[i + j])
        if hand_tensor[34 + i // 9]:
            lst.insert(0, 0)
        tile_strs.append("".join(list(map(str, lst))))

    lst = []
    for j in range(27, 34):
        lst += [j - 26] * int(hand_tensor[j])
    tile_strs.append("".join(list(map(str, lst))))

    man, pin, sou, honors = tile_strs
    return man, pin, sou, honors


if __name__ == "__main__":
    hand_tensors_full = [
        np.array(
            [
                1,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
        np.array(
            [
                0,
                2,
                2,
                0,
                2,
                2,
                2,
                1,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
        np.array(
            [
                1,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
        np.array(
            [
                0,
                2,
                2,
                0,
                2,
                2,
                2,
                1,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
    ]
    naki_list = [[], [], [], []]
    sutehai_tensor = np.zeros((4, 37))
    sutehai_tensor[3, 7] = 1  # furiten
    discarded_tile = 28
    doras = [50]
    reaches = [0, 0, 1, 1]
    ippatsu = [0, 0, 0, 0]
    is_chankan = False
    is_haitei = False
    double_reaches = [0, 0, 0, 0]
    is_renhou = [False] * 4
    player_wind = 2
    round_wind = 0
    kyotaku = 1
    honba = 0

    decisions = evaluate_ron(
        2,
        hand_tensors_full,
        naki_list,
        sutehai_tensor,
        discarded_tile,
        doras,
        reaches,
        ippatsu,
        is_chankan,
        is_haitei,
        double_reaches,
        is_renhou,
        player_wind,
        round_wind,
        kyotaku,
        honba,
        verbose=True,
    )

    print(decisions)
