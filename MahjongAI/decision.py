import numpy as np
from typing import List
from MahjongAI.draw import Naki


class Decision:
    NAKI = 1
    REACH = 2
    AGARI = 3
    PASS = 4

    def __init__(self, player: int, type_: int, executed: bool):
        self.player = player
        self.type = type_
        self.executed = executed


class NakiDecision(Decision):
    def __init__(self, player: int, naki: Naki, executed: bool):
        super().__init__(player, Decision.NAKI, executed)
        self.naki = naki

    def __repr__(self) -> str:
        return f"NakiDecision: player={self.player}, naki={self.naki}, executed={self.executed}"


class ReachDecision(Decision):
    def __init__(self, player: int, executed: bool):
        super().__init__(player, Decision.REACH, executed)

    def __repr__(self) -> str:
        return f"ReachDecision: player={self.player}, executed={self.executed}"


class AgariDecision(Decision):
    def __init__(self, player: int, executed: bool):
        super().__init__(player, Decision.AGARI, executed)

    def __repr__(self) -> str:
        return f"AgariDecision: player={self.player}, executed={self.executed}"


class PassDecision(Decision):
    def __init__(self, player: int, executed: bool):
        super().__init__(player, Decision.PASS, executed)

    def __repr__(self) -> str:
        return f"PassDecision: player={self.player}, executed={self.executed}"


def decision_mask(
    player: int, hand_tensors: np.ndarray, tile_idx: int, is_red: bool = False
):
    if tile_idx > 26:
        return _jihai_mask(player, hand_tensors, tile_idx)
    elif tile_idx % 9 in [0, 8]:
        return _one_and_nine_mask(player, hand_tensors, tile_idx, tile_idx % 9 == 0)
    elif tile_idx % 9 in [1, 7]:
        return _two_and_eight_mask(player, hand_tensors, tile_idx, tile_idx % 9 == 1)
    elif tile_idx % 9 in [2, 6]:
        return _three_and_seven_mask(player, hand_tensors, tile_idx, tile_idx % 9 == 2)
    elif tile_idx % 9 in [3, 5]:
        return _four_and_six_mask(player, hand_tensors, tile_idx, tile_idx % 9 == 3)
    elif tile_idx % 9 == 4:
        return _five_mask(player, hand_tensors, tile_idx, is_red)


def _one_and_nine_mask(
    player: int, hand_tensors: np.ndarray, tile_idx: int, is_one: bool = True
):
    decisions = [[] for _ in range(4)]
    coef = 1 if is_one else -1

    for p, hand_tensor in enumerate(hand_tensors):
        if p == player:
            continue

        if (p - player) % 4 == 1 and hand_tensor[tile_idx + coef] * hand_tensor[
            tile_idx + coef * 2
        ]:
            naki = Naki.from_chi_info(
                tile_idx // 9, 0 if is_one else 6, 0 if is_one else 2, False
            )
            decisions[p].append(NakiDecision(p, naki, False))
        if hand_tensor[tile_idx] == 2.0:
            naki = Naki.from_pon_info(tile_idx, 0, False, (player - p) % 4)
            decisions[p].append(NakiDecision(p, naki, False))
        if hand_tensor[tile_idx] == 3.0:
            naki = Naki.from_minkan_info(tile_idx, (player - p) % 4, 0)
            decisions[p].append(NakiDecision(p, naki, False))

    return decisions


def _two_and_eight_mask(
    player: int, hand_tensors: np.ndarray, tile_idx: int, is_two: bool = True
):
    decisions = [[] for _ in range(4)]
    coef = 1 if is_two else -1

    for p, hand_tensor in enumerate(hand_tensors):
        if p == player:
            continue

        if (p - player) % 4 == 1:
            if hand_tensor[tile_idx - 1] * hand_tensor[tile_idx + 1]:
                naki = Naki.from_chi_info(tile_idx // 9, 0 if is_two else 6, 1, False)
                decisions[p].append(NakiDecision(p, naki, False))
            if hand_tensor[tile_idx + coef] * hand_tensor[tile_idx + coef * 2]:
                naki = Naki.from_chi_info(
                    tile_idx // 9, 1 if is_two else 5, 0 if is_two else 2, False
                )
                decisions[p].append(NakiDecision(p, naki, False))

        if hand_tensor[tile_idx] == 2.0:
            naki = Naki.from_pon_info(tile_idx, 0, False, (player - p) % 4)
            decisions[p].append(NakiDecision(p, naki, False))

        if hand_tensor[tile_idx] == 3.0:
            naki = Naki.from_minkan_info(tile_idx, (player - p) % 4, 0)
            decisions[p].append(NakiDecision(p, naki, False))

    return decisions


def _three_and_seven_mask(
    player: int,
    hand_tensors: np.ndarray,
    tile_idx: int,
    is_three: bool = True,
):
    decisions = [[] for _ in range(4)]
    coef = 1 if is_three else -1

    for p, hand_tensor in enumerate(hand_tensors):
        if p == player:
            continue

        player_has_red = hand_tensors[p][34 + tile_idx // 9] > 0.0
        black_fives = hand_tensor[tile_idx] - float(player_has_red)

        if (p - player) % 4 == 1:
            if hand_tensor[tile_idx - 2 * coef] * hand_tensor[tile_idx - coef]:
                naki = Naki.from_chi_info(
                    tile_idx // 9, 0 if is_three else 6, 2 if is_three else 0, False
                )
                decisions[p].append(NakiDecision(p, naki, False))
            if hand_tensor[tile_idx - 1] * hand_tensor[tile_idx + 1]:
                naki = Naki.from_chi_info(tile_idx // 9, 1 if is_three else 5, 1, False)
                decisions[p].append(NakiDecision(p, naki, False))
            if hand_tensor[tile_idx + coef] * hand_tensor[tile_idx + 2 * coef]:
                if player_has_red:
                    naki = Naki.from_chi_info(
                        tile_idx // 9, 2 if is_three else 4, 0 if is_three else 2, True
                    )
                    decisions[p].append(NakiDecision(p, naki, False))
                    if black_fives > 0:
                        naki = Naki.from_chi_info(
                            tile_idx // 9,
                            2 if is_three else 4,
                            0 if is_three else 2,
                            False,
                        )
                        decisions[p].append(NakiDecision(p, naki, False))
                else:
                    naki = Naki.from_chi_info(
                        tile_idx // 9, 2 if is_three else 4, 0 if is_three else 2, False
                    )
                    decisions[p].append(NakiDecision(p, naki, False))

        if hand_tensor[tile_idx] == 2.0:
            naki = Naki.from_pon_info(tile_idx, 0, False, (player - p) % 4)
            decisions[p].append(NakiDecision(p, naki, False))

        if hand_tensor[tile_idx] == 3.0:
            naki = Naki.from_minkan_info(tile_idx, (player - p) % 4, 0)
            decisions[p].append(NakiDecision(p, naki, False))

    return decisions


def _four_and_six_mask(
    player: int,
    hand_tensors: np.ndarray,
    tile_idx: int,
    is_four: bool = True,
):
    decisions = [[] for _ in range(4)]
    coef = 1 if is_four else -1

    for p, hand_tensor in enumerate(hand_tensors):
        if p == player:
            continue

        player_has_red = hand_tensors[p][34 + tile_idx // 9] > 0.0
        black_fives = hand_tensor[tile_idx] - float(player_has_red)

        if (p - player) % 4 == 1:
            if hand_tensor[tile_idx - 2 * coef] * hand_tensor[tile_idx - coef]:
                naki = Naki.from_chi_info(
                    tile_idx // 9, 1 if is_four else 5, 2 if is_four else 0, False
                )
                decisions[p].append(NakiDecision(p, naki, False))
            if hand_tensor[tile_idx - 1] * hand_tensor[tile_idx + 1]:
                if player_has_red:
                    naki = Naki.from_chi_info(
                        tile_idx // 9, 2 if is_four else 4, 0 if is_four else 2, True
                    )
                    decisions[p].append(NakiDecision(p, naki, False))
                    if black_fives > 0:
                        naki = Naki.from_chi_info(
                            tile_idx // 9,
                            2 if is_four else 4,
                            0 if is_four else 2,
                            False,
                        )
                        decisions[p].append(NakiDecision(p, naki, False))
                else:
                    naki = Naki.from_chi_info(
                        tile_idx // 9, 2 if is_four else 4, 1, False
                    )
                    decisions[p].append(NakiDecision(p, naki, False))
            if hand_tensor[tile_idx + coef] * hand_tensor[tile_idx + 2 * coef]:
                if player_has_red:
                    naki = Naki.from_chi_info(
                        tile_idx // 9, 3, 0 if is_four else 2, True
                    )
                    decisions[p].append(NakiDecision(p, naki, False))
                    if black_fives > 0:
                        naki = Naki.from_chi_info(
                            tile_idx // 9, 3, 0 if is_four else 2, False
                        )
                        decisions[p].append(NakiDecision(p, naki, False))
                else:
                    naki = Naki.from_chi_info(
                        tile_idx // 9, 3, 0 if is_four else 2, False
                    )
                    decisions[p].append(NakiDecision(p, naki, False))

        if hand_tensor[tile_idx] == 2.0:
            naki = Naki.from_pon_info(tile_idx, 0, False, (player - p) % 4)
            decisions[p].append(NakiDecision(p, naki, False))

        if hand_tensor[tile_idx] == 3.0:
            naki = Naki.from_minkan_info(tile_idx, (player - p) % 4, 0)
            decisions[p].append(NakiDecision(p, naki, False))

    return decisions


def _five_mask(
    player: int,
    hand_tensors: np.ndarray,
    tile_idx: int,
    is_red: bool = False,
):
    decisions = [[] for _ in range(4)]

    for p, hand_tensor in enumerate(hand_tensors):
        if p == player:
            continue

        player_has_red = hand_tensors[p][34 + tile_idx // 9] > 0.0
        black_fives = hand_tensor[tile_idx] - float(player_has_red)

        if (p - player) % 4 == 1:
            if hand_tensor[tile_idx - 2] * hand_tensor[tile_idx - 1]:
                naki = Naki.from_chi_info(tile_idx // 9, 2, 2, is_red)
                decisions[p].append(NakiDecision(p, naki, False))
            if hand_tensor[tile_idx - 1] * hand_tensor[tile_idx + 1]:
                naki = Naki.from_chi_info(tile_idx // 9, 3, 1, is_red)
                decisions[p].append(NakiDecision(p, naki, False))
            if hand_tensor[tile_idx + 1] * hand_tensor[tile_idx + 2]:
                naki = Naki.from_chi_info(tile_idx // 9, 4, 0, is_red)
                decisions[p].append(NakiDecision(p, naki, False))

        if player_has_red:
            if hand_tensor[tile_idx] == 2.0:
                naki = Naki.from_pon_info(tile_idx, 1, True, (player - p) % 4)
                decisions[p].append(NakiDecision(p, naki, False))
            if black_fives == 2.0:
                naki = Naki.from_pon_info(tile_idx, 0, False, (player - p) % 4)
                decisions[p].append(NakiDecision(p, naki, False))
        else:
            if hand_tensor[tile_idx] == 2.0:
                naki = Naki.from_pon_info(
                    tile_idx, 1 if is_red else 0, is_red, (player - p) % 4
                )
                decisions[p].append(NakiDecision(p, naki, False))

        if hand_tensor[tile_idx] == 3.0:
            naki = Naki.from_minkan_info(tile_idx, (player - p) % 4, 1 if is_red else 0)
            decisions[p].append(NakiDecision(p, naki, False))

    return decisions


def _jihai_mask(player: int, hand_tensors: np.ndarray, tile_idx: int):
    decisions = [[] for _ in range(4)]

    for p, hand_tensor in enumerate(hand_tensors):
        if p == player:
            continue

        if hand_tensor[tile_idx] == 2.0:
            decisions[p].append(NakiDecision(p, Naki(0), False))
        if hand_tensor[tile_idx] == 3.0:
            decisions[p].append(NakiDecision(p, Naki(0), False))

    return decisions
