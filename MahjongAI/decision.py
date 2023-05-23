from MahjongAI.draw import Naki
import numpy as np


class Decision:
    NAKI = 1
    REACH = 2
    AGARI = 3

    def __init__(self, player: int):
        self.player = player


class NakiDecision(Decision):
    def __init__(self, player: int, naki: Naki, executed: bool):
        super().__init__(player)
        self.naki = naki
        self.executed = executed


class ReachDecision(Decision):
    def __init__(self, player: int, executed: bool):
        super().__init__(player)
        self.executed = executed


class AgariDecision(Decision):
    def __init__(self, player: int, executed: bool):
        super().__init__(player)
        self.executed = executed


class PassDecision(Decision):
    def __init__(self, player: int, executed: bool):
        super().__init__(player)
        self.executed = executed


def _one_and_nine_mask(
    player: int, hand_tensor: np.ndarray, tile_idx: int, is_one: bool = True
):
    decisions = []
    coef = 1 if is_one else -1
    for p in range(4):
        if p == player:
            continue
        if hand_tensor[tile_idx] * hand_tensor[tile_idx + coef]:
            decisions.append(NakiDecision(p, Naki(), False))
        if hand_tensor[tile_idx] == 2.0:
            decisions.append(NakiDecision(p, Naki(), False))
        if hand_tensor[tile_idx] == 3.0:
            decisions.append(NakiDecision(p, Naki(), False))
    return decisions


def _two_and_eight_mask(
    player: int, hand_tensor: np.ndarray, tile_idx: int, is_two: bool = True
):
    decisions = []
    coef = 1 if is_two else -1
    for p in range(4):
        if p == player:
            continue
        if hand_tensor[tile_idx - 1] * hand_tensor[tile_idx + 1]:
            decisions.append(NakiDecision(p, Naki(), False))
        if hand_tensor[tile_idx + coef] * hand_tensor[tile_idx + coef * 2]:
            decisions.append(NakiDecision(p, Naki(), False))
        if hand_tensor[tile_idx] == 2.0:
            decisions.append(NakiDecision(p, Naki(), False))
        if hand_tensor[tile_idx] == 3.0:
            decisions.append(NakiDecision(p, Naki(), False))
    return decisions


def _three_to_seven_mask(player: int, hand_tensor: np.ndarray, tile_idx: int):
    decisions = []
    num = tile_idx % 9
    red_idx = 34 + tile_idx // 9

    for p in range(4):
        if p == player:
            continue
        if hand_tensor[tile_idx - 2] * hand_tensor[tile_idx - 1]:
            decisions.append(NakiDecision(p, Naki(), False))
        if hand_tensor[tile_idx - 1] * hand_tensor[tile_idx + 1]:
            decisions.append(NakiDecision(p, Naki(), False))
        if hand_tensor[tile_idx + 1] * hand_tensor[tile_idx + 2]:
            decisions.append(NakiDecision(p, Naki(), False))
        if hand_tensor[tile_idx] == 2.0:
            decisions.append(NakiDecision(p, Naki(), False))
        if hand_tensor[tile_idx] == 3.0:
            decisions.append(NakiDecision(p, Naki(), False))
        if hand_tensor[red_idx]:
            if num == 5:
                if hand_tensor[tile_idx] == 1:
                    decisions.append(NakiDecision(p, Naki(), False))
                elif hand_tensor[tile_idx] == 2:
                    decisions.append(NakiDecision(p, Naki(), False))
            if num == 3 and hand_tensor[tile_idx + 1]:
                decisions.append(NakiDecision(p, Naki(), False))
            if num == 4:
                if hand_tensor[tile_idx - 1]:
                    decisions.append(NakiDecision(p, Naki(), False))
                if hand_tensor[tile_idx + 2]:
                    decisions.append(NakiDecision(p, Naki(), False))
            if num == 6:
                if hand_tensor[tile_idx - 2]:
                    decisions.append(NakiDecision(p, Naki(), False))
                if hand_tensor[tile_idx + 1]:
                    decisions.append(NakiDecision(p, Naki(), False))
            if num == 7 and hand_tensor[tile_idx - 1]:
                decisions.append(NakiDecision(p, Naki(), False))
    return decisions


def _jihai_mask(player: int, hand_tensor: np.ndarray, tile_idx: int):
    decisions = []
    for p in range(4):
        if p == player:
            continue
        if hand_tensor[tile_idx] == 2.0:
            decisions.append(NakiDecision(p, Naki(), False))
        if hand_tensor[tile_idx] == 3.0:
            decisions.append(NakiDecision(p, Naki(), False))
    return decisions


def decision_mask(player, hand_tensor: np.ndarray, tile_idx: int):
    if tile_idx > 33:
        return _three_to_seven_mask(player, hand_tensor, 4 + 9 * tile_idx)
    elif tile_idx > 26:
        return _jihai_mask(player, hand_tensor, tile_idx)
    elif tile_idx % 9 in [0, 8]:
        return _one_and_nine_mask(player, hand_tensor, tile_idx)
    else:
        assert tile_idx % 9 in [1, 7]
        return _two_and_eight_mask(player, hand_tensor, tile_idx)
