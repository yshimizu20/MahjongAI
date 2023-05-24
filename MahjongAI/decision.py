import numpy as np
from MahjongAI.draw import Naki


class Decision:
    NAKI = 1
    REACH = 2
    AGARI = 3
    PASS = 4

    def __init__(self, player: int, type_: int):
        self.player = player
        self.type = type_


class NakiDecision(Decision):
    def __init__(self, player: int, naki: Naki, executed: bool):
        super().__init__(player, Decision.NAKI)
        self.naki = naki
        self.executed = executed

    def __repr__(self) -> str:
        return f"NakiDecision: player={self.player}, naki={self.naki}, executed={self.executed}"


class ReachDecision(Decision):
    def __init__(self, player: int, executed: bool):
        super().__init__(player, Decision.REACH)
        self.executed = executed

    def __repr__(self) -> str:
        return f"ReachDecision: player={self.player}, executed={self.executed}"


class AgariDecision(Decision):
    def __init__(self, player: int, executed: bool):
        super().__init__(player, Decision.AGARI)
        self.executed = executed

    def __repr__(self) -> str:
        return f"AgariDecision: player={self.player}, executed={self.executed}"


class PassDecision(Decision):
    def __init__(self, player: int, executed: bool):
        super().__init__(player, Decision.PASS)
        self.executed = executed

    def __repr__(self) -> str:
        return f"PassDecision: player={self.player}, executed={self.executed}"


def _one_and_nine_mask(
    player: int, hand_tensors: np.ndarray, tile_idx: int, is_one: bool = True
):
    decisions = []
    coef = 1 if is_one else -1

    for p, hand_tensor in enumerate(hand_tensors):
        if p == player:
            continue

        if hand_tensor[tile_idx] * hand_tensor[tile_idx + coef]:
            # TODO: get Naki code
            decisions.append(NakiDecision(p, Naki(0), False))
        if hand_tensor[tile_idx] == 2.0:
            decisions.append(NakiDecision(p, Naki(0), False))
        if hand_tensor[tile_idx] == 3.0:
            decisions.append(NakiDecision(p, Naki(0), False))

    return decisions


def _two_and_eight_mask(
    player: int, hand_tensors: np.ndarray, tile_idx: int, is_two: bool = True
):
    decisions = []
    coef = 1 if is_two else -1

    for p, hand_tensor in enumerate(hand_tensors):
        if p == player:
            continue

        if hand_tensor[tile_idx - 1] * hand_tensor[tile_idx + 1]:
            decisions.append(NakiDecision(p, Naki(0), False))
        if hand_tensor[tile_idx + coef] * hand_tensor[tile_idx + coef * 2]:
            decisions.append(NakiDecision(p, Naki(0), False))
        if hand_tensor[tile_idx] == 2.0:
            decisions.append(NakiDecision(p, Naki(0), False))
        if hand_tensor[tile_idx] == 3.0:
            decisions.append(NakiDecision(p, Naki(0), False))

    return decisions


def _three_to_seven_mask(player: int, hand_tensors: np.ndarray, tile_idx: int):
    decisions = []
    num = tile_idx % 9
    red_idx = 34 + tile_idx // 9

    for p, hand_tensor in enumerate(hand_tensors):
        if p == player:
            continue

        if hand_tensor[tile_idx - 2] * hand_tensor[tile_idx - 1]:
            decisions.append(NakiDecision(p, Naki(0), False))
        if hand_tensor[tile_idx - 1] * hand_tensor[tile_idx + 1]:
            decisions.append(NakiDecision(p, Naki(0), False))
        if hand_tensor[tile_idx + 1] * hand_tensor[tile_idx + 2]:
            decisions.append(NakiDecision(p, Naki(0), False))
        if hand_tensor[tile_idx] == 2.0:
            decisions.append(NakiDecision(p, Naki(0), False))
        if hand_tensor[tile_idx] == 3.0:
            decisions.append(NakiDecision(p, Naki(0), False))

        if hand_tensor[red_idx]:
            if num == 5:
                if hand_tensor[tile_idx] == 1:
                    decisions.append(NakiDecision(p, Naki(0), False))
                elif hand_tensor[tile_idx] == 2:
                    decisions.append(NakiDecision(p, Naki(0), False))
            if num == 3 and hand_tensor[tile_idx + 1]:
                decisions.append(NakiDecision(p, Naki(0), False))
            if num == 4:
                if hand_tensor[tile_idx - 1]:
                    decisions.append(NakiDecision(p, Naki(0), False))
                if hand_tensor[tile_idx + 2]:
                    decisions.append(NakiDecision(p, Naki(0), False))
            if num == 6:
                if hand_tensor[tile_idx - 2]:
                    decisions.append(NakiDecision(p, Naki(0), False))
                if hand_tensor[tile_idx + 1]:
                    decisions.append(NakiDecision(p, Naki(0), False))
            if num == 7 and hand_tensor[tile_idx - 1]:
                decisions.append(NakiDecision(p, Naki(0), False))

    return decisions


def _jihai_mask(player: int, hand_tensors: np.ndarray, tile_idx: int):
    decisions = []

    for p, hand_tensor in enumerate(hand_tensors):
        if p == player:
            continue

        if hand_tensor[tile_idx] == 2.0:
            decisions.append(NakiDecision(p, Naki(0), False))
        if hand_tensor[tile_idx] == 3.0:
            decisions.append(NakiDecision(p, Naki(0), False))

    return decisions


def decision_mask(player, hand_tensors: np.ndarray, tile_idx: int):
    if tile_idx > 33:
        return _three_to_seven_mask(player, hand_tensors, 4 + 9 * (tile_idx - 34))
    elif tile_idx > 26:
        return _jihai_mask(player, hand_tensors, tile_idx)
    elif tile_idx % 9 in [0, 8]:
        return _one_and_nine_mask(player, hand_tensors, tile_idx)
    elif tile_idx % 9 in [1, 7]:
        return _two_and_eight_mask(player, hand_tensors, tile_idx)
    else:
        return _three_to_seven_mask(player, hand_tensors, tile_idx)
