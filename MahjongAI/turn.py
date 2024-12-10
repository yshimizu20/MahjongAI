from typing import List

from MahjongAI.state import StateObject
from MahjongAI.decision import Decision


class HalfTurn:
    EMPTY = -1
    DURING = 0
    DISCARD = 1
    POST = 2

    def __init__(
        self,
        player: int,
        type_: int,
        stateObj: StateObject,
        encoding_idx: int,
    ):
        self.player = player
        self.type_ = type_
        self.stateObj = stateObj
        self.encoding_idx = encoding_idx


class DuringTurn(HalfTurn):
    def __init__(
        self,
        player: int,
        stateObj: StateObject,
        decisions: List[List[Decision]],
        encoding_idx: int,
    ):
        super().__init__(player, HalfTurn.DURING, stateObj, encoding_idx)
        self.decisions = decisions

    def __repr__(self):
        return "DuringTurn Object | Player: {} | Decisions: {}".format(
            self.player, self.decisions
        )


class DiscardTurn(HalfTurn):
    def __init__(
        self,
        player: int,
        stateObj: StateObject,
        discarded_tile: int,  # 0-135
        encoding_idx: int,
    ):
        super().__init__(player, HalfTurn.DISCARD, stateObj, encoding_idx)
        self.discarded_tile = discarded_tile

    def __repr__(self):
        return "DiscardTurn Object | Player: {} | Discarded Tile: {}".format(
            self.player, self.discarded_tile
        )


class PostTurn(HalfTurn):
    def __init__(
        self,
        player: int,
        stateObj: StateObject,
        decisions: List[List[List[Decision]]],
        encoding_idx: int,
    ):
        super().__init__(player, HalfTurn.POST, stateObj, encoding_idx)
        self.decisions = decisions

    def __repr__(self):
        return "PostTurn Object | Player: {} | Decisions: {}".format(
            self.player, self.decisions
        )


# class EmptyTurn(HalfTurn):
#     def __init__(self, encoding_idx: int):
#         super().__init__(-1, HalfTurn.EMPTY, None, encoding_idx)
