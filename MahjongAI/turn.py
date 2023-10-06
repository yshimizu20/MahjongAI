from typing import List

from MahjongAI.state import StateObject
from MahjongAI.decision import Decision


class HalfTurn:
    DURING = 0
    DISCARD = 1
    POST = 2

    def __init__(
        self,
        player: int,
        type_: int,
        stateObj: StateObject,
        encoding_tokens: List[int],
    ):
        self.player = player
        self.type_ = type_
        self.stateObj = stateObj
        self.encoding_tokens = encoding_tokens


class DuringTurn(HalfTurn):
    def __init__(
        self,
        player: int,
        stateObj: StateObject,
        decisions: List[Decision],
        encoding_tokens: List[int],
    ):
        super().__init__(player, HalfTurn.DURING, stateObj, encoding_tokens)
        self.decisions = decisions


class DiscardTurn(HalfTurn):
    def __init__(
        self,
        player: int,
        stateObj: StateObject,
        discarded_tile: int,  # 0-135
        encoding_tokens: List[int],
    ):
        super().__init__(player, HalfTurn.DISCARD, stateObj, encoding_tokens)
        self.discarded_tile = discarded_tile


class PostTurn(HalfTurn):
    def __init__(
        self,
        player: int,
        stateObj: StateObject,
        decisions: List[List[Decision]],
        encoding_tokens: List[int],
    ):
        super().__init__(player, HalfTurn.POST, stateObj, encoding_tokens)
        self.decisions = decisions
