from abc import abstractmethod
from typing import List
from MahjongAI.draw import Draw, Naki, Tsumo
from MahjongAI.discard import Discard
from MahjongAI.state import StateObject
from MahjongAI.decision import Decision


class Turn:
    TSUMO = 0
    NAKI = 1

    def __init__(
        self,
        player: int,
        type_: int,
        draw: Draw,
        stateObj: StateObject = None,
        discard: Discard = None,
    ):
        self.player = player
        self.type = type_
        self.draw = draw
        self.stateObj = stateObj
        self.discard = discard
        self.pre_decisions = []
        self.post_decisions = []

    @abstractmethod
    def is_tsumogiri(self):
        if self.draw is None or self.discard is None:
            raise ValueError("draw and discard must be set")


class TsumoTurn(Turn):
    def __init__(
        self,
        player: int,
        draw: Tsumo = None,
        stateObj: StateObject = None,
        discard: Discard = None,
    ):
        super().__init__(player, Turn.TSUMO, draw, stateObj, discard)

    def is_tsumogiri(self):
        super().is_tsumogiri()
        return self.draw.tile == self.discard.tile


class NakiTurn(Turn):
    def __init__(
        self,
        player: int,
        naki: Naki,
        stateObj: StateObject = None,
        discard: Discard = None,
    ):
        super().__init__(player, Turn.NAKI, naki, stateObj, discard)

    def is_tsumogiri(self):
        super().is_tsumogiri()
        return False


class HalfTurn:
    DURING = 0
    DISCARD = 1
    POST = 2

    def __init__(
        self,
        player: int,
        type_: int,
        enc_indices: List[int],
        stateObj: StateObject,
    ):
        self.player = player
        self.type_ = type_
        self.enc_indices = enc_indices[:]
        self.stateObj = stateObj


class DuringTurn(HalfTurn):
    def __init__(
        self,
        player: int,
        enc_indices: List[int],
        stateObj: StateObject,
        decisions: List[Decision],
    ):
        super().__init__(player, HalfTurn.DURING, enc_indices, stateObj)
        self.decisions = decisions


class DiscardTurn(HalfTurn):
    def __init__(
        self,
        player: int,
        enc_indices: List[int],
        stateObj: StateObject,
        discarded_tile: int,  # 0-136
    ):
        super().__init__(player, HalfTurn.DISCARD, enc_indices, stateObj)
        self.discarded_tile = discarded_tile


class PostTurn(HalfTurn):
    def __init__(
        self,
        player: int,
        enc_indices: List[int],
        stateObj: StateObject,
        decisions: List[List[Decision]],
    ):
        super().__init__(player, HalfTurn.POST, enc_indices, stateObj)
        self.decisions = decisions
