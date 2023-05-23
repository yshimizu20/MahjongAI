from MahjongAI.draw import Draw, Naki, Tsumo
from MahjongAI.discard import Discard
from MahjongAI.state import StateObject
from abc import abstractmethod


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
        self.pre_decisions = []
        self.post_decisions = []

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
        self.post_decisions = []

    def is_tsumogiri(self):
        super().is_tsumogiri()
        return False
