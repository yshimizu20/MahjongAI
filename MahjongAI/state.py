import numpy as np
from typing import List
from MahjongAI.draw import Naki


class StateObject:

    def __init__(
        self,
        remaining_turns: int,
        hand_tensor: np.ndarray,
        remaining_tiles: np.ndarray,
        remaining_tiles_pov: np.ndarray,
        sutehai_tensor: np.ndarray,
        is_menzen: List[int],
        double_reaches: List[int],
        reaches: List[int],
        ippatsu: List[int],
        melds: List[List[Naki]],
        scores: List[int],
        kyotaku: int,
        honba: int,
        dora: List[int],
        parent_tensor: np.ndarray,
        rounds_remaining: int,
        parent_rounds_remaining: List[int],
        remaining_tsumo: int,
    ):
        self.remaining_turns = remaining_turns
        self.hand_tensor = hand_tensor
        self.remaining_tiles = remaining_tiles
        self.remaining_tiles_pov = remaining_tiles_pov
        self.sutehai_tensor = sutehai_tensor
        self.is_menzen = is_menzen
        self.double_reaches = double_reaches
        self.reaches = reaches
        self.ippatsu = ippatsu
        self.melds = melds
        self.scores = scores
        self.kyotaku = kyotaku
        self.honba = honba
        self.dora = dora
        self.parent_tensor = parent_tensor
        self.rounds_remaining = rounds_remaining
        self.parent_rounds_remaining = parent_rounds_remaining
        self.remaining_tsumo = remaining_tsumo
