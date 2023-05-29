import numpy as np
from typing import List
from MahjongAI.draw import Naki


class StateObject:
    def __init__(
        self,
        remaining_turns: int,
        hand_tensor: List[int],
        remaining_tiles: np.ndarray,
        remaining_tiles_pov: np.ndarray,
        sutehai_tensor: np.ndarray,
        reaches: List[int],
        melds: List[List[Naki]],
        scores: List[int],
        kyotaku: int,
        honba: int,
        dora: List[int],
    ):
        self.remaining_turns = remaining_turns
        self.hand_tensor = hand_tensor
        self.remaining_tiles = remaining_tiles
        self.remaining_tiles_pov = remaining_tiles_pov
        self.sutehai_tensor = sutehai_tensor
        self.reaches = reaches
        self.melds = melds
        self.scores = scores
        self.kyotaku = kyotaku
        self.honba = honba
        self.dora = dora
