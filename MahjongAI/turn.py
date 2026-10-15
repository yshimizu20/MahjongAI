from typing import Dict, List, Optional

from MahjongAI.state import StateObject
from MahjongAI.decision import Decision
from MahjongAI.draw import Naki


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
        tsumo_blocked: Optional[str] = None,
    ):
        """tsumo_blocked (debug, read by the gates only): why a complete hand got no tsumo option (utils/agari.py
        NO_YAKU), else None."""
        super().__init__(player, HalfTurn.DURING, stateObj, encoding_idx)
        self.decisions = decisions
        self.tsumo_blocked = tsumo_blocked

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
        is_riichi_declaration: bool = False,
        after_call: Optional[Naki] = None,
    ):
        """is_riichi_declaration: the player chose this tile while declaring riichi (a free choice).
        Every later discard by a player in riichi is forced: it must be the drawn tile.
        after_call: the chi / pon this discard directly follows (kuikae applies), else None. A kan is
        followed by a rinshan draw, and riichi needs a closed hand, so neither can come with a call."""
        assert after_call is None or after_call.is_chi() or after_call.is_pon(), f"after_call must be a chi / pon: {after_call!r}"
        assert not (is_riichi_declaration and after_call is not None), "a riichi declaration cannot follow a call"
        super().__init__(player, HalfTurn.DISCARD, stateObj, encoding_idx)
        self.discarded_tile = discarded_tile
        self.is_riichi_declaration = is_riichi_declaration
        self.after_call = after_call

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
        responder_states: Dict[int, StateObject],
        ron_blocked: Optional[Dict[int, str]] = None,
    ):
        """
        player: the discarder (or the kakan player for chankan). stateObj: the DISCARDER's state, kept
        for debugging only. responder_states[p]: the state as responder p sees it; the model must use this.
        encoding_idx: history length INCLUDING the discard/kakan token being answered.
        ron_blocked (debug, read by the gates only): responder -> why the answered tile completes their hand but
        gives no ron option (a furiten reason or NO_YAKU, utils/agari.py).
        """
        super().__init__(player, HalfTurn.POST, stateObj, encoding_idx)
        self.decisions = decisions
        self.responder_states = responder_states
        self.ron_blocked = dict(ron_blocked or {})

    def __repr__(self):
        return "PostTurn Object | Player: {} | Decisions: {}".format(
            self.player, self.decisions
        )


# class EmptyTurn(HalfTurn):
#     def __init__(self, encoding_idx: int):
#         super().__init__(-1, HalfTurn.EMPTY, None, encoding_idx)
