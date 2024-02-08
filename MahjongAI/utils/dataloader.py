import os
import torch
import numpy as np

from MahjongAI.process_xml import process, InvalidGameException
from MahjongAI.turn import HalfTurn, DuringTurn, DiscardTurn, PostTurn
from MahjongAI.utils.constants import DECISION_AGARI_IDX, DECISION_REACH_IDX, DECISION_NAKI_IDX


class DataLoader:
    def __init__(self, path: str):
        self.path = path
        self.file_list = os.listdir(self.path)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index == len(self.file_list):
            self.current_index = 0

        filename = self.file_list[self.current_index]
        print(filename)
        self.current_index += 1

        try:
            turns = process(os.path.join(self.path, filename))
            return turns

        except InvalidGameException as e:
            print(e.msg)
            return self.__next__()

    def build_tensor(self, turns):
        during_turns = []
        discard_turns = []
        post_turns = []

        for turn in turns:
            if turn.type_ == HalfTurn.DURING:
                during_turns.append(turn)
            elif turn.type_ == HalfTurn.DISCARD:
                discard_turns.append(turn)
            elif turn.type_ == HalfTurn.POST:
                post_turns.append(turn)
            else:
                raise ZeroDivisionError

    def _build_during_tensor(self, during_turns):
        L = len(during_turns)
        X = []
        y = torch.zeros(L, 1, dtype=torch.long)
        filter_tensors = torch.zeros(L, 71, dtype=torch.long)

        for i, turn in enumerate(during_turns):
            n_decisions = sum(map(len, turn.decisions))

            if n_decisions == 0:
                continue

            for decision_idx, decision_type in enumerate(turn.decisions):
                if len(decision_type) == 0:
                    continue

                # populate filter tensor and y tensor
                if decision_idx == DECISION_AGARI_IDX:
                    filter_tensors[i, 1] = 1
                    if decision_type[0].executed:
                        y[i] = 1

                elif decision_idx == DECISION_REACH_IDX:
                    filter_tensors[i, 2] = 1
                    if decision_type[0].executed:
                        y[i] = 2

                elif decision_idx == DECISION_NAKI_IDX:
                    for meld in decision_type:
                        idx = meld.get_during_turn_filter_idx()
                        filter_tensors[i, idx] = 1
                        if meld.executed:
                            y[i] = idx
                
                filter_tensors[i, 0] = 1 # pass is always an option

            # TODO: create state object tensor
        
        return torch.cat(X, dim=0), y, filter_tensors

    def _build_discard_tensor(self, discard_turns):
        tensors = []

        for turn in discard_turns:
            # TODO: create a filter tensor
            # TODO: make a translation table
            # TODO: create state object tensor
            pass
        
        return torch.cat(tensors, dim=0)

    def _build_post_tensor(self, post_turns):
        pass
