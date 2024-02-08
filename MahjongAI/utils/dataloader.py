import os
import torch
import numpy as np

from MahjongAI.process_xml import process, InvalidGameException
from MahjongAI.turn import HalfTurn, DuringTurn, DiscardTurn, PostTurn


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
        for turn in during_turns:
            if len(turn.decisions) == 0:
                continue

    def _build_discard_tensor(self, discard_turns):
        pass

    def _build_post_tensor(self, post_turns):
        pass
