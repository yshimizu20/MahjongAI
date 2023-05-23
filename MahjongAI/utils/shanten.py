import numpy as np
from MahjongAI.utils.constants import YAOCHU_TENSOR


class ShantenSolver:
    def _init(self, hand_tensor: np.ndarray):
        self.hand_tensor = hand_tensor
        self.n_melds = (14 - self.hand_tensor.sum()) // 3
        self.n_mentsu = self.n_melds + sum(self.hand_tensor[27:] >= 3.0)
        self.n_tatsu = 0
        self.n_toitsu = sum(self.hand_tensor[27:] == 2.0)
        self.n_kuttsuki = 0
        self.best_shanten = 14
        self.tune = 0

    def add_shuntsu(self, idx: int):
        self.hand_tensor[idx : idx + 3] -= 1
        self.n_mentsu += 1

    def remove_shuntsu(self, idx: int):
        self.hand_tensor[idx : idx + 3] += 1
        self.n_mentsu -= 1

    def add_koutsu(self, idx: int):
        self.hand_tensor[idx] -= 3
        self.n_mentsu += 1

    def remove_koutsu(self, idx: int):
        self.hand_tensor[idx] += 3
        self.n_mentsu -= 1

    def add_toitsu(self, idx: int):
        self.hand_tensor[idx] -= 2
        self.n_toitsu += 1

    def remove_toitsu(self, idx: int):
        self.hand_tensor[idx] += 2
        self.n_toitsu -= 1

    def add_taatsu(self, idx: int, which: int):
        self.hand_tensor[idx : idx + 3] -= 1
        self.hand_tensor[idx + which] += 1
        self.n_tatsu += 1

    def remove_taatsu(self, idx: int, which: int):
        self.hand_tensor[idx : idx + 3] += 1
        self.hand_tensor[idx + which] -= 1
        self.n_tatsu -= 1

    def add_kuttsuki(self, idx: int, tune: int):
        self.hand_tensor[idx] -= 1
        self.n_kuttsuki += 1
        if tune:
            self.tune += 1

    def remove_kuttsuki(self, idx: int, tune: int):
        self.hand_tensor[idx] += 1
        self.n_kuttsuki -= 1
        if tune:
            self.tune -= 1

    def solve_normal(self, idx: int = 0):
        while idx < 27:
            if self.hand_tensor[idx]:
                break
            idx += 1

        if idx == 27:
            shanten = min(
                self.best_shanten, 8 - self.n_mentsu * 2 - self.n_tatsu - self.n_toitsu
            )
            potential_mentsu = max(0, self.n_toitsu - 1) + self.n_tatsu
            if potential_mentsu + self.n_mentsu > 4:
                shanten += potential_mentsu + self.n_mentsu - 4
            if self.tune and not self.n_toitsu:
                shanten += 1
            self.best_shanten = min(self.best_shanten, shanten)
            return

        num = idx % 9

        def process_shuntsu(idx: int, tune: bool = False):
            if num < 7 and self.hand_tensor[idx + 2]:
                if self.hand_tensor[idx + 1]:
                    self.add_shuntsu(idx)
                    self.solve_normal(idx)
                    self.remove_shuntsu(idx)
                self.add_taatsu(idx, 1)
                self.solve_normal(idx)
                self.remove_taatsu(idx, 1)

            if num < 8 and self.hand_tensor[idx + 1]:
                self.add_taatsu(idx, 2)
                self.solve_normal(idx)
                self.remove_taatsu(idx, 2)

            self.add_kuttsuki(idx, tune)
            self.solve_normal(idx)
            self.remove_kuttsuki(idx, tune)

        if self.hand_tensor[idx] == 4:
            self.add_koutsu(idx)
            process_shuntsu(idx, tune=True)
            self.solve_normal(idx + 1)
            self.remove_koutsu(idx)

            self.add_toitsu(idx)
            process_shuntsu(idx)
            self.remove_toitsu(idx)

        elif self.hand_tensor[idx] == 3:
            self.add_koutsu(idx)
            self.solve_normal(idx + 1)
            self.remove_koutsu(idx)

            self.add_toitsu(idx)
            self.solve_normal(idx)
            self.remove_toitsu(idx)

        elif self.hand_tensor[idx] == 2:
            self.add_toitsu(idx)
            self.solve_normal(idx + 1)
            self.remove_toitsu(idx)

            process_shuntsu(idx)

        elif self.hand_tensor[idx] == 1:
            process_shuntsu(idx)

    def solve_chitoitsu(self):
        self.best_shanten = min(self.best_shanten, 6 - sum(self.hand_tensor >= 2.0))

    def solve_kokushi(self):
        yaochu_arr = self.hand_tensor * YAOCHU_TENSOR
        self.best_shanten = min(
            self.best_shanten, 13 - sum(yaochu_arr >= 1.0) - (max(yaochu_arr) - 1.0)
        )

    def solve(self):
        self.solve_normal()
        self.solve_chitoitsu()
        self.solve_kokushi()

        return self.best_shanten
