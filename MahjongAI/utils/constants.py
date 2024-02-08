import numpy as np


REMAINING_TILES = np.ones(37, dtype=np.float32) * 4
REMAINING_TILES[34:37] = 1.0


TILE2IDX = [[i // 4] for i in range(136)]
TILE2IDX[16] += [34]
TILE2IDX[52] += [35]
TILE2IDX[88] += [36]
TILE2IDX = np.array(TILE2IDX, dtype=object)


REMAINING_TSUMO = 70


YAOCHU_TENSOR = np.zeros(37)
YAOCHU_TENSOR[[0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 34]] = 1.0


DECISION_AGARI_IDX = 0
DECISION_REACH_IDX = 1
DECISION_NAKI_IDX = 2
