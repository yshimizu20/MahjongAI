import numpy as np


REMAINING_TILES = np.ones(37, dtype=np.int32) * 4
REMAINING_TILES[[4, 13, 22]] = 3
REMAINING_TILES[34:37] = 1


TILE2IDX = np.array([i // 4 for i in range(136)], dtype=np.int32)
TILE2IDX[16] = 34
TILE2IDX[52] = 35
TILE2IDX[88] = 36


REMAINING_TSUMO = 70


YAOCHU_TENSOR = np.zeros(37)
YAOCHU_TENSOR[[0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 34]] = 1.0
