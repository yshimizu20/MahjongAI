import numpy as np


REMAINING_TILES = np.ones(37, dtype=np.float32) * 4
REMAINING_TILES[34:37] = 1.0


TILE2IDX = [[i // 4] for i in range(136)]
TILE2IDX[16] += [34]
TILE2IDX[52] += [35]
TILE2IDX[88] += [36]
TILE2IDX = np.array(TILE2IDX, dtype=object)
RED_FIVE_SLOTS = ((4, 34), (13, 35), (22, 36))  # (plain slot, red slot) of 5m, 5p, 5s in the 37-index


REMAINING_TSUMO = 70
MAX_SEQUENCE_LENGTH = 150


DECISION_AGARI_IDX = 0
DECISION_REACH_IDX = 1
DECISION_NAKI_IDX = 2
DECISION_ABORT_IDX = 3  # DuringTurn only: kyuushu kyuuhai

PAD_ID = 0


class EventTypes:
    EMPTY = 0
    DISCARD = 1
    NAKI = 2
    RIICHI = 3
    NEW_DORA = 4


class EventStateTypes:
    EMPTY = 0
    DISCARD = 1
    DURING = 2
    POST = 3
    NEW_DORA = 4
