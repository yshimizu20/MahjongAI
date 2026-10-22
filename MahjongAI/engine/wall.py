"""The wall of one round in Tenhou's layout.

136 tiles (Tenhou's 136-ids; the red fives are 16, 52, 88). The first 122 are the live wall, the last 14 the dead
wall: 4 rinshan tiles, 5 dora indicators, 5 ura indicators. The deal takes 4-4-4 tiles per seat, dealer first,
then 1 each (hai0-3 list the tiles in the order dealt, as Tenhou does); 70 live draws remain. A kan takes its
rinshan tile from the dead wall and moves the last live tile (the haitei tile) into it, so every draw, rinshan
included, lowers `draws_left` by one: 70 draws per round, the parser's `remaining_tsumo`."""
import random
from typing import List, Sequence

N_TILES = 136
DEAD_WALL = 14
N_RINSHAN = 4
N_INDICATORS = 5
LIVE_DRAWS = 70


class Wall:
    def __init__(self, order: Sequence[int]):
        """order: the 136 ids, order[0] dealt first."""
        assert sorted(order) == list(range(N_TILES)), "the wall must hold every id once"
        live, dead = list(order[:N_TILES - DEAD_WALL]), list(order[N_TILES - DEAD_WALL:])
        self._live = live
        self._rinshan = dead[:N_RINSHAN]
        self._indicators = dead[N_RINSHAN:N_RINSHAN + N_INDICATORS]
        self._ura = dead[N_RINSHAN + N_INDICATORS:]
        self._next = 0  # the next live tile
        self._kans = 0
        self._dealt = False

    @classmethod
    def shuffled(cls, rng: random.Random) -> "Wall":
        tiles = list(range(N_TILES))
        rng.shuffle(tiles)
        return cls(tiles)

    def deal(self, oya: int) -> List[List[int]]:
        """hands[seat], 13 tiles each in the order dealt (4-4-4 per seat from the dealer, then 1 each)."""
        assert not self._dealt, "dealt twice"
        self._dealt = True
        hands: List[List[int]] = [[] for _ in range(4)]
        for _ in range(3):
            for k in range(4):
                hands[(oya + k) % 4] += self._take(4)
        for k in range(4):
            hands[(oya + k) % 4] += self._take(1)
        assert self.draws_left == LIVE_DRAWS, self.draws_left
        return hands

    def _take(self, n: int) -> List[int]:
        tiles = self._live[self._next:self._next + n]
        assert len(tiles) == n, "the live wall is empty"
        self._next += n
        return tiles

    @property
    def draws_left(self) -> int:
        """Live draws left, the haitei tile included; a kan moved one live tile away per kan."""
        return len(self._live) - self._next

    @property
    def kans(self) -> int:
        return self._kans

    def draw(self) -> int:
        assert self._dealt and self.draws_left > 0, "no live draw left"
        return self._take(1)[0]

    def rinshan_draw(self) -> int:
        """The kan player's replacement tile; the haitei tile joins the dead wall."""
        assert self._kans < N_RINSHAN, "a fifth kan"
        assert self.draws_left > 0, "a kan without a live draw left"
        tile = self._rinshan[self._kans]
        self._kans += 1
        self._live.pop()
        return tile

    def indicator(self, i: int) -> int:
        """Dora indicator i: 0 the initial one, i the one flipped for the i-th kan."""
        return self._indicators[i]

    def ura_indicators(self, n: int) -> List[int]:
        """The ura indicators under the first n indicators."""
        return self._ura[:n]
