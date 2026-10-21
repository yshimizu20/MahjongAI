"""T-053 engine: the pure tables (settlement, round transition, owari, meld codes), the wall, the match rotation."""
import random

from MahjongAI.engine import engine_lib, match
from MahjongAI.engine.game import header
from MahjongAI.engine.wall import LIVE_DRAWS, N_TILES, Wall


def test_tables():
    engine_lib.check_tables()


def test_wall_deal_order_and_partition():
    wall = Wall(list(range(N_TILES)))
    hands = wall.deal(oya=1)
    assert hands[1] == [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48]
    assert hands[2] == [4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 49]
    assert hands[0] == [12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 51]
    assert sorted(t for h in hands for t in h) == list(range(52))
    assert wall.draws_left == LIVE_DRAWS
    assert [wall.draw() for _ in range(3)] == [52, 53, 54]
    assert wall.indicator(0) == 126 and wall.ura_indicators(2) == [131, 132]
    assert wall.rinshan_draw() == 122 and wall.draws_left == LIVE_DRAWS - 4  # 3 draws + the haitei tile moved
    shuffled = Wall.shuffled(random.Random(0))
    hands = shuffled.deal(0)
    assert len({t for h in hands for t in h}) == 52 and shuffled.draws_left == LIVE_DRAWS


def test_match_rotation_and_header():
    specs = ["a", "b", "c", "d"]
    assert match.seats_for_game(specs, 0) == ["a", "b", "c", "d"]
    assert match.seats_for_game(specs, 1) == ["d", "a", "b", "c"]
    assert match.seats_for_game(specs, 3) == ["b", "c", "d", "a"]
    assert sorted(match.seats_for_game(specs, i)[2] for i in range(4)) == specs  # every spec sits in seat 2 once
    assert '<GO type="169" lobby="0"/>' in header(["model:x.pt", "greedy", "uniform", "a b"], 7)
    assert 'n3="a%20b"' in header(["model:x.pt", "greedy", "uniform", "a b"], 7)
