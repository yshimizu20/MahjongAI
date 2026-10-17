"""Unit test for the discard-head tensors of encoder_conv.TransformerTensorProcessor on a hand-built round:
seven discards in seat order 0, 1, 2, 3, 0, 1, 2, each of a plain (non-red) tile the actor holds."""
import numpy as np
import pytest
import torch
from mahjong.constants import EAST, NORTH, SOUTH

import MahjongAI.process_xml as px
from MahjongAI.agents.encoder_conv import N_X1_PLANES, TransformerTensorProcessor
from MahjongAI.state import StateObject
from MahjongAI.turn import DiscardTurn

SCORES = [250, 240, 230, 280]  # Tenhou "ten" units (x100 points); distinct per seat so a wrong POV shows in x2
DISCARDED = [0, 4, 8, 12, 17, 20, 24]  # 136-index tiles: 1m..7m, and 17 is a plain 5m (16 is the red one)
TSUMOGIRI = [False, True] * 4


def state_for(player: int, round_wind: int = EAST, rounds_remaining: int = 4) -> StateObject:
    hand = np.zeros(37)
    hand[:14] = 1  # one each of 1m..9m, 1p..5p: every discarded kind is held, no red five
    return StateObject(
        remaining_turns=70,
        hand_tensor=hand,
        remaining_tiles=np.full(37, 3.0),
        remaining_tiles_pov=np.full(37, 3.0),
        sutehai_tensors=np.zeros((4, 37)),
        is_menzen=[1, 1, 1, 1],
        double_reaches=[0, 0, 0, 0],
        reaches=[0, 0, 0, 0],
        ippatsu=[0, 0, 0, 0],
        melds=[[], [], [], []],
        scores=SCORES,
        kyotaku=0,
        honba=0,
        dora=[0],
        parent_tensor=np.array([1, 0, 0, 0]),
        rounds_remaining=rounds_remaining,
        parent_rounds_remaining=[1, 1, 1, 1],
        remaining_tsumo=70,
        player=player,
        round_wind=round_wind,
    )


def test_discard():
    n = len(DISCARDED)
    tokens = [px.discard2idx(j % 4, DISCARDED[j] // 4, TSUMOGIRI[j]) for j in range(n)]
    token_tensor = torch.nn.functional.pad(torch.tensor(tokens, dtype=torch.int64), (0, 150 - n))
    turns = [DiscardTurn(i % 4, state_for(i % 4), DISCARDED[i], i) for i in range(n)]

    tp = TransformerTensorProcessor()
    history, (x1, x2, x3), mask, y, padding = tp._build_discard_tensor(turns, token_tensor)

    assert not tp.skipped
    assert history.shape == (n, 150)
    assert x1.shape == (n, N_X1_PLANES, 37, 4)
    assert x2.shape == (n, 7, 4)
    assert x3.shape == (n, 7)
    assert mask.shape == (n, 37)

    for i in range(n):
        actor = i % 4
        # sample i sees exactly the i earlier discards (nothing from its own turn on), seats relative to the actor
        expected = [px.discard2idx((j % 4 - actor) % 4, DISCARDED[j] // 4, TSUMOGIRI[j]) for j in range(i)]
        assert history[i, :i].tolist() == expected
        assert (history[i, i:] == 0).all()
        assert padding[i].tolist() == [p < i for p in range(150)]
        # x2 row 0 is the score row rotated actor-first: column 0 is the actor's own score
        torch.testing.assert_close(x2[i, 0], torch.tensor([SCORES[(actor + k) % 4] / 250.0 for k in range(4)]))

    # legal mask = the held kinds (no riichi declaration, no call), and every target is inside it
    assert (mask[:, :14] == 1).all() and (mask[:, 14:] == 0).all()
    assert mask[torch.arange(n), y].all()
    assert y.tolist() == list(range(n))


def test_round_wind_one_hot():
    """Tonpuu E1 and hanchan S1 with the same scores, dealer and rounds_remaining (4) differ ONLY in the bakaze
    slots 4-6 of x3; a North round has no slot and fails loudly."""
    tp = TransformerTensorProcessor()
    t1, t2, t3 = tp._convert_state_obj_to_tensors(state_for(0, round_wind=EAST, rounds_remaining=4))  # tonpuu E1
    h1, h2, h3 = tp._convert_state_obj_to_tensors(state_for(0, round_wind=SOUTH, rounds_remaining=4))  # hanchan S1
    assert torch.equal(t1, h1) and torch.equal(t2, h2)
    assert torch.equal(t3[0, :4], h3[0, :4])
    assert t3[0, 4:].tolist() == [1.0, 0.0, 0.0]
    assert h3[0, 4:].tolist() == [0.0, 1.0, 0.0]
    with pytest.raises(AssertionError):
        tp._convert_state_obj_to_tensors(state_for(0, round_wind=NORTH))
