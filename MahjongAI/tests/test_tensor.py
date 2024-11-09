import sys

sys.path.append("../../")

import MahjongAI.utils.dataloader as dl
from MahjongAI.agents.transformer import TransformerModel
import MahjongAI.process_xml as px
from MahjongAI.turn import DuringTurn, DiscardTurn, PostTurn
from MahjongAI.state import StateObject
import torch
import numpy as np
import pytest


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model():
    return TransformerModel()


@pytest.fixture(scope="module")
def dataloader(model):
    return dl.DataLoader("data/")


def test_discard(dataloader):
    hand_tensor = np.zeros(37)
    hand_tensor[:14] = 1

    remaining_tiles = np.ones(37) * 3

    default_state_obj = StateObject(
        remaining_turns=70,
        hand_tensor=hand_tensor,
        remaining_tiles=remaining_tiles,
        remaining_tiles_pov=remaining_tiles,
        sutehai_tensor=np.zeros(37),
        is_menzen=[1, 1, 1, 1],
        double_reaches=[0, 0, 0, 0],
        reaches=[0, 0, 0, 0],
        ippatsu=[0, 0, 0, 0],
        melds=[[], [], [], []],
        scores=[25000, 25000, 25000, 25000],
        kyotaku=0,
        honba=0,
        dora=[0],
        parent_tensor=np.array([1, 0, 0, 0]),
        rounds_remaining=4,
        parent_rounds_remaining=[1, 1, 1, 1],
        remaining_tsumo=70,
    )
    encodings = torch.tensor(
        [
            px.discard2idx(0, 0, False),
            px.discard2idx(1, 1, True),
            px.discard2idx(2, 2, False),
            px.discard2idx(3, 3, True),
            px.discard2idx(0, 4, False),
            px.discard2idx(1, 5, True),
            px.discard2idx(2, 6, False),
            px.discard2idx(3, 7, True),
        ]
    ).to(device)

    halfturns = [
        DiscardTurn(
            0,
            default_state_obj,
            0,
            0,
        ),
        DiscardTurn(
            1,
            default_state_obj,
            4,
            1,
        ),
        DiscardTurn(
            2,
            default_state_obj,
            8,
            2,
        ),
        DiscardTurn(
            3,
            default_state_obj,
            12,
            3,
        ),
        DiscardTurn(
            0,
            default_state_obj,
            16,
            4,
        ),
        DiscardTurn(
            1,
            default_state_obj,
            20,
            5,
        ),
        DiscardTurn(
            2,
            default_state_obj,
            24,
            6,
        ),
    ]

    embeddings_tensor, state_obj_tensor_batch, action_mask_batch, y_tensor = (
        dataloader._build_discard_tensor(halfturns, encodings)
    )

    assert embeddings_tensor.shape == (7, 150)
    assert max(embeddings_tensor[0]) == min(embeddings_tensor[0]) == 0
    assert max(embeddings_tensor[1, 1:]) == min(embeddings_tensor[1, 1:]) == 0
    assert (
        max(embeddings_tensor[1:, 0])
        == min(embeddings_tensor[1:, 0])
        == px.discard2idx(0, 0, False)
    )
    assert (
        max(embeddings_tensor[2:, 1])
        == min(embeddings_tensor[2:, 1])
        == px.discard2idx(1, 1, True)
    )

    x1, x2, x3 = state_obj_tensor_batch
    assert x1.shape == torch.Size([7, 3, 37])
    assert x2.shape == torch.Size([7, 7, 4])
    assert x3.shape == torch.Size([7, 4])

    assert action_mask_batch.shape == (7, 37)
    assert (action_mask_batch[:, :14] == 1).all()
    assert (action_mask_batch[:, 14:] == 0).all()

    assert (y_tensor == torch.tensor([0, 1, 2, 3, 4, 5, 6]).to(device)).all().item()
