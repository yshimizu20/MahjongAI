"""Round wind (bakaze, T-038) end to end on the committed gate fixtures: the parser's StateObjects reach x3 slots 4-6
with the wind of their round, a North round is rejected (parser and gates.run), and the model takes the 7-slot x3."""
import os

import pytest
import torch

from MahjongAI import process_xml
from MahjongAI.agents.encoder_conv import EncoderConvModel, TransformerTensorProcessor, device
from MahjongAI.process_xml import InvalidGameException, process
from MahjongAI.utils import gates

DAY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games", "2021", "20210101",
                       "20210101_00")
TONPUU = os.path.join(DAY_DIR, gates.TONPUU_SOUTH_LOG[1])  # tonpuu: E1-E4, S1, S2
WEST = os.path.join(DAY_DIR, gates.HANCHAN_WEST_LOG[1])  # hanchan, INIT seeds 0,0,1,2,3,3,4,5,6,7,8
E, S, W = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]


def round_winds(path):
    """Per round, the distinct x3[4:7] one-hots of every StateObject of that round."""
    tp = TransformerTensorProcessor()
    H, _ = process(path)
    return [sorted({tuple(tp._convert_state_obj_to_tensors(so)[2][0, 4:].tolist()) for so in gates._states([hts])})
            for hts in H]


def test_tonpuu_rounds_east_then_south():
    assert round_winds(TONPUU) == [[tuple(E)]] * 4 + [[tuple(S)]] * 2


def test_hanchan_rounds_east_south_west():
    assert round_winds(WEST) == [[tuple(E)]] * 6 + [[tuple(S)]] * 4 + [[tuple(W)]]


def last_round_seed(monkeypatch, curr_round):
    """process_xml._get_rounds returns the West fixture with the last round's INIT seed[0] set to curr_round."""
    orig = process_xml._get_rounds

    def patched(file_path):
        gameinfo, rounds = orig(file_path)
        attr = rounds[-1][0]["attr"]
        attr["seed"] = ",".join([str(curr_round)] + attr["seed"].split(",")[1:])
        return gameinfo, rounds

    monkeypatch.setattr(process_xml, "_get_rounds", patched)


def test_round_11_is_west(monkeypatch):
    last_round_seed(monkeypatch, 11)
    H, _ = process(WEST)
    assert all(so.round_wind == 29 for so in gates._states([H[-1]]))  # mahjong.constants.WEST


def test_round_12_rejected_as_north(monkeypatch):
    last_round_seed(monkeypatch, 12)
    with pytest.raises(InvalidGameException) as e:
        process(WEST)
    assert e.value.msg == "North round"


def test_gates_run_reports_north(monkeypatch, capsys):
    last_round_seed(monkeypatch, 12)
    C = gates.run([WEST])
    assert (C["games"], C["skipped_invalid"], C["skipped_invalid_north"]) == (0, 1, 1)
    assert f"North round rejected: {WEST}" in capsys.readouterr().out


def test_model_smoke_7_slot_x3():
    """The model runs one parsed tonpuu game end to end; fc_main_1 takes 1480 + 40 + 28 + 7 = 1555 features."""
    H, T = process(TONPUU)
    model = EncoderConvModel(verbose=False).to(device)  # forward moves the inputs to `device`
    assert model.state_net.fc_main_1.in_features == 1555
    out = model([H], [T], sample=False)
    assert set(out) == {"during", "discard", "post"}
    assert all(torch.isfinite(o.loss) for o in out.values())
