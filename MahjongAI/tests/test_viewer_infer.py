"""T-052 G1 on two fixture games: the viewer's label-free per-sample path (viewer/infer.py) reproduces the training
forward row for row (mask exact, finite masked logits within 1e-8 with both paths in float64; the float32 production
path within 1e-3 as a sanity bound), and the rows the training builders drop get the expected mask and finite logits.
A NaN on one row fails. A seeded random model suffices: parity is a property of the code path, not of the weights."""
import glob
import os

import pytest
import torch

from MahjongAI.agents.encoder_conv import EncoderConvModel, device
from MahjongAI.process_xml import process
from MahjongAI.viewer import infer, viewer_lib

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")
GAMES = ("2021010101gm-00a9-0000-6422c07e.log",  # chained kans, open-kan indicator, ron (tests/test_kan_dora.py)
         "2021010521gm-00a9-0000-e76b4550.log")  # triple ron, riichi discards


def fixture(name):
    found = glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", name))
    assert len(found) == 1, found
    return found[0]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(52)
    return EncoderConvModel(verbose=False).to(device).eval()


@pytest.mark.parametrize("name", GAMES)
def test_parity_with_the_training_forward_in_float64(model, name):
    all_halfturns, all_tokens = process(fixture(name))
    report = viewer_lib.check_parity(viewer_lib.double_model(model), all_halfturns, all_tokens)
    assert set(report) == {"during", "discard", "post"}
    assert all(r["n"] > 0 for r in report.values()), report
    assert all(r["max_abs_delta"] < viewer_lib.PARITY_TOL for r in report.values()), report


def test_float32_production_path_is_close(model):
    all_halfturns, all_tokens = process(fixture(GAMES[0]))
    report = viewer_lib.check_parity(model, all_halfturns, all_tokens, tol=1e-3)
    assert all(r["max_abs_delta"] < 1e-3 for r in report.values()), report


def test_parity_detects_a_nan_on_one_row(model, monkeypatch):
    all_halfturns, all_tokens = process(fixture(GAMES[0]))
    real, calls = infer.predict, []

    def nan_once(m, sample):
        mask, logits, probs = real(m, sample)
        calls.append(sample.head)
        if len(calls) == 3:
            logits = logits.clone()
            logits[mask > 0] = float("nan")
        return mask, logits, probs

    monkeypatch.setattr(infer, "predict", nan_once)
    with pytest.raises(viewer_lib.ViewerCheckError, match="non-finite legal logits"):
        viewer_lib.check_parity(model, all_halfturns, all_tokens, tol=1e-3)


@pytest.mark.parametrize("name", GAMES)
def test_viewer_only_rows(model, name):
    all_halfturns, all_tokens = process(fixture(name))
    counts = viewer_lib.check_viewer_only(model, all_halfturns, all_tokens)
    assert counts["forced_riichi_discard"] > 0, counts  # both games have riichi discards
    assert counts["no_option_during"] > 0, counts


def test_build_sample_returns_none_for_a_seat_without_a_decision(model):
    all_halfturns, all_tokens = process(fixture(GAMES[0]))
    turn, tokens = all_halfturns[0][0], all_tokens[0]
    assert infer.build_sample(turn, tokens, (turn.player + 1) % 4) is None
    sample = infer.build_sample(turn, tokens, turn.player)
    assert sample.head == "during" and sample.mask.shape == (1, 72) and sample.mask[0, 0] == 1.0


def test_parity_detects_a_wrong_mask(model, monkeypatch):
    all_halfturns, all_tokens = process(fixture(GAMES[0]))
    real = infer.during_mask

    def wrong(turn):
        m = real(turn)
        m[71] = 1.0  # an abort offered everywhere
        return m

    monkeypatch.setattr(infer, "during_mask", wrong)
    with pytest.raises(viewer_lib.ViewerCheckError, match="mask differs"):
        viewer_lib.check_parity(model, all_halfturns, all_tokens)


def test_parity_detects_a_wrong_history_cut(model, monkeypatch):
    all_halfturns, all_tokens = process(fixture(GAMES[0]))
    real = infer._history

    def one_token_short(tokens, encoding_idx, seat):
        return real(tokens, max(0, encoding_idx - 1), seat)

    monkeypatch.setattr(infer, "_history", one_token_short)
    with pytest.raises(viewer_lib.ViewerCheckError, match="delta logits"):
        viewer_lib.check_parity(model, all_halfturns, all_tokens)
