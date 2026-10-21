"""T-053 engine, S4: a model's identity is its checkpoint's sha256, never its file name; meta.json `seats` holds the
complete per-seat specification and players.from_description rebuilds exactly those players (S10)."""
import os

import pytest
import torch

from MahjongAI.agents.encoder_conv import EncoderConvModel
from MahjongAI.engine import match
from MahjongAI.engine.players import GreedyPlayer, ModelPlayer, UniformPlayer, from_description, make_player


def _save(path: str, seed: int) -> str:
    torch.manual_seed(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state_dict": EncoderConvModel(verbose=False).state_dict()}, path)
    return path


@pytest.fixture(scope="module")
def two_checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    return _save(str(root / "run-A" / "checkpoints" / "step_100.pt"), 1), \
        _save(str(root / "run-B" / "checkpoints" / "step_100.pt"), 2)


def test_same_file_name_different_identity(two_checkpoints):
    a, b = (ModelPlayer(p, 0.5, seed=7) for p in two_checkpoints)
    assert a.checkpoint_sha256 != b.checkpoint_sha256
    assert a.label != b.label and a.label.startswith("model:step_100.pt@") and b.label.startswith("model:step_100.pt@")
    assert a.label.endswith(a.checkpoint_sha256[:12])
    assert make_player("model:" + two_checkpoints[0], 0.5, 7).label == a.label
    assert a.describe() == {"kind": "model", "label": a.label, "checkpoint": os.path.abspath(two_checkpoints[0]),
                            "checkpoint_sha256": a.checkpoint_sha256, "temperature": 0.5, "seed": 7}


def test_match_summary_never_merges_the_two(two_checkpoints):
    a, b = (ModelPlayer(p) for p in two_checkpoints)
    rows = [{"labels": [a.label, b.label, "greedy", "uniform"], "placements": [1, 4, 2, 3],
             "final_scores": [400, 100, 300, 200], "rounds": 8, "wins": [3, 0, 1, 1], "deal_ins": [0, 3, 1, 1],
             "n_decisions": 500, "duration_s": 1.0},
            {"labels": [b.label, a.label, "uniform", "greedy"], "placements": [4, 1, 3, 2],
             "final_scores": [100, 400, 200, 300], "rounds": 8, "wins": [0, 3, 1, 1], "deal_ins": [3, 0, 1, 1],
             "n_decisions": 500, "duration_s": 1.0}]
    summary = match.summarize(rows)
    assert set(summary["labels"]) == {a.label, b.label, "greedy", "uniform"}
    assert summary["labels"][a.label]["mean_placement"] == 1.0 and summary["labels"][b.label]["mean_placement"] == 4.0


def test_from_description_rebuilds_each_seat_exactly(two_checkpoints):
    players = [ModelPlayer(two_checkpoints[0], 0.0, 3), GreedyPlayer(), UniformPlayer(3),
               ModelPlayer(two_checkpoints[1], 0.5, 3)]
    rebuilt = [from_description(p.describe()) for p in players]
    assert [type(p) for p in rebuilt] == [type(p) for p in players]
    assert [p.describe() for p in rebuilt] == [p.describe() for p in players]
    assert rebuilt[0].checkpoint_sha256 != rebuilt[3].checkpoint_sha256 and rebuilt[3].temperature == 0.5


def test_from_description_refuses_a_changed_or_missing_checkpoint(two_checkpoints):
    d = ModelPlayer(two_checkpoints[0]).describe()
    wrong = dict(d, checkpoint=two_checkpoints[1])  # the other run's step_100.pt under this seat's sha
    with pytest.raises(ValueError, match="sha256"):
        from_description(wrong)
    with pytest.raises(FileNotFoundError):
        from_description(dict(d, checkpoint=two_checkpoints[0] + ".missing"))
