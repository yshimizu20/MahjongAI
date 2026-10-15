"""pytest entry for the pre-training gates: the same regression as `python -m MahjongAI.utils.gates --test`.

Runs on the committed fixture games in tests/data/gate_games/ (same <yyyy>/<day>/<day>_00/ layout as
data/processed/), so it needs no local data. They are exactly the processed/ files the self-test reads, so its
sorted picks are unchanged. When self_test() starts reading a new game, re-trace the set against the full tree:
wrap builtins.open, run gates.self_test() with the default DATA_ROOT, and copy every opened processed/ file here."""
import os

from MahjongAI.utils import gates

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")


def test_gates_self_test(monkeypatch):
    monkeypatch.setattr(gates, "DATA_ROOT", GATE_GAMES)
    gates.self_test()
