"""pytest entry for the pre-training gates: the same regression as `python -m MahjongAI.utils.gates --test`.
Needs the sample game and MahjongAI/data/processed/2021/20210101 (the self-test searches a few real games)."""
from MahjongAI.utils import gates


def test_gates_self_test():
    gates.self_test()
