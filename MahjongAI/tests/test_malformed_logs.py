"""Malformed Tenhou logs (T-046): an own-turn declaration (kakan, ankan, riichi) right after the actor's own discard.
The parser rejects the game with a named InvalidGameException, gates.run counts and names it, and a kan by a seat
that neither drew nor just discarded still crashes on the label assert (the reject stays narrow)."""
import inspect
import os
import traceback

import pytest

from MahjongAI import process_xml
from MahjongAI.process_xml import InvalidGameException, process
from MahjongAI.utils import gates

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")
MSG = "Own-turn action after own discard"


def fixture_path(fixture):
    day, name, _ = fixture
    return os.path.join(GATE_GAMES, day[:4], day, f"{day}_00", name)


PATHS = [fixture_path(g) for g in gates.AFTER_OWN_DISCARD_LOGS]  # kakan, ankan, riichi
KAKAN = PATHS[0]  # d4d83e3e: seat 1 discards the red 5s (E88), then kakans it (m=34834)


@pytest.mark.parametrize("path", PATHS, ids=["kakan", "ankan", "riichi"])
def test_process_rejects(path):
    with pytest.raises(InvalidGameException) as e:
        process(path)
    assert e.value.msg == MSG


def test_gates_run_counts_and_names(capsys):
    C = gates.run(PATHS)
    assert (C["games"], C["skipped_invalid"], C["skipped_invalid_after_own_discard"], C["parse_error"]) == (0, 3, 3, 0)
    out = capsys.readouterr().out
    for p in PATHS:
        assert f"{MSG} rejected: {p}" in out


def test_kan_by_another_seat_still_asserts(monkeypatch):
    """The kakan moved to seat 2 (neither the drawer nor the last discarder): not this reject, the label assert."""
    orig = process_xml._get_rounds

    def patched(file_path):
        gameinfo, rounds = orig(file_path)
        hits = [ev for r in rounds for ev in r if ev["event"] == "N" and ev["attr"].get("m") == "34834"]
        if len(hits) != 1 or hits[0]["attr"]["who"] != "1":  # not an assert: it must not pass the test below
            raise RuntimeError(f"expected one kakan m=34834 by seat 1, got {[h['attr'] for h in hits]}")
        hits[0]["attr"]["who"] = "2"
        return gameinfo, rounds

    monkeypatch.setattr(process_xml, "_get_rounds", patched)
    with pytest.raises(AssertionError) as e:
        process(KAKAN)
    last = traceback.extract_tb(e.value.__traceback__)[-1]
    where = (os.path.basename(last.filename), last.name, last.lineno)
    assert where == ("process_xml.py", "_process_round", kan_label_assert_line())


def kan_label_assert_line():
    """1-based line of the ankan / kakan label assert in process_xml._process_round: the assert right after the reject call
    in the `if naki.is_ankan() or naki.is_kakan():` arm."""
    lines, start = inspect.getsourcelines(process_xml._process_round)
    i = next(k for k, line in enumerate(lines) if line.strip() == "if naki.is_ankan() or naki.is_kakan():")
    assert lines[i + 1].strip() == "_reject_after_own_discard(curr_halfturn, player)"
    assert lines[i + 2].strip() == "assert isinstance(curr_halfturn, DuringTurn) and curr_halfturn.player == player"
    return start + i + 2
